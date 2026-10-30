"""分步测试。每一步完成后运行：python -m pytest tests/test_steps.py -k step01"""
import pytest

from conftest import FakeInstance


# ------------------------------------------------------------------ 第 1 步：前缀缓存
def test_step01_match_is_prefix_only():
    from cluster.prefix_cache import PrefixCache
    c = PrefixCache(100)
    c.acquire(["a", "b", "c"])
    assert c.match(["a", "b", "c", "d"]) == 3
    assert c.match(["a", "x", "c"]) == 1          # 第二块不同，之后的块即使存在也不算
    assert c.match(["x", "a"]) == 0


def test_step01_lru_eviction_skips_blocks_in_use():
    from cluster.prefix_cache import PrefixCache
    c = PrefixCache(4)
    c.acquire(["a", "b"]); c.release(["a", "b"])  # a、b 空闲，a 最久未使用
    c.acquire(["c", "d"])                          # c、d 正在使用
    c.match(["a"])                                 # 访问 a，现在 b 是最久未使用的空闲块
    c.acquire(["e"])                               # 满了，要淘汰一块：应该是 b
    assert "b" not in c and "a" in c and "c" in c and "d" in c and "e" in c
    c.acquire(["f", "g"])                          # 再淘汰两块：只有 a 空闲，c、d、e 都在使用
    assert "a" not in c and "c" in c and "d" in c and len(c) == 5   # 暂时超出容量


def test_step01_release_keeps_blocks_for_reuse():
    from cluster.prefix_cache import PrefixCache
    c = PrefixCache(10)
    c.acquire(["p0", "p1", "u0"]); c.release(["p0", "p1", "u0"])
    assert c.match(["p0", "p1", "v0"]) == 2


# ------------------------------------------------------------------ 第 2 步：基本路由
def test_step02_round_robin():
    from cluster.router import RoundRobin
    r = RoundRobin()
    insts = [FakeInstance() for _ in range(3)]
    assert [r.choose(None, insts) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_step02_least_load_uses_tokens_not_connections():
    from cluster.router import LeastLoad
    insts = [FakeInstance(queued=9000, n=1), FakeInstance(queued=800, n=5), FakeInstance(queued=3000, n=2)]
    assert LeastLoad().choose(None, insts) == 1


# ------------------------------------------------------------------ 第 3 步：前缀路由
class _Req:
    def __init__(self, prefix_id, rid=0):
        self.prefix_id, self.rid = prefix_id, rid


def test_step03_consistent_hash_is_stable_and_balanced():
    from cluster.router import PrefixHash
    h4 = PrefixHash(4)
    a = [h4.choose(_Req(k), [None] * 4) for k in range(2000)]
    assert a == [h4.choose(_Req(k), [None] * 4) for k in range(2000)]       # 同一前缀总是同一个实例
    share = [a.count(i) / len(a) for i in range(4)]
    assert min(share) > 0.15 and max(share) < 0.35
    h5 = PrefixHash(5)
    moved = sum(x != h5.choose(_Req(k), [None] * 5) for k, x in enumerate(a)) / len(a)
    assert 0.1 < moved < 0.35                                                # 加一个实例，只有约 1/5 换了位置


def test_step03_cache_aware_prefers_hits_until_imbalanced():
    from cluster.router import CacheAware
    r = CacheAware(balance_abs=10000, balance_rel=1.5, min_hit=256)
    insts = [FakeInstance(queued=5000, hit=0), FakeInstance(queued=6000, hit=4096), FakeInstance(queued=4000, hit=1024)]
    assert r.choose(_Req(1), insts) == 1                                     # 负载接近，选命中最多的
    insts = [FakeInstance(queued=2000, hit=0), FakeInstance(queued=40000, hit=4096), FakeInstance(queued=3000, hit=0)]
    assert r.choose(_Req(1), insts) == 0                                     # 严重失衡，选负载最低的
    insts = [FakeInstance(queued=5000, hit=100), FakeInstance(queued=3000, hit=0)]
    assert r.choose(_Req(1), insts) == 1                                     # 都没什么命中，按负载


# ------------------------------------------------------------------ 第 4 步：PD 分离
def test_step04_kv_transfer_time():
    from cluster.costmodel import QWEN3_8B
    from cluster.pd import kv_transfer_time
    t = kv_transfer_time(4096, QWEN3_8B, link_gbs=50, layerwise=False, latency=0)
    assert t == pytest.approx(4096 * 147456 / 50e9)                          # 约 12 ms
    t2 = kv_transfer_time(4096, QWEN3_8B, link_gbs=50, layerwise=True, latency=0)
    assert t2 == pytest.approx(t / 36)


def test_step04_disaggregated_cluster_runs_and_isolates_decode():
    from cluster.instance import Instance
    from cluster.pd import DisaggCluster
    from cluster.router import LeastLoad
    from cluster.sim import Cluster
    from cluster.workload import long_prompt
    agg = Cluster([Instance(i) for i in range(4)], LeastLoad()).run(long_prompt(300, 6, seed=3))
    reqs = long_prompt(300, 6, seed=3)
    pd = DisaggCluster([Instance(i, role="prefill") for i in range(2)],
                       [Instance(10 + i, role="decode") for i in range(2)], LeastLoad()).run(reqs)
    assert pd["完成"] == 300
    assert all(r.instance >= 10 for r in reqs)                               # 每个请求最后都在 decode 实例上完成
    assert pd["ITL_P99"] < 0.5 * agg["ITL_P99"]                              # decode 不再被 prefill 打断


# ------------------------------------------------------------------ 第 5 步：goodput 与端到端
def test_step05_goodput():
    from cluster.sim import goodput
    from cluster.workload import Request

    def mk(arr, first, times):
        r = Request(0, arr, 100, len(times))
        r.first_token, r.token_times, r.finish = first, times, times[-1]
        return r
    reqs = [mk(0, 0.5, [0.5, 0.52, 0.54]), mk(0, 3.0, [3.0, 3.01]), mk(0, 0.2, [0.2, 0.4, 0.6])]
    res = {"_requests": reqs, "总数": 4}                                     # 第 4 个请求没有完成
    assert goodput(res, ttft_slo=1.0, tpot_slo=0.05) == pytest.approx(1 / 4)


def test_step05_cache_aware_beats_round_robin():
    from cluster.instance import Instance
    from cluster.router import CacheAware, RoundRobin
    from cluster.sim import Cluster
    from cluster.workload import multi_tenant
    rr = Cluster([Instance(i) for i in range(4)], RoundRobin()).run(multi_tenant(800, 24, n_tenants=128, prefix_len=4096, seed=1))
    ca = Cluster([Instance(i) for i in range(4)], CacheAware()).run(multi_tenant(800, 24, n_tenants=128, prefix_len=4096, seed=1))
    assert ca["命中率"] > rr["命中率"] + 0.1
