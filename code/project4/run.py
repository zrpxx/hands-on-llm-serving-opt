"""项目 4 的实验。全部在 CPU 上运行，每组几秒钟。

    python run.py routers                 # 比较四种路由策略（4.6 节）
    python run.py pd --rate 6             # 聚合部署与 PD 分离（4.3 节）
    python run.py pd --link 5 --no-layerwise   # 实例之间只有 5 GB/s、且不逐层传输时
    python run.py ratio                   # 不同的 P:D 配比

默认使用 starter/ 中你的实现；设置 CLUSTER_IMPL=solution 使用参考实现。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("CLUSTER_IMPL", "starter")))
from cluster.costmodel import H100, Hardware  # noqa: E402
from cluster.instance import Instance  # noqa: E402
from cluster.pd import DisaggCluster  # noqa: E402
from cluster.router import CacheAware, LeastLoad, PrefixHash, RoundRobin  # noqa: E402
from cluster.sim import Cluster, goodput  # noqa: E402
from cluster.workload import long_prompt, multi_tenant  # noqa: E402

ms = lambda x: f"{x * 1e3:7.0f}"
ms1 = lambda x: f"{x * 1e3:6.1f}"


def routers(a):
    print(f"{a.instances} 个实例，{a.tenants} 个租户，每个前缀 {a.prefix} token，{a.rate} 请求/秒，共 {a.n} 个请求\n")
    print(f"{'策略':<14}{'命中率':>8}{'TTFT P50':>10}{'TTFT P99':>10}{'ITL P99':>9}   各实例的请求数")
    for name, mk in [("轮询", lambda: RoundRobin()), ("最少负载", lambda: LeastLoad()),
                     ("一致性哈希", lambda: PrefixHash(a.instances)), ("缓存感知", lambda: CacheAware())]:
        reqs = multi_tenant(a.n, a.rate, n_tenants=a.tenants, prefix_len=a.prefix, seed=a.seed)
        res = Cluster([Instance(i) for i in range(a.instances)], mk()).run(reqs)
        counts = [sum(1 for r in res["_requests"] if r.instance == i) for i in range(a.instances)]
        print(f"{name:<12}{res['命中率']:>9.2f}{ms(res['TTFT_P50'])} ms{ms(res['TTFT_P99'])} ms{ms1(res['ITL_P99'])} ms   {counts}")


def _pd_row(name, res, a):
    print(f"{name:<12}{ms(res['TTFT_P50'])} ms{ms(res['TTFT_P99'])} ms{ms1(res['ITL_P50'])} ms{ms1(res['ITL_P99'])} ms"
          f"{ms(res['第二个token_P99'])} ms{goodput(res, a.ttft_slo, a.tpot_slo):>9.2f}")


def pd(a):
    hw = Hardware(H100.name, H100.mem_gb, H100.bw_tbs, H100.tflops, a.link)
    print(f"4 张 H100，长输入负载（2K～8K token），{a.rate} 请求/秒；实例之间 {a.link} GB/s，"
          f"{'逐层传输' if a.layerwise else '整体传输'}；goodput 的目标：TTFT ≤ {a.ttft_slo}s，TPOT ≤ {a.tpot_slo * 1e3:.0f}ms\n")
    print(f"{'配置':<12}{'TTFT P50':>10}{'TTFT P99':>11}{'ITL P50':>10}{'ITL P99':>10}{'第 2 个 token':>13}{'goodput':>9}")
    res = Cluster([Instance(i, hw) for i in range(4)], LeastLoad()).run(long_prompt(a.n, a.rate, seed=a.seed))
    _pd_row("聚合 ×4", res, a)
    for p in (1, 2, 3):
        c = DisaggCluster([Instance(i, hw, role="prefill") for i in range(p)],
                          [Instance(10 + i, hw, role="decode") for i in range(4 - p)], LeastLoad(), layerwise=a.layerwise)
        res = c.run(long_prompt(a.n, a.rate, seed=a.seed))
        _pd_row(f"{p}P{4 - p}D", res, a)
    print(f"\n「第 2 个 token」是第一个与第二个 token 之间间隔的 P99，PD 分离时包含了 KV 传输。"
          f"最后一种配置平均每个请求的 KV 传输暴露时间：{sum(c.transfer_times) / len(c.transfer_times) * 1e3:.1f} ms")


def ratio(a):
    for rate in (2, 4, 6, 8):
        a.rate = rate
        pd(a)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["routers", "pd", "ratio"])
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--tenants", type=int, default=128)
    ap.add_argument("--prefix", type=int, default=4096)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--link", type=float, default=50.0, help="实例之间传 KV 的带宽，GB/s")
    ap.add_argument("--no-layerwise", dest="layerwise", action="store_false")
    ap.add_argument("--ttft-slo", type=float, default=1.0)
    ap.add_argument("--tpot-slo", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.exp == "routers":
        a.rate, a.n = a.rate or 32, a.n or 2000
        routers(a)
    else:
        a.rate, a.n = a.rate or 6, a.n or 1000
        pd(a) if a.exp == "pd" else ratio(a)


if __name__ == "__main__":
    main()
