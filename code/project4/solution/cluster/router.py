"""第 2、3 步：路由策略。对应 4.6 节。

每个路由器实现 choose(req, instances) -> 实例下标。instances 是 Instance 的列表，
可以读取它们的负载信号：queued_tokens()、num_requests()、cached_prefix_tokens(req)。
"""
import bisect
import hashlib


class RoundRobin:
    """轮流分配，不看负载。"""

    def __init__(self):
        self.i = 0

    def choose(self, req, instances) -> int:
        k = self.i % len(instances)
        self.i += 1
        return k


class LeastLoad:
    """选择排队工作量（还要处理的 token 数）最少的实例。只看连接数的"最少连接"忽略了请求的长短。"""

    def choose(self, req, instances) -> int:
        return min(range(len(instances)), key=lambda i: (instances[i].queued_tokens(), i))


def _h(s) -> int:
    return int.from_bytes(hashlib.md5(str(s).encode()).digest()[:8], "big")


class PrefixHash:
    """一致性哈希：按请求的前缀（这里用 prefix_id，没有共享前缀时用请求编号）选择实例。
    同一前缀总是去同一个实例，缓存命中率高；增减实例时，只有约 1/N 的前缀需要换实例。
    每个实例在环上放 vnodes 个虚拟节点，让分布更均匀。"""

    def __init__(self, n_instances: int, vnodes: int = 64):
        self.ring = sorted((_h(f"inst{i}-{v}"), i) for i in range(n_instances) for v in range(vnodes))
        self.points = [p for p, _ in self.ring]

    def lookup(self, key) -> int:
        k = bisect.bisect(self.points, _h(key)) % len(self.ring)
        return self.ring[k][1]

    def choose(self, req, instances) -> int:
        key = req.prefix_id if req.prefix_id >= 0 else f"r{req.rid}"
        return self.lookup(key)


class CacheAware:
    """缓存感知路由，思路与 SGLang 的路由器相同：
    - 如果各实例的负载差距不大，选择能命中最多前缀 token 的实例（并列时选负载低的）；
    - 如果负载已经明显失衡（最大与最小之差超过 balance_abs，且最大超过最小的 balance_rel 倍），
      不再追求命中，直接选负载最低的实例，防止热门前缀把一个实例压垮。"""

    def __init__(self, balance_abs: int = 32768, balance_rel: float = 1.5, min_hit: int = 256):
        self.balance_abs, self.balance_rel, self.min_hit = balance_abs, balance_rel, min_hit

    def choose(self, req, instances) -> int:
        loads = [inst.queued_tokens() for inst in instances]
        lo, hi = min(loads), max(loads)
        if hi - lo > self.balance_abs and hi > self.balance_rel * max(lo, 1):
            return loads.index(lo)
        hits = [inst.cached_prefix_tokens(req) for inst in instances]
        best = max(hits)
        if best < self.min_hit:                     # 哪里都命中不了多少，按负载分
            return loads.index(lo)
        cands = [i for i, h in enumerate(hits) if h == best]
        return min(cands, key=lambda i: loads[i])
