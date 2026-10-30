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
        # TODO(第 2 步)：依次返回 0, 1, ..., n-1, 0, 1, ...
        raise NotImplementedError("TODO(第 2 步): RoundRobin.choose")


class LeastLoad:
    """选择排队工作量（还要处理的 token 数）最少的实例。只看连接数的"最少连接"忽略了请求的长短。"""

    def choose(self, req, instances) -> int:
        # TODO(第 2 步)：返回 queued_tokens() 最小的实例下标，并列时取下标小的
        raise NotImplementedError("TODO(第 2 步): LeastLoad.choose")


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
        # TODO(第 3 步)：在环上找到第一个哈希值大于 _h(key) 的虚拟节点（用 bisect），越过末尾时回到开头，
        # 返回它对应的实例下标。self.points 是排好序的虚拟节点哈希值，self.ring 是 (哈希值, 实例下标)。
        raise NotImplementedError("TODO(第 3 步): PrefixHash.lookup")

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
        # TODO(第 3 步)：
        # 1. 取各实例的 queued_tokens()；如果最大与最小之差 > balance_abs，且最大 > balance_rel × max(最小, 1)，
        #    返回负载最低的实例；
        # 2. 否则取各实例的 cached_prefix_tokens(req)；最大命中 < min_hit 时也返回负载最低的实例；
        # 3. 否则在命中最多的实例中，返回负载最低的那个。
        raise NotImplementedError("TODO(第 3 步): CacheAware.choose")
