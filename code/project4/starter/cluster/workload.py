"""生成请求。每个请求由若干个 KV 块组成：前面是可能被共享的前缀块，后面是这个请求独有的块。"""
import math
import random
from dataclasses import dataclass, field

BLOCK = 16


@dataclass
class Request:
    rid: int
    arrival: float
    input_len: int
    output_len: int
    tenant: int = 0
    prefix_id: int = -1            # 共享前缀的编号；-1 表示没有共享前缀
    prefix_len: int = 0            # 共享前缀的长度（token），是 BLOCK 的整数倍
    # 运行时由模拟器填写
    first_token: float = None
    token_times: list = field(default_factory=list)
    finish: float = None
    cached_tokens: int = 0         # 命中前缀缓存的 token 数
    instance: int = -1

    def block_keys(self):
        """这个请求的 prompt 按 BLOCK 切分后，每一块的键。前缀块的键只取决于前缀和块号，
        所以不同请求的同一前缀会得到相同的键；之后的块是这个请求独有的。
        真实的引擎用链式哈希（1.4 节），效果相同。"""
        if getattr(self, "_keys", None) is None:
            n_prefix = self.prefix_len // BLOCK
            keys = [("p", self.prefix_id, i) for i in range(n_prefix)]
            keys += [("u", self.rid, i) for i in range(n_prefix, self.input_len // BLOCK)]
            self._keys = keys
        return self._keys


def multi_tenant(n: int, rate: float, n_tenants: int = 16, prefix_len: int = 2048, unique_len=(128, 512),
                 output_len=(64, 256), zipf: float = 1.0, seed: int = 0):
    """多租户的聊天类负载：每个租户有一个固定的长前缀（系统提示词、文档），请求在其后追加自己的问题。
    租户的流量服从 Zipf 分布，少数租户占了大部分请求。到达过程为泊松过程，rate 是每秒请求数。"""
    rng = random.Random(seed)
    weights = [1 / (k + 1) ** zipf for k in range(n_tenants)]
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        tenant = rng.choices(range(n_tenants), weights)[0]
        u = rng.randint(*unique_len)
        reqs.append(Request(i, t, prefix_len + u, rng.randint(*output_len), tenant, tenant, prefix_len))
    return reqs


def long_prompt(n: int, rate: float, input_len=(2048, 8192), output_len=(128, 512), seed: int = 0):
    """没有共享前缀、输入较长的负载（例如长文档问答），用于 PD 分离的实验。"""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        reqs.append(Request(i, t, rng.randint(*input_len), rng.randint(*output_len)))
    return reqs


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
