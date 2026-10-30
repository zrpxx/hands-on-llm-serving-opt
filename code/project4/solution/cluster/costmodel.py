"""每一步的耗时估算。对应 0.7 节的 Roofline。

一步 = 一次前向：读一遍权重、计算本步所有 token、读本步所有序列的 KV。
    耗时 = max(读权重, 线性层计算) + 读 KV + 固定开销
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    mem_gb: float          # 显存，GB
    bw_tbs: float          # 显存带宽，TB/s
    tflops: float          # BF16 稠密算力，TFLOPS
    link_gbs: float        # 与其他实例之间传 KV 的带宽，GB/s（单向）


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float        # 参数量，十亿
    layers: int
    kv_heads: int
    head_dim: int

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * 2

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * 2


H100 = Hardware("H100 SXM", 80, 3.35, 989, 50)          # 实例之间按 400 Gb/s 的 RDMA 网卡计
L40S = Hardware("L40S", 48, 0.864, 362, 50)
QWEN3_8B = ModelSpec("Qwen3-8B", 8.2, 36, 8, 128)

BW_EFF, FLOPS_EFF, STEP_OVERHEAD = 0.8, 0.6, 1e-3


def kv_capacity_tokens(hw: Hardware, model: ModelSpec, util: float = 0.9, reserve_gb: float = 2.0) -> int:
    free = hw.mem_gb * 1e9 * util - model.weight_bytes - reserve_gb * 1e9
    return max(0, int(free // model.kv_bytes_per_token))


def step_time(hw: Hardware, model: ModelSpec, prefill_tokens: int, decode_seqs: int, kv_tokens: int) -> float:
    """返回秒。prefill_tokens：本步 prefill 的 token 数；decode_seqs：本步 decode 的序列数；
    kv_tokens：本步参与计算的所有序列已有的 KV token 总数。"""
    bw = hw.bw_tbs * 1e12 * BW_EFF
    fl = hw.tflops * 1e12 * FLOPS_EFF
    n = prefill_tokens + decode_seqs
    t_linear = max(model.weight_bytes / bw, 2 * model.params_b * 1e9 * n / fl)
    t_kv = kv_tokens * model.kv_bytes_per_token / bw
    return t_linear + t_kv + STEP_OVERHEAD
