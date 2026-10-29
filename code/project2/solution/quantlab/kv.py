"""KV Cache 量化。对应 2.5 节。

KIVI（Liu et al., ICML 2024）观察到：Key 有少数通道数值特别大，适合按通道量化；
Value 没有明显的通道离群值，按 token 量化即可。
这里的输入形状是 [tokens, heads, head_dim]。
"""
import torch

from .formats import fake_quant_int, fake_quant_fp8


def fake_quant_kv(t: torch.Tensor, bits: int = 8, axis: str = "token", fmt: str = "int") -> torch.Tensor:
    """axis="token"：每个 token 每个头的 head_dim 个数共用一个 scale；
    axis="channel"：每个头的每个通道，在所有 token 上共用一个 scale。"""
    if fmt == "fp8":
        if axis == "token":
            return fake_quant_fp8(t, "row")
        return fake_quant_fp8(t.transpose(0, 2), "row").transpose(0, 2)
    if axis == "token":
        return fake_quant_int(t, bits, -1, symmetric=False)
    if axis == "channel":
        tt = t.permute(1, 2, 0)                       # [heads, head_dim, tokens]
        return fake_quant_int(tt, bits, -1, symmetric=False).permute(2, 0, 1)
    raise ValueError(axis)
