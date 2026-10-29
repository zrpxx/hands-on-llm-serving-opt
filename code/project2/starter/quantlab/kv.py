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
    # TODO(第 6 步)：t 的形状是 [tokens, heads, head_dim]。fake_quant_int 总是沿最后一维分组，
    # 所以按 token 量化直接调用即可；按通道量化要先把 tokens 换到最后一维，量化后再换回来。整数用非对称量化。
    raise NotImplementedError("TODO(第 6 步): fake_quant_kv")
