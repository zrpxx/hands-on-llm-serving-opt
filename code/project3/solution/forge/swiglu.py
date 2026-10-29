"""第 2 步：融合的 SwiGLU 激活。对应 3.3 节。

vLLM 等引擎把 gate_proj 和 up_proj 合并成一个矩阵乘，输出 [N, 2I]：前 I 列是 gate，后 I 列是 up。
激活函数 silu(gate) * up 如果用 PyTorch 写，要经过切片、sigmoid、两次乘法，中间结果都会写回显存。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, I, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pid = tl.program_id(1)                       # 一行可能很长，按 BLOCK 列一块切开
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(x_ptr + row * 2 * I + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * I + I + cols, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + row * I + cols, y, mask=mask)


def swiglu(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """x: [N, 2I]，返回 [N, I]。"""
    assert x.is_contiguous() and x.dim() == 2 and x.shape[1] % 2 == 0
    N, I2 = x.shape
    I = I2 // 2
    out = torch.empty((N, I), dtype=x.dtype, device=x.device)
    _swiglu_kernel[(N, triton.cdiv(I, block))](x, out, I, BLOCK=block)
    return out


def swiglu_ref(x):
    g, u = x.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(g) * u).to(x.dtype)
