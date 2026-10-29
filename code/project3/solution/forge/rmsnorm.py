"""第 1 步：融合的 RMSNorm（可选地先加上残差）。对应 3.3 节。

不融合时，"加残差、求均方、乘 rsqrt、乘权重"是四五个算子，每个都要把整行读写一遍显存。
融合之后，每行只读一次输入、写一次输出。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, D, eps,
                    HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # 每个 program 处理一行
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        x += tl.load(res_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(res_out_ptr + row * D + cols, x, mask=mask)   # 加完残差的结果，供下一层作为残差
    var = tl.sum(x * x, axis=0) / D
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(var + eps) * w
    tl.store(out_ptr + row * D + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, residual: torch.Tensor = None):
    """x: [N, D]。返回 (输出, 新的残差)。没有 residual 时第二个返回值为 None。"""
    assert x.is_contiguous() and x.dim() == 2
    N, D = x.shape
    out = torch.empty_like(x)
    res_out = torch.empty_like(x) if residual is not None else out
    BLOCK = triton.next_power_of_2(D)
    _rmsnorm_kernel[(N,)](x, residual if residual is not None else x, weight, out, res_out, D, eps,
                          HAS_RES=residual is not None, BLOCK=BLOCK)
    return out, (res_out if residual is not None else None)


def rmsnorm_ref(x, weight, eps=1e-6, residual=None):
    if residual is not None:
        x = x + residual
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return y.to(x.dtype), (x if residual is not None else None)
