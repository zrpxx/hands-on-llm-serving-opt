"""第 3 步：在线 softmax。对应 3.2 节。

普通的 softmax 要读三遍输入：求最大值、求指数和、归一化。在线 softmax 在一遍扫描中同时维护
当前的最大值 m 和指数和 l：遇到更大的值时，把已经累加的 l 乘以 exp(m_旧 - m_新) 修正。
这是 FlashAttention 的核心技巧。这里一行可能比 BLOCK 长得多，所以要分块循环。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(x_ptr, out_ptr, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row * D
    m = tl.full((), float("-inf"), tl.float32)  # 目前为止的最大值
    l = tl.zeros((), tl.float32)                 # 目前为止的 sum(exp(x - m))
    for start in range(0, D, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < D, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    for start in range(0, D, BLOCK):             # 第二遍：写出归一化的结果
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < D, other=float("-inf")).to(tl.float32)
        tl.store(out_ptr + row * D + cols, tl.exp(x - m) / l, mask=cols < D)


def online_softmax(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """x: [N, D]，沿最后一维做 softmax。"""
    assert x.is_contiguous() and x.dim() == 2
    out = torch.empty_like(x, dtype=torch.float32)
    _online_softmax_kernel[(x.shape[0],)](x, out, x.shape[1], BLOCK=block)
    return out.to(x.dtype)
