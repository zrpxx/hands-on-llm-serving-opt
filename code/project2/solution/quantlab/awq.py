"""AWQ：按激活的幅度对输入通道做缩放，保护重要的权重，再按组量化。对应 2.3 节。

参考 Lin et al., AWQ (MLSys 2024)。这里只实现缩放搜索；原论文还会再搜索一次权重裁剪范围。
"""
import torch

from .formats import fake_quant_int


def awq_search(weight: torch.Tensor, x: torch.Tensor, bits: int = 4, group_size: int = 128,
               n_grid: int = 20, symmetric: bool = True):
    """weight: [out, in]，x: [N, in] 校准输入。

    对 α ∈ {0, 1/n, …, (n-1)/n}，令 s = mean|x|^α，量化 W·diag(s)，再除回 s，
    选使输出误差 ||X Wqᵀ − X Wᵀ||² 最小的 α。α = 0 就是普通的 RTN，所以结果不会比 RTN 差。
    返回 (伪量化后的等效权重, 最优 α, 对应的误差列表)。
    """
    W = weight.float()
    X = x.float()
    y = X @ W.T
    x_mean = X.abs().mean(dim=0).clamp(min=1e-6)
    best = (float("inf"), None, None)
    errors = []
    for k in range(n_grid):
        alpha = k / n_grid
        s = x_mean.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()          # 归一化，避免整体放大或缩小
        wq = fake_quant_int(W * s, bits, group_size, symmetric) / s
        err = (X @ wq.T - y).pow(2).mean().item()
        errors.append(err)
        if err < best[0]:
            best = (err, alpha, wq)
    return best[2].to(weight.dtype), best[1], errors
