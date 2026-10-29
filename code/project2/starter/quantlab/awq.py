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
    # TODO(第 4 步)：网格搜索 α，返回 (误差最小的等效权重, 对应的 α, 每个 α 的误差列表)
    # s = mean|x|^α，再除以 sqrt(max(s)·min(s)) 归一化；等效权重 = fake_quant_int(W·s, ...) / s
    raise NotImplementedError("TODO(第 4 步): awq_search")
