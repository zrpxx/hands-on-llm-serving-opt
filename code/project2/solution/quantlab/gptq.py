"""GPTQ：用校准数据的二阶信息，逐列量化并把误差补偿到尚未量化的列上。对应 2.2 节。

参考 Frantar et al., GPTQ (ICLR 2023) 及其官方实现。
"""
import torch

from .formats import int_qparams, int_quant_dequant


def hessian_from_inputs(x: torch.Tensor) -> torch.Tensor:
    """x: [N, in]。返回 H = 2/N · Xᵀ X，形状 [in, in]。"""
    x = x.float()
    return 2.0 / x.shape[0] * x.T @ x


def gptq_quantize(weight: torch.Tensor, H: torch.Tensor, bits: int = 4, group_size: int = 128,
                  symmetric: bool = True, damp: float = 0.01, act_order: bool = False) -> torch.Tensor:
    """返回伪量化后的权重（与 weight 同形状）。weight: [out, in]，H: [in, in]。"""
    W = weight.clone().float()
    n_in = W.shape[1]
    H = H.clone().float()

    dead = torch.diag(H) == 0            # 校准数据中从未激活的输入通道
    H[dead, dead] = 1
    W[:, dead] = 0

    perm = None
    if act_order:                         # 先量化 H 对角元大（激活能量大）的列
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]

    H += damp * torch.mean(torch.diag(H)) * torch.eye(n_in)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    block = group_size if group_size > 0 else 128
    Q = torch.zeros_like(W)
    if group_size <= 0:
        scale, zero = int_qparams(W, bits, symmetric)          # 每个输出通道一组，整行共用

    for i1 in range(0, n_in, block):
        i2 = min(i1 + block, n_in)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if group_size > 0:                # 块和组对齐：进入一组时，用误差补偿后的当前值计算 scale
            scale, zero = int_qparams(W1, bits, symmetric)
        for i in range(i2 - i1):
            w = W1[:, i]
            q = int_quant_dequant(w.unsqueeze(1), scale, zero, bits, symmetric).squeeze(1)
            Q[:, i1 + i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)   # 补偿到本块剩余的列
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]                           # 一次性补偿到之后的块

    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q.to(weight.dtype)
