"""SmoothQuant 与 W8A8。对应 2.4 节。

激活中个别通道的数值很大（离群值），按张量量化时会把其余通道的精度挤掉。SmoothQuant 把激活每个通道
除以 s、权重对应的输入通道乘以 s，结果不变，但量化难度从激活转移了一部分到权重上：
    s_j = max|X_j|^α / max|W_j|^(1-α)
参考 Xiao et al., SmoothQuant (ICML 2023)。
"""
import torch

from .formats import fake_quant_int, fake_quant_fp8


def smooth_scales(act_absmax: torch.Tensor, weight: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """act_absmax: [in] 每个输入通道的激活最大绝对值；weight: [out, in]。返回 s: [in]。"""
    w_absmax = weight.float().abs().amax(dim=0).clamp(min=1e-5)
    return (act_absmax.float().clamp(min=1e-5).pow(alpha) / w_absmax.pow(1 - alpha)).clamp(min=1e-5)


def w8a8_matmul(x: torch.Tensor, weight: torch.Tensor, act: str = "tensor", fmt: str = "int8") -> torch.Tensor:
    """W8A8 的伪量化矩阵乘：权重按输出通道量化，激活按张量（act="tensor"）或按 token（act="token"）量化。"""
    if fmt == "int8":
        wq = fake_quant_int(weight.float(), 8, -1, True)
        if act == "tensor":
            xq = fake_quant_int(x.float().reshape(1, -1), 8, -1, True).reshape(x.shape)
        else:
            xq = fake_quant_int(x.float(), 8, -1, True)
    else:
        wq = fake_quant_fp8(weight.float(), "row")
        xq = fake_quant_fp8(x.float(), "tensor" if act == "tensor" else "row")
    return xq @ wq.T


class W8A8Linear(torch.nn.Module):
    """把 nn.Linear 换成 W8A8 伪量化版本，可选先做 SmoothQuant 的平滑。"""

    def __init__(self, linear: torch.nn.Linear, smooth: torch.Tensor = None, act: str = "token", fmt: str = "int8"):
        super().__init__()
        w = linear.weight.data.float()
        self.register_buffer("inv_s", None if smooth is None else 1.0 / smooth.float())
        if smooth is not None:
            w = w * smooth.float()
        self.register_buffer("weight", w)
        self.bias = linear.bias
        self.act, self.fmt = act, fmt

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1]).float()
        if self.inv_s is not None:
            h = h * self.inv_s
        y = w8a8_matmul(h, self.weight, self.act, self.fmt)
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*shape[:-1], -1).to(x.dtype)
