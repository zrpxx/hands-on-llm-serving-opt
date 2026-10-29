"""数值格式与基本的量化、反量化。对应 2.1 节。"""
import torch

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


def _group(x: torch.Tensor, group_size: int):
    """把最后一维切成若干组，返回 [..., n_groups, group_size] 的视图。group_size <= 0 表示整行一组。"""
    d = x.shape[-1]
    if group_size <= 0 or group_size >= d:
        return x.reshape(*x.shape[:-1], 1, d)
    assert d % group_size == 0, f"最后一维 {d} 不能被 group_size={group_size} 整除"
    return x.reshape(*x.shape[:-1], d // group_size, group_size)


def int_qparams(x: torch.Tensor, bits: int, symmetric: bool = True):
    """沿最后一维计算整数量化的 scale 和 zero（zero 为整数，对称量化时为 0）。

    对称：q ∈ [-2^(b-1), 2^(b-1)-1]，scale = max|x| / (2^(b-1)-1)
    非对称：q ∈ [0, 2^b - 1]，scale = (max - min) / (2^b - 1)，zero = round(-min / scale)
    """
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        zero = torch.zeros_like(scale)
    else:
        qmax = 2 ** bits - 1
        xmin = x.amin(dim=-1, keepdim=True).clamp(max=0)
        xmax = x.amax(dim=-1, keepdim=True).clamp(min=0)
        scale = (xmax - xmin).clamp(min=1e-8) / qmax
        zero = torch.round(-xmin / scale)
    return scale, zero


def int_quant_dequant(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, symmetric: bool = True):
    """用给定的 scale、zero 量化再反量化。"""
    if symmetric:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        lo, hi = 0, 2 ** bits - 1
    q = torch.clamp(torch.round(x / scale) + zero, lo, hi)
    return (q - zero) * scale


def fake_quant_int(x: torch.Tensor, bits: int, group_size: int = -1, symmetric: bool = True) -> torch.Tensor:
    """按组做整数伪量化。group_size=-1 时每一行（例如权重的每个输出通道）一组。"""
    g = _group(x.float(), group_size)
    scale, zero = int_qparams(g, bits, symmetric)
    return int_quant_dequant(g, scale, zero, bits, symmetric).reshape(x.shape).to(x.dtype)


def fake_quant_fp8(x: torch.Tensor, granularity: str = "tensor", fmt: str = "e4m3") -> torch.Tensor:
    """FP8 伪量化。granularity 取 "tensor"（整个张量一个 scale）或 "row"（每行一个 scale，
    对权重是每个输出通道，对激活是每个 token）。"""
    dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    fmax = FP8_E4M3_MAX if fmt == "e4m3" else FP8_E5M2_MAX
    xf = x.float()
    if granularity == "tensor":
        amax = xf.abs().amax()
    elif granularity == "row":
        amax = xf.abs().amax(dim=-1, keepdim=True)
    else:
        raise ValueError(granularity)
    scale = amax.clamp(min=1e-12) / fmax
    return ((xf / scale).to(dtype).to(torch.float32) * scale).to(x.dtype)


def rtn(weight: torch.Tensor, bits: int = 4, group_size: int = 128, symmetric: bool = True) -> torch.Tensor:
    """最简单的权重量化：每组直接四舍五入（round-to-nearest）。"""
    return fake_quant_int(weight, bits, group_size, symmetric)
