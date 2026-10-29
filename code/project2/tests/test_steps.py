"""分步测试。每一步完成后运行：pytest tests/test_steps.py -k step01"""
import torch

from conftest import rel_err


# ------------------------------------------------------------------ 第 1 步：整数量化
def test_step01_symmetric_int8_roundtrip():
    from quantlab.formats import int_qparams, int_quant_dequant
    x = torch.tensor([[-1.0, 0.5, 0.25, 1.27]])
    scale, zero = int_qparams(x, 8, symmetric=True)
    assert torch.allclose(scale, torch.tensor([[1.27 / 127]]))
    assert torch.all(zero == 0)
    y = int_quant_dequant(x, scale, zero, 8, True)
    assert (y - x).abs().max() <= scale.max() / 2 + 1e-7


def test_step01_asymmetric_uses_full_range():
    from quantlab.formats import int_qparams, int_quant_dequant
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    scale, zero = int_qparams(x, 2, symmetric=False)        # 4 个等级正好表示 0、1、2、3
    assert torch.allclose(scale, torch.tensor([[1.0]]))
    assert torch.allclose(int_quant_dequant(x, scale, zero, 2, False), x)


def test_step01_group_size_reduces_error():
    from quantlab.formats import fake_quant_int
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 512, generator=g)
    w[:, 7] *= 30                                          # 一个离群列会拉大它所在组的 scale
    e_row = rel_err(fake_quant_int(w, 4, -1), w)
    e_g128 = rel_err(fake_quant_int(w, 4, 128), w)
    e_g32 = rel_err(fake_quant_int(w, 4, 32), w)
    assert e_g32 < e_g128 < e_row


# ------------------------------------------------------------------ 第 2 步：FP8
def test_step02_fp8_e4m3_range_and_error():
    from quantlab.formats import fake_quant_fp8
    x = torch.linspace(-3, 3, 1000)
    y = fake_quant_fp8(x, "tensor")
    rel = ((y - x).abs() / x.abs().clamp(min=1e-3))[x.abs() > 0.05]
    assert rel.max() < 0.07                                 # E4M3 有 3 位尾数，相对误差不超过 2^-4
    big = torch.tensor([[1000.0, -2000.0, 1.0]])
    assert torch.allclose(fake_quant_fp8(big, "row")[0, :2], big[0, :2])   # 最大值能精确表示，不会被截断


def test_step02_fp8_row_scaling_helps_mixed_rows():
    from quantlab.formats import fake_quant_fp8
    x = torch.cat([torch.randn(1, 64) * 1000, torch.randn(1, 64) * 0.001])
    e_tensor = rel_err(fake_quant_fp8(x, "tensor")[1], x[1])
    e_row = rel_err(fake_quant_fp8(x, "row")[1], x[1])
    assert e_row < e_tensor


# ------------------------------------------------------------------ 第 3 步：GPTQ
def test_step03_gptq_beats_rtn(layer_data):
    from quantlab.formats import rtn
    from quantlab.gptq import gptq_quantize, hessian_from_inputs
    w, x = layer_data
    y = x @ w.T
    H = hessian_from_inputs(x)
    for bits in (4, 3):
        e_rtn = rel_err(x @ rtn(w, bits, 128).T, y)
        e_gptq = rel_err(x @ gptq_quantize(w, H, bits, 128).T, y)
        assert e_gptq < 0.5 * e_rtn, (bits, e_gptq, e_rtn)


def test_step03_gptq_output_is_on_grid(layer_data):
    """GPTQ 只改变取整的方向，结果仍然在每组的量化网格上：每组最多 2^bits 个不同的值。"""
    from quantlab.gptq import gptq_quantize, hessian_from_inputs
    w, x = layer_data
    q = gptq_quantize(w, hessian_from_inputs(x), 3, 128)
    for row in q[:4]:
        for grp in row.reshape(-1, 128):
            assert grp.unique().numel() <= 8


# ------------------------------------------------------------------ 第 4 步：AWQ
def test_step04_awq_never_worse_than_rtn(layer_data):
    from quantlab.awq import awq_search
    from quantlab.formats import rtn
    w, x = layer_data
    y = x @ w.T
    wq, alpha, errors = awq_search(w, x, 4, 128)
    assert len(errors) == 20 and 0 <= alpha < 1
    assert abs(errors[0] - ((x @ rtn(w, 4, 128).T - y) ** 2).mean().item()) < 1e-6   # α = 0 就是 RTN
    assert abs(min(errors) - ((x @ wq.T - y) ** 2).mean().item()) < 1e-6   # 返回的是误差最小的那组
    assert alpha > 0                                         # 有离群通道时，最优的 α 大于 0


# ------------------------------------------------------------------ 第 5 步：SmoothQuant 与 W8A8
def test_step05_smooth_scales_formula():
    from quantlab.smoothquant import smooth_scales
    act = torch.tensor([16.0, 1.0])
    w = torch.tensor([[1.0, 4.0]])
    s = smooth_scales(act, w, 0.5)
    assert torch.allclose(s, torch.tensor([4.0, 0.5]))


def test_step05_smoothing_reduces_w8a8_error(layer_data):
    from quantlab.smoothquant import smooth_scales, w8a8_matmul
    w, x = layer_data
    y = x @ w.T
    e_plain = rel_err(w8a8_matmul(x, w, "tensor"), y)
    s = smooth_scales(x.abs().amax(0), w, 0.5)
    e_smooth = rel_err(w8a8_matmul(x / s, w * s, "tensor"), y)
    e_token = rel_err(w8a8_matmul(x, w, "token"), y)
    assert e_smooth < 0.3 * e_plain
    assert e_token < e_plain


# ------------------------------------------------------------------ 第 6 步：KV Cache
def test_step06_kv_axes():
    from quantlab.kv import fake_quant_kv
    g = torch.Generator().manual_seed(2)
    k = torch.randn(128, 2, 32, generator=g)
    k[:, :, 5] *= 20                                         # Key 的通道离群值
    e_token = rel_err(fake_quant_kv(k, 4, "token"), k)
    e_channel = rel_err(fake_quant_kv(k, 4, "channel"), k)
    assert e_channel < e_token
    assert fake_quant_kv(k, 4, "channel").shape == k.shape
    assert rel_err(fake_quant_kv(k, 8, "token"), k) < 1e-3
