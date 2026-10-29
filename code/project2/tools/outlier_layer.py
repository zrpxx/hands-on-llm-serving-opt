"""在一个模拟的线性层上比较各种量化方法的输出误差。

真实的大模型（大约 6B 以上）在少数激活通道上会出现比其他通道大几十倍的数值，这是 AWQ 和
SmoothQuant 要处理的问题。几 M 参数的小模型里通常看不到这种现象，所以单独用模拟数据演示。
    python tools/outlier_layer.py --outlier-scale 20
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                os.environ.get("QUANTLAB_IMPL", "starter")))
from quantlab.awq import awq_search                                   # noqa: E402
from quantlab.formats import rtn                                      # noqa: E402
from quantlab.gptq import gptq_quantize, hessian_from_inputs         # noqa: E402
from quantlab.smoothquant import smooth_scales, w8a8_matmul         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier-scale", type=float, default=20.0, help="离群通道比其他通道大多少倍；1 表示没有离群值")
    ap.add_argument("--outliers", type=int, default=8)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    n_in, n_out, n = 512, 256, 2048
    x = torch.randn(n, n_in, generator=g) @ (torch.randn(n_in, n_in, generator=g) / n_in ** 0.5)
    x[:, :a.outliers] *= a.outlier_scale
    w = torch.randn(n_out, n_in, generator=g) * 0.02
    y = x @ w.T
    err = lambda out: ((out - y) ** 2).mean().item() / (y ** 2).mean().item()

    print(f"离群通道 {a.outliers} 个，放大 {a.outlier_scale:g} 倍。下面是输出的相对均方误差：")
    for bits in (4, 3):
        H = hessian_from_inputs(x)
        wq, alpha, _ = awq_search(w, x, bits, 128)
        print(f"  W{bits} g128   RTN {err(x @ rtn(w, bits, 128).T):.2e}   GPTQ {err(x @ gptq_quantize(w, H, bits, 128).T):.2e}"
              f"   AWQ {err(x @ wq.T):.2e}（α = {alpha:.2f}）")
    s = smooth_scales(x.abs().amax(0), w, 0.5)
    print(f"  W8A8 按张量 {err(w8a8_matmul(x, w, 'tensor')):.2e}   按 token {err(w8a8_matmul(x, w, 'token')):.2e}"
          f"   SmoothQuant(α=0.5) {err(w8a8_matmul(x / s, w * s, 'tensor')):.2e}"
          f"   FP8 按 token {err(w8a8_matmul(x, w, 'token', 'fp8')):.2e}")


if __name__ == "__main__":
    main()
