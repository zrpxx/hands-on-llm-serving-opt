"""测量 [M, K] x [K, N] 矩阵乘在不同 M 下的实际算力，观察从访存受限到计算受限的转折。

M 对应 decode 时的批大小，或 prefill 时的 token 数。K = N = hidden_size。

用法：
    python roofline_sweep.py --hidden 4096 --peak-tflops 312 --peak-tbs 2.039
输出每个 M 的理论算术强度、实测 TFLOPS、实测等效带宽，并可选画图。
"""
import argparse
import time

import torch


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--peak-tflops", type=float, required=True, help="BF16 稠密峰值，例如 A100/A800 为 312")
    ap.add_argument("--peak-tbs", type=float, required=True, help="显存带宽 TB/s，例如 A100/A800 80GB SXM 为 2.039")
    ap.add_argument("--plot", default="", help="输出图片路径，例如 roofline.png")
    a = ap.parse_args()

    h = a.hidden
    ridge = a.peak_tflops / a.peak_tbs
    w = torch.randn(h, h, device="cuda", dtype=torch.bfloat16)
    rows = []
    print(f"拐点（ridge point）= {ridge:.0f} FLOPs/B")
    print(f"{'M':>6} {'AI(FLOPs/B)':>12} {'TFLOPS':>8} {'占峰值':>7} {'等效TB/s':>9}")
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
        t = bench(lambda: x @ w)
        flops = 2 * m * h * h
        bytes_ = 2 * (m * h + h * h + m * h)
        ai = flops / bytes_
        tf = flops / t / 1e12
        rows.append((m, ai, tf))
        print(f"{m:>6} {ai:>12.1f} {tf:>8.1f} {tf / a.peak_tflops:>6.0%} {bytes_ / t / 1e12:>9.2f}")

    if a.plot:
        import matplotlib.pyplot as plt
        xs = [10 ** (i / 20) for i in range(0, 81)]
        plt.loglog(xs, [min(a.peak_tflops, x * a.peak_tbs) for x in xs], label="roofline (theory)")
        plt.loglog([r[1] for r in rows], [r[2] for r in rows], "o-", label="measured")
        plt.axvline(ridge, ls="--", c="gray")
        plt.xlabel("Arithmetic intensity (FLOPs/Byte)")
        plt.ylabel("TFLOPS")
        plt.legend()
        plt.savefig(a.plot, dpi=150)
        print(f"图已保存到 {a.plot}")


if __name__ == "__main__":
    main()
