"""用 NumPy 模拟几种 4 位格式，比较它们的量化误差。对应 6.2 节。

- INT4，每 128 个元素一组，FP16 缩放因子（GPTQ、AWQ 的常见设置）；以及每 16 个一组，用来对照块大小的影响
- MXFP4：元素为 FP4 E2M1，每 32 个元素共享一个 E8M0 缩放因子（只能是 2 的幂）
- NVFP4：元素为 FP4 E2M1，每 16 个元素共享一个 E4M3 缩放因子，整个张量再有一个 FP32 缩放因子

只模拟数值，不模拟存储和计算。误差用信噪比 SQNR（分贝，越大越好）衡量。

    python code/stage6/fp4_formats.py
"""
import argparse

import numpy as np

# FP4 E2M1 能表示的非负值：符号 1 位，指数 2 位，尾数 1 位
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def round_to_grid(x, grid):
    """把 x 的绝对值舍入到 grid 中最近的值（就近舍入），保留符号。"""
    a = np.abs(x)
    idx = np.abs(a[..., None] - grid).argmin(-1)
    return np.sign(x) * grid[idx]


def to_e4m3(s):
    """把正数舍入到 FP8 E4M3 能表示的值（最大 448，最小正规数 2^-6，带次正规数）。"""
    s = np.clip(s, 2.0 ** -9, 448.0)
    e = np.floor(np.log2(s))
    e = np.maximum(e, -6)                  # 次正规数区间，步长固定为 2^-9
    step = 2.0 ** (e - 3)                  # 3 位尾数
    return np.clip(np.round(s / step) * step, 2.0 ** -9, 448.0)


def blocks(x, size):
    return x.reshape(-1, size)


def quant_int4(x, group=128):
    b = blocks(x, group)
    scale = np.abs(b).max(1, keepdims=True) / 7.0
    scale = scale.astype(np.float16).astype(np.float64)
    scale[scale == 0] = 1.0
    q = np.clip(np.round(b / scale), -8, 7)
    return (q * scale).reshape(x.shape)


def quant_mxfp4(x, block=32):
    b = blocks(x, block)
    amax = np.abs(b).max(1, keepdims=True)
    amax[amax == 0] = 1.0
    # OCP MX 规范：共享指数 = floor(log2(amax)) - emax，E2M1 的 emax = 2（最大值 6 = 1.5 × 2^2）
    scale = 2.0 ** (np.floor(np.log2(amax)) - 2)
    q = round_to_grid(np.clip(b / scale, -6, 6), E2M1)
    return (q * scale).reshape(x.shape)


def quant_nvfp4(x, block=16):
    b = blocks(x, block)
    # 张量级的 FP32 缩放：让每块的缩放因子落进 E4M3 的表示范围
    gscale = np.abs(x).max() / (6.0 * 448.0)
    amax = np.abs(b).max(1, keepdims=True)
    s = to_e4m3(amax / 6.0 / gscale) * gscale
    s[s == 0] = 1.0
    q = round_to_grid(np.clip(b / s, -6, 6), E2M1)
    return (q * s).reshape(x.shape)


FORMATS = {
    "INT4 g128": (quant_int4, 4 + 16 / 128),
    "INT4 g16": (lambda x: quant_int4(x, 16), 4 + 16 / 16),
    "MXFP4": (quant_mxfp4, 4 + 8 / 32),
    "NVFP4": (quant_nvfp4, 4 + 8 / 16 + 32 / 4096 / 4096),
}


def sqnr(x, y):
    return 10 * np.log10((x ** 2).sum() / ((x - y) ** 2).sum())


def make_data(rng, n=4096, k=4096):
    w = rng.standard_normal((n, k)) * 0.02                     # 权重：近似正态
    a = rng.standard_normal((n, k))                            # 激活：少数通道有离群值
    hot = rng.choice(k, k // 100, replace=False)
    a[:, hot] *= 30
    t = rng.standard_t(3, (n, k))                              # 重尾分布
    return {"权重（正态）": w, "激活（1% 通道放大 30 倍）": a, "重尾（t 分布，自由度 3）": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    data = make_data(rng, 1024, 4096)

    print("每个元素占用的位数（含缩放因子）：")
    for name, (_, bits) in FORMATS.items():
        print(f"  {name:10s} {bits:.3f}")
    print()
    print(f"{'数据':24s}" + "".join(f"{n:>12s}" for n in FORMATS))
    for dname, x in data.items():
        row = [sqnr(x, f(x)) for f, _ in FORMATS.values()]
        print(f"{dname:24s}" + "".join(f"{v:10.1f} dB" for v in row))
    print("\nSQNR 每高 6 dB，误差的幅度约减半。")


if __name__ == "__main__":
    main()
