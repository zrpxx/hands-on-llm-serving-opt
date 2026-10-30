"""同一个请求，换一个 batch 计算，结果为什么会变；怎样让它不变。对应 6.7 节。

在 CPU 上就能复现：PyTorch 的矩阵乘法会根据矩阵的形状选择不同的分块和并行方式，
batch 不同，同一行的累加顺序就不同，浮点加法又不满足结合律。

    python code/stage6/batch_invariance.py
"""
import argparse
import time

import numpy as np
import torch


def part1():
    print("1) 浮点加法不满足结合律")
    a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
    print(f"   (1e8 + -1e8) + 1 = {(a + b) + c}    1e8 + (-1e8 + 1) = {a + (b + c)}")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096).astype(np.float32)
    sums = {np.float32(sum(x[p] for p in perm)) for perm in (rng.permutation(4096) for _ in range(5))}
    print(f"   同样 4096 个数，按 5 种顺序相加，得到 {len(sums)} 个不同的结果\n")


def part2(dtype):
    print(f"2) PyTorch 矩阵乘法：第 0 行单独计算，与放在不同大小的 batch 里计算（{dtype}）")
    torch.manual_seed(0)
    W = torch.randn(4096, 4096, dtype=dtype)
    x = torch.randn(1024, 4096, dtype=dtype)
    ref = x[:1] @ W
    for b in [1, 4, 16, 64, 256, 1024]:
        diff = ((x[:b] @ W)[:1] - ref).abs().max().item()
        print(f"   batch {b:5d}：与单独计算的最大差 {diff:.2e}")
    print("   结果与机器、线程数、PyTorch 版本有关，但在大多数机器上都能看到差异。\n")


def chunk_dot(x, W, lo, hi):
    """逐个 k 顺序累加，每个输出元素的加法顺序固定，与 batch 中有多少行无关。"""
    acc = np.zeros((x.shape[0], W.shape[1]), np.float32)
    for k in range(lo, hi):
        acc += x[:, k:k + 1] * W[k]
    return acc


def splitk_matmul(x, W, n_split):
    """把 K 维切成 n_split 段，各段分别累加（GPU 上由不同的线程块并行完成），最后把各段的部分和相加。"""
    bounds = np.linspace(0, W.shape[0], n_split + 1).astype(int)
    out = np.zeros((x.shape[0], W.shape[1]), np.float32)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        out += chunk_dot(x, W, lo, hi)
    return out


def part3():
    print("3) split-K：分几段由 batch 大小决定时会变，固定分段就不变")
    rng = np.random.default_rng(1)
    W = rng.standard_normal((2048, 256)).astype(np.float32)
    x = rng.standard_normal((256, 2048)).astype(np.float32)
    # 常见的启发式：batch 小时 GPU 的并行度不够，就把 K 切得更碎，让更多的线程块一起算
    heuristic = lambda b: 16 if b < 16 else 4 if b < 128 else 1
    ref_v = splitk_matmul(x[:1], W, heuristic(1))
    ref_i = splitk_matmul(x[:1], W, 8)
    for b in [1, 8, 64, 256]:
        v = splitk_matmul(x[:b], W, heuristic(b))[:1]
        i = splitk_matmul(x[:b], W, 8)[:1]
        print(f"   batch {b:4d}：分段数随 batch 变（{heuristic(b):2d} 段），差 {np.abs(v - ref_v).max():.1e}；固定分 8 段，差 {np.abs(i - ref_i).max():.1e}")
    print("   固定分段放弃了小 batch 时多切几段带来的并行度，这就是批不变的性能代价。\n")


def part4():
    print("4) 逐个计算与整批计算的 logits")
    torch.manual_seed(2)
    h = torch.randn(256, 2048, dtype=torch.bfloat16)
    Wv = torch.randn(2048, 32000, dtype=torch.bfloat16) * 0.02
    t0 = time.perf_counter()
    one = torch.cat([h[i:i + 1] @ Wv for i in range(h.shape[0])])
    t1 = time.perf_counter()
    batched = h @ Wv
    t2 = time.perf_counter()
    for b in [16, 64]:
        part = torch.cat([h[i:i + b] @ Wv for i in range(0, h.shape[0], b)])
        print(f"   每 {b:3d} 行一批：与逐行计算不同的 logits 占 {(part != one).float().mean().item():.4%}")
    print(f"   整批 256 行：与逐行计算不同的 logits 占 {(batched != one).float().mean().item():.4%}")
    print(f"   逐行计算（天然批不变）用时 {1e3 * (t1 - t0):.0f} ms，整批计算 {1e3 * (t2 - t1):.0f} ms")
    print("   CPU 上不同的比例很小；GPU 上的 kernel 按形状切换的策略更多，差异常见得多。")
    print("   只要某一步选出的 token 不同，后面的生成就完全分岔了。")


def main():
    argparse.ArgumentParser().parse_args()
    part1()
    part2(torch.bfloat16)
    part2(torch.float32)
    part3()
    part4()


if __name__ == "__main__":
    main()
