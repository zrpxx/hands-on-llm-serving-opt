"""长上下文的三个问题：KV Cache 有多大、稀疏注意力选得准不准、上下文并行怎样切分才均衡。对应 6.6 节。

    python code/stage6/long_context.py
"""
import argparse

import numpy as np

# 每个 token 在所有层上的 KV 字节数
MODELS = [
    ("Llama-3.1-8B（GQA，BF16）", 32 * 2 * 8 * 128 * 2),
    ("Qwen3-235B-A22B（GQA，BF16）", 94 * 2 * 4 * 128 * 2),
    ("DeepSeek-V3（MLA，BF16）", 61 * 576 * 2),
    ("DeepSeek-V4.1-Flash（全局 KV，FP4）", 890),
]


def kv_table():
    print("1) 每条序列的 KV Cache")
    print(f"   {'模型':34s}{'每 token':>10s}{'128K':>10s}{'1M':>10s}")
    for name, b in MODELS:
        print(f"   {name:34s}{b / 1024:>7.1f} KB{b * 131072 / 2**30:>7.1f} GB{b * 2**20 / 2**30:>7.1f} GB")
    print("   DeepSeek-V4.1-Flash 按公开资料中的每 token 890 字节计，只是全局 KV，不含滑动窗口部分。\n")


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sparse_demo(rng, n=32768, d=128, n_needle=8, window=4096, k=2048, idx_dim=16):
    """一条 32K 的序列，其中 8 个位置与当前 query 高度相关，散落在全文各处。"""
    keys = rng.standard_normal((n, d)).astype(np.float32)
    vals = rng.standard_normal((n, d)).astype(np.float32)
    q = rng.standard_normal(d).astype(np.float32)
    needles = rng.choice(n - window, n_needle, replace=False)        # 都在滑动窗口之外
    keys[needles] = q + 0.3 * rng.standard_normal((n_needle, d))
    scale = 1 / np.sqrt(d)

    def attend(sel):
        p = softmax(keys[sel] @ q * scale)
        return p @ vals[sel]

    full = attend(np.arange(n))
    swa = attend(np.arange(n - window, n))
    topk = attend(np.argsort(keys @ q)[-k:])
    # 「闪电索引器」的思路：用一个很小的投影算出近似分数，只用它来挑选位置，再对选中的位置做完整的注意力
    proj = rng.standard_normal((d, idx_dim)).astype(np.float32) / np.sqrt(idx_dim)
    approx = (keys @ proj) @ (q @ proj)
    sel = np.argsort(approx)[-k:]
    idx = attend(sel)
    recall = np.isin(needles, sel).mean()

    err = lambda o: np.linalg.norm(o - full) / np.linalg.norm(full)
    print(f"2) 稀疏注意力：{n} 个位置，{n_needle} 个关键位置都在最近 {window} 个之外")
    print(f"   {'方式':34s}{'参与计算的位置':>12s}{'与完整注意力的相对误差':>22s}")
    print(f"   {'完整注意力':34s}{n:>12d}{0:>20.3f}")
    print(f"   {'滑动窗口':34s}{window:>12d}{err(swa):>20.3f}")
    print(f"   {f'精确 top-{k}':34s}{k:>12d}{err(topk):>20.3f}")
    print(f"   {f'{idx_dim} 维索引器选出的 top-{k}':34s}{k:>12d}{err(idx):>20.3f}")
    print(f"   索引器选中了 {recall:.0%} 的关键位置。")
    print("   滑动窗口看不到远处的关键信息；按相关性挑选的稀疏注意力只算 6% 的位置，误差约为滑动窗口的十分之一。\n")


def cp_balance(n_rank=8, n_tok=131072):
    """因果注意力中，第 i 个 token 要与前面 i 个 token 计算，越靠后的 token 越贵。"""
    cost = lambda a, b: (b * (b + 1) - a * (a + 1)) / 2         # 位置 [a, b) 的总计算量
    chunk = n_tok // n_rank
    naive = [cost(r * chunk, (r + 1) * chunk) for r in range(n_rank)]
    half = n_tok // (2 * n_rank)
    zig = [cost(r * half, (r + 1) * half) + cost((2 * n_rank - 1 - r) * half, (2 * n_rank - r) * half) for r in range(n_rank)]
    print(f"3) 上下文并行：{n_tok} 个 token 的 prefill 切到 {n_rank} 张卡")
    print(f"   {'切分方式':34s}{'最忙 / 平均':>12s}")
    print(f"   {'按顺序切成 8 段':34s}{max(naive) / np.mean(naive):>10.2f} 倍")
    print(f"   {'切成 16 段，第 i 张卡拿第 i 段和倒数第 i 段':34s}{max(zig) / np.mean(zig):>10.2f} 倍")
    print("   顺序切分时，最后一张卡的 token 要看完前面所有内容，算得最多；首尾配对后各卡几乎相同。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    kv_table()
    sparse_demo(rng)
    cp_balance()


if __name__ == "__main__":
    main()
