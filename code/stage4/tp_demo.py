"""在 CPU 上模拟张量并行：把一个 SwiGLU FFN 切到 tp 张"卡"上，验证结果不变，并统计通信量。对应 4.1 节。

Megatron 式的切分：
- gate_proj、up_proj 按输出维度切（列并行）：每张卡算出中间层的一段，不需要通信；
- down_proj 按输入维度切（行并行）：每张卡用自己那一段中间结果算出一个部分和，
  最后把各卡的部分和相加，这就是一次 all-reduce。
注意力同理：q、k、v 按头切，o_proj 按输入切，每层一共两次 all-reduce。

    python code/stage4/tp_demo.py --tp 4
"""
import argparse

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=3072)
    a = ap.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.tokens, a.hidden, dtype=torch.float64)
    Wg, Wu = torch.randn(a.inter, a.hidden, dtype=torch.float64), torch.randn(a.inter, a.hidden, dtype=torch.float64)
    Wd = torch.randn(a.hidden, a.inter, dtype=torch.float64)
    ref = (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T

    shard = a.inter // a.tp
    partial = []
    for rank in range(a.tp):                       # 每一轮循环相当于一张卡上的计算
        sl = slice(rank * shard, (rank + 1) * shard)
        h = F.silu(x @ Wg[sl].T) * (x @ Wu[sl].T)  # 列并行：本卡的那一段中间结果，[tokens, inter/tp]
        partial.append(h @ Wd[:, sl].T)            # 行并行：部分和，[tokens, hidden]
    out = torch.stack(partial).sum(0)              # all-reduce

    print(f"tp = {a.tp}：与单卡结果的最大差 {(out - ref).abs().max().item():.2e}")
    w_full = (2 * a.inter * a.hidden + a.hidden * a.inter) * 2
    print(f"每张卡存的 FFN 权重：{w_full / a.tp / 2**20:.1f} MiB（单卡 {w_full / 2**20:.1f} MiB），BF16")
    msg = a.tokens * a.hidden * 2
    ring = 2 * (a.tp - 1) / a.tp * msg
    print(f"一次 all-reduce 的数据：{a.tokens} × {a.hidden} × 2 字节 = {msg / 1024:.0f} KiB；"
          f"ring 算法下每张卡收发约 {ring / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
