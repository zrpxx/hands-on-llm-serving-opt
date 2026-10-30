"""固定草稿长度与按置信度决定草稿长度，在小 batch 和大 batch 下的对比。对应 6.5 节。

每个草稿位置有一个真实的接受概率 a：它因内容而异（重复上下文、写代码时很高，开放式的叙述时很低），
并且越靠后越低。草稿模块同时给出一个带噪声的置信度估计 c。两种策略：

- 固定 K：每步都提出 K 个草稿 token
- 置信度：沿着草稿往后走，累计置信度（各位置 c 的乘积）低于阈值就停，最多 5 个

目标模型验证的耗时按 Roofline 估算：一个 8B 的 BF16 模型在 H100 上，读一遍权重约 5 ms；
batch 中所有序列的验证 token 数之和超过约 180 时，开始受算力限制，每多一个 token 都要付出计算。
草稿的开销按每个草稿 token 0.15 ms 计（与 batch 无关的近似）。

    python code/stage6/mtp_sim.py
"""
import argparse

import numpy as np

T_MEM = 4.8e-3                       # 读一遍 16 GB 权重
T_TOK = 2 * 8e9 / (989e12 * 0.6)      # 每个 token 的计算时间
DRAFT = 0.15e-3
DECAY = np.array([1.0, 0.95, 0.9, 0.86, 0.82])           # 越靠后的位置越难猜


def target_time(n_tokens):
    return max(T_MEM, n_tokens * T_TOK) + 0.3e-3


def simulate(batch, policy, rng, steps=400, noise=0.15):
    produced = 0
    total_time = 0.0
    verified = 0
    for _ in range(steps):
        tokens_this_step = 0
        drafts = []
        for _ in range(batch):
            easy = rng.beta(1.6, 0.6)                                        # 这段内容好不好猜：抄写代码时接近 1
            a = np.clip(easy * DECAY + rng.normal(0, 0.05, 5), 0.01, 0.99)    # 这个序列此刻各位置的真实接受率
            c = np.clip(a + rng.normal(0, noise, 5), 0.01, 0.99)              # 草稿模块估计的置信度
            if policy[0] == "fixed":
                k = policy[1]
            else:
                k, cum = 0, 1.0
                while k < 5 and cum * c[k] >= policy[1]:
                    cum *= c[k]
                    k += 1
            acc = 0
            while acc < k and rng.random() < a[acc]:
                acc += 1
            produced += acc + 1          # 被接受的草稿，加上目标模型自己的一个 token
            tokens_this_step += k + 1
            drafts.append(k)
        verified += tokens_this_step
        total_time += target_time(tokens_this_step) + DRAFT * max(drafts)
    return produced / total_time, produced / (steps * batch), verified / (steps * batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.15, help="置信度估计的噪声")
    args = ap.parse_args()
    policies = [("fixed", 0), ("fixed", 1), ("fixed", 2), ("fixed", 3), ("fixed", 5),
                ("conf", 0.3), ("conf", 0.5), ("conf", 0.7), ("conf", 0.85)]
    names = {("fixed", 0): "不用投机"}
    for batch in [1, 16, 64, 256]:
        rng = np.random.default_rng(args.seed)
        print(f"batch = {batch}")
        print(f"  {'策略':14s}{'吞吐（token/s）':>16s}{'每步产出':>10s}{'每步验证':>10s}")
        base = None
        for p in policies:
            thr, out, ver = simulate(batch, p, rng, steps=max(40, 4000 // batch), noise=args.noise)
            base = base or thr
            name = names.get(p, f"固定 K = {p[1]}" if p[0] == "fixed" else f"置信度 ≥ {p[1]}")
            print(f"  {name:14s}{thr:>12.0f} ({thr / base:4.2f}){out:>10.2f}{ver:>10.2f}")
        print()
    print("1) 每步验证的 token 数相同时（比较「每步验证」接近的两行），按置信度挑选比固定 K 产出更多。")
    print("2) batch 小时，多验证几个 token 几乎不花时间，草稿越长越好；batch 大时 GPU 受算力限制，")
    print("   被拒绝的草稿白白占用计算，最优的 K 变小、最优的阈值变高，直到投机解码本身不再划算。")


if __name__ == "__main__":
    main()
