"""专家并行中的负载不均衡，以及 EPLB 怎样用冗余专家缓解它。对应 6.3 节。

模拟一个 256 个路由专家、每个 token 选 8 个的 MoE 层，部署在 32 张卡上（EP = 32）。
专家的热度服从 Zipf 分布，并且随时间缓慢漂移。比较三种放置方式：

- 静态：专家按编号均匀分到各卡，每卡 8 个
- EPLB：每卡多放若干个冗余槽位，先把冗余副本分给最热的专家，再用贪心法把所有副本装进各卡，
  使每卡的预计负载尽量平均。副本之间平均分摊这个专家的 token
- 过期的 EPLB：同样的算法，但用的是很久以前统计的热度

一层 MoE 的耗时由最忙的那张卡决定，所以用"均衡度 = 平均负载 / 最大负载"衡量，1 表示完全均衡。

    python code/stage6/eplb_sim.py
"""
import argparse
import heapq

import numpy as np


def zipf_popularity(n, s, rng):
    p = 1.0 / np.arange(1, n + 1) ** s
    rng.shuffle(p)
    return p / p.sum()


def drift(p, rng, rate):
    """热度缓慢变化：乘上一个对数正态的扰动。"""
    q = p * np.exp(rng.normal(0, rate, p.shape))
    return q / q.sum()


def static_placement(n_exp, n_gpu):
    per = n_exp // n_gpu
    return [list(range(g * per, (g + 1) * per)) for g in range(n_gpu)], np.ones(n_exp)


def eplb_placement(load, n_gpu, slots_per_gpu):
    """简化版的 EPLB：1）贪心复制最热的专家；2）按副本负载从大到小，放进当前最空、还有槽位的卡。"""
    n_exp = len(load)
    n_slots = n_gpu * slots_per_gpu
    replicas = np.ones(n_exp)
    heap = [(-load[e], e) for e in range(n_exp)]
    heapq.heapify(heap)
    for _ in range(n_slots - n_exp):
        _, e = heapq.heappop(heap)
        replicas[e] += 1
        heapq.heappush(heap, (-load[e] / replicas[e], e))
    items = sorted(((load[e] / replicas[e], e) for e in range(n_exp) for _ in range(int(replicas[e]))), reverse=True)
    gpus = [[] for _ in range(n_gpu)]
    g_load = np.zeros(n_gpu)
    for w, e in items:
        cand = [g for g in range(n_gpu) if len(gpus[g]) < slots_per_gpu and e not in gpus[g]]
        if not cand:   # 剩下有空位的卡都已有这个专家，只好放在同一张卡上
            cand = [g for g in range(n_gpu) if len(gpus[g]) < slots_per_gpu]
        g = min(cand, key=lambda i: g_load[i])
        gpus[g].append(e)
        g_load[g] += w
    return gpus, replicas


def gpu_loads(tokens_per_expert, gpus, replicas):
    return np.array([sum(tokens_per_expert[e] / replicas[e] for e in exps) for exps in gpus])


def balancedness(loads):
    return loads.mean() / loads.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=32)
    ap.add_argument("--redundant", type=int, default=1, help="每卡的冗余槽位数")
    ap.add_argument("--skew", type=float, default=0.7, help="Zipf 指数，越大越不均衡")
    ap.add_argument("--tokens", type=int, default=8192, help="每步的 token 数")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pop = zipf_popularity(args.experts, args.skew, rng)
    hist = []          # 每步每个专家实际收到的 token 数
    for _ in range(args.steps):
        pop = drift(pop, rng, 0.03)
        hist.append(rng.multinomial(args.tokens * args.topk, pop))
    hist = np.array(hist)

    slots = args.experts // args.gpus + args.redundant
    st_g, st_r = static_placement(args.experts, args.gpus)
    # EPLB：用前 100 步的统计决定放置，评估第 100～199 步
    fresh_g, fresh_r = eplb_placement(hist[:100].mean(0), args.gpus, slots)
    # 过期：同一套放置，评估最后 100 步（热度已经漂移了 200～300 步）
    print(f"{args.experts} 个专家，top-{args.topk}，{args.gpus} 张卡，每卡 {args.experts // args.gpus} 个专家 + {args.redundant} 个冗余槽位；Zipf 指数 {args.skew}")
    print()
    print(f"{'放置方式':18s}{'均衡度':>8s}{'最忙的卡 / 平均':>16s}")
    rows = [
        ("静态", hist[100:200], st_g, st_r),
        ("EPLB（统计是新的）", hist[100:200], fresh_g, fresh_r),
        ("EPLB（统计已过期）", hist[-100:], fresh_g, fresh_r),
    ]
    for name, h, g, r in rows:
        b = np.mean([balancedness(gpu_loads(t, g, r)) for t in h])
        print(f"{name:18s}{b:8.2f}{1 / b:14.2f} 倍")
    n_slot_total = args.gpus * slots
    print(f"\n冗余槽位占全部槽位的 {args.gpus * args.redundant / n_slot_total:.0%}，即每卡多存 {args.redundant} 个专家的权重。")
    print("MoE 层的耗时由最忙的卡决定：均衡度 0.5 意味着一半的算力在等待。")


if __name__ == "__main__":
    main()
