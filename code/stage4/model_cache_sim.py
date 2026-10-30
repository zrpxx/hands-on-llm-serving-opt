"""模拟多模型服务的模型缓存：N 个模型共用 G 张卡，每张卡同时只能放 slots 个模型，按 LRU 换入换出。
对应 4.5 节。统计冷启动的比例和延迟的长尾，与"每个模型一个常驻实例"比较。

    python code/stage4/model_cache_sim.py --models 30 --slots 2
"""
import argparse
import heapq
import random


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def simulate(n_models, gpus, slots, rate, n_req, load_s, serve_s, zipf, seed):
    rng = random.Random(seed)
    w = [1 / (k + 1) ** zipf for k in range(n_models)]
    # 每张卡：已加载的模型（LRU 顺序）和空闲时间
    loaded = [[] for _ in range(gpus)]
    free_at = [0.0] * gpus
    t, lat, cold = 0.0, [], 0
    for _ in range(n_req):
        t += rng.expovariate(rate)
        m = rng.choices(range(n_models), w)[0]
        holders = [g for g in range(gpus) if m in loaded[g]]
        if holders:                                   # 已加载：去最早空闲的那张卡
            g = min(holders, key=lambda g: free_at[g])
            start = max(t, free_at[g])
            loaded[g].remove(m); loaded[g].append(m)
            end = start + serve_s
        else:                                         # 冷启动：选最早空闲的卡，必要时淘汰最久未用的模型
            g = min(range(gpus), key=lambda g: free_at[g])
            start = max(t, free_at[g])
            if len(loaded[g]) >= slots:
                loaded[g].pop(0)
            loaded[g].append(m)
            end = start + load_s + serve_s
            cold += 1
        free_at[g] = end
        lat.append(end - t)
    return cold / n_req, percentile(lat, 50), percentile(lat, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=30)
    ap.add_argument("--slots", type=int, default=2, help="每张卡同时能放几个模型")
    ap.add_argument("--rate", type=float, default=2.0, help="所有模型合计的每秒请求数")
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--load", type=float, default=20.0, help="加载一个模型的秒数")
    ap.add_argument("--serve", type=float, default=0.2, help="一个请求占用卡的秒数")
    ap.add_argument("--zipf", type=float, default=1.2)
    a = ap.parse_args()
    print(f"{a.models} 个模型，流量按 Zipf({a.zipf}) 分布，合计 {a.rate} 请求/秒；加载一个模型 {a.load} 秒，处理一个请求 {a.serve} 秒\n")
    print(f"{'配置':<28}{'冷启动比例':>10}{'延迟 P50':>10}{'延迟 P99':>10}")
    for gpus in (4, 6, 8, 12):
        c, p50, p99 = simulate(a.models, gpus, a.slots, a.rate, a.requests, a.load, a.serve, a.zipf, 1)
        print(f"共享：{gpus:>2} 张卡 × 每卡 {a.slots} 个模型{'':<5}{c:>10.1%}{p50:>9.1f}s{p99:>9.1f}s")
    print(f"{'专属：每个模型一张卡':<26}{0:>10.1%}{a.serve:>9.1f}s{'约 ' + str(a.serve):>10}  （需要 {a.models} 张卡）")


if __name__ == "__main__":
    main()
