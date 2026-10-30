"""毕业项目的容量规划工具：给定负载和 SLO，在项目 4 的集群模拟器上搜索满足目标、成本最低的部署方案。

搜索的维度：
- 实例数（每个实例一张 H100，部署 Qwen3-8B）
- 权重精度：BF16，或 FP8（权重字节减半，线性层的算力按两倍计，KV 仍为 BF16）
- 路由：最少负载，或缓存感知
- 部署方式：聚合，或 PD 分离（P:D 取 1:N-1 到 N/2:N/2）

对每个方案跑一遍模拟，计算 goodput（同时满足 TTFT 和 TPOT 目标的请求比例），
报告满足 goodput 目标的方案中卡数最少的几个，以及每百万输出 token 的成本。

    python code/graduation/plan.py                          # 默认：多租户聊天负载，峰值 48 请求/秒
    python code/graduation/plan.py --workload long --rate 4  # 长输入负载
    python code/graduation/plan.py --ttft 0.5 --tpot 0.03 --target 0.99

模拟器只是规划的起点。选出的方案必须在真实环境中验证（见毕业项目的说明）。
"""
import argparse
import os
import sys
import time
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "project4", "solution"))
from cluster.costmodel import H100, QWEN3_8B, ModelSpec  # noqa: E402
from cluster.instance import Instance  # noqa: E402
from cluster.pd import DisaggCluster  # noqa: E402
from cluster.router import CacheAware, LeastLoad  # noqa: E402
from cluster.sim import Cluster, goodput  # noqa: E402
from cluster.workload import long_prompt, multi_tenant  # noqa: E402


class FP8Model(ModelSpec):
    """权重以 FP8 存储：每个参数 1 字节。"""

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9


FP8_MODEL = FP8Model(QWEN3_8B.name + " FP8", QWEN3_8B.params_b, QWEN3_8B.layers, QWEN3_8B.kv_heads, QWEN3_8B.head_dim)
H100_FP8 = replace(H100, tflops=H100.tflops * 2)       # FP8 Tensor Core 的峰值是 BF16 的两倍


def make_requests(a):
    if a.workload == "chat":
        return multi_tenant(a.n, a.rate, n_tenants=a.tenants, prefix_len=a.prefix, seed=a.seed)
    return long_prompt(a.n, a.rate, seed=a.seed)


def run_one(a, n, fp8, router, pd_split):
    hw, model = (H100_FP8, FP8_MODEL) if fp8 else (H100, QWEN3_8B)
    mk_router = (lambda: CacheAware()) if router == "cache" else (lambda: LeastLoad())
    reqs = make_requests(a)
    if pd_split:
        p = pd_split
        c = DisaggCluster([Instance(i, hw, model, role="prefill") for i in range(p)],
                          [Instance(100 + i, hw, model, role="decode") for i in range(n - p)], mk_router())
    else:
        c = Cluster([Instance(i, hw, model) for i in range(n)], mk_router())
    res = c.run(reqs)
    out_tokens = sum(r.output_len for r in res["_requests"])
    span = max(r.finish for r in res["_requests"]) - min(r.arrival for r in res["_requests"])
    return {
        "goodput": goodput(res, a.ttft, a.tpot),
        "ttft99": res["TTFT_P99"], "itl99": res["ITL_P99"], "hit": res["命中率"],
        "tok_s": out_tokens / span,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["chat", "long"], default="chat")
    ap.add_argument("--rate", type=float, default=48.0, help="峰值请求速率，请求/秒")
    ap.add_argument("--n", type=int, default=1500, help="模拟的请求数")
    ap.add_argument("--tenants", type=int, default=128)
    ap.add_argument("--prefix", type=int, default=4096)
    ap.add_argument("--ttft", type=float, default=1.0, help="TTFT 目标，秒")
    ap.add_argument("--tpot", type=float, default=0.03, help="TPOT 目标，秒")
    ap.add_argument("--target", type=float, default=0.99, help="goodput 目标")
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--extra", type=int, default=1, help="找到可行方案后，再多搜索几种卡数，看看更多卡能换来什么")
    ap.add_argument("--gpu-hour", type=float, default=2.5, help="每张卡每小时的成本，美元")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    print(f"负载：{'多租户聊天' if a.workload == 'chat' else '长输入'}，{a.rate} 请求/秒；"
          f"SLO：TTFT ≤ {a.ttft * 1e3:.0f} ms，TPOT ≤ {a.tpot * 1e3:.0f} ms，goodput ≥ {a.target:.0%}\n")
    print(f"{'卡数':>4}  {'精度':<5}{'路由':<8}{'部署':<7}{'goodput':>8}{'TTFT P99':>10}{'ITL P99':>9}{'命中率':>7}"
          f"{'输出 token/s':>13}{'$/百万 token':>13}")
    t0 = time.time()
    found = []
    for n in range(1, a.max_gpus + 1):
        configs = [(fp8, r, None) for fp8 in (False, True) for r in ("least", "cache")]
        if n >= 2:
            configs += [(fp8, "least", p) for fp8 in (False, True) for p in range(1, n // 2 + 1)]
        for fp8, router, pd_split in configs:
            m = run_one(a, n, fp8, router, pd_split)
            ok = m["goodput"] >= a.target
            cost = a.gpu_hour * n / 3600 / m["tok_s"] * 1e6
            dep = f"{pd_split}P{n - pd_split}D" if pd_split else "聚合"
            print(f"{n:>4}  {'FP8' if fp8 else 'BF16':<5}{'缓存感知' if router == 'cache' else '最少负载':<6}{dep:<6}"
                  f"{m['goodput']:>8.2f}{m['ttft99'] * 1e3:>8.0f} ms{m['itl99'] * 1e3:>7.1f} ms{m['hit']:>7.2f}"
                  f"{m['tok_s']:>13.0f}{cost:>13.2f}{'  ✓' if ok else ''}")
            if ok:
                found.append((n, cost, fp8, router, dep))
        if found and n >= found[0][0] + a.extra:
            break
    print(f"\n用时 {time.time() - t0:.0f} 秒。")
    if found:
        print(f"满足目标的最少卡数：{found[0][0]}。可行的方案，按成本排序：")
        for n, cost, fp8, router, dep in sorted(found, key=lambda x: x[1]):
            print(f"  {n} 张卡，{'FP8' if fp8 else 'BF16'}，{'缓存感知' if router == 'cache' else '最少负载'}路由，{dep}；每百万输出 token 约 ${cost:.2f}")
        print("下一步：在真实环境中验证这些方案，并测量模拟器没有建模的因素（量化后的精度、尾延迟、故障）。")
    else:
        print(f"{a.max_gpus} 张卡以内没有满足目标的方案。放宽 SLO、增大 --max-gpus，或者考虑更小的模型、投机解码。")


if __name__ == "__main__":
    main()
