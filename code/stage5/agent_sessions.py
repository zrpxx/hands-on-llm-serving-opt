"""用项目 4 的集群模拟器，比较 Agent 负载下不同路由策略的效果。对应 5.7 节。

每个会话是一个 Agent 任务：第一轮带着系统提示词和工具定义，之后每一轮在上一轮的完整上下文后面追加
工具的返回结果，模型输出下一个工具调用。轮与轮之间隔着工具执行的时间。
为了简单，下一轮的到达时间按"上一轮到达 + 预计处理时间 + 工具耗时"预先生成，而不是等上一轮真正完成。

    python code/stage5/agent_sessions.py
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "project4", "solution"))
from cluster.instance import Instance  # noqa: E402
from cluster.router import CacheAware, LeastLoad, PrefixHash, RoundRobin  # noqa: E402
from cluster.sim import Cluster  # noqa: E402
from cluster.workload import BLOCK, Request  # noqa: E402


def agent_sessions(n_sessions=60, rate=0.25, turns=(8, 30), sys_len=4000, tool_len=(100, 800),
                   out_len=(100, 400), tool_time=(0.5, 8.0), seed=0):
    rng = random.Random(seed)
    reqs, t0, rid = [], 0.0, 0
    for s in range(n_sessions):
        t0 += rng.expovariate(rate)                     # 新会话的到达
        t, ctx = t0, sys_len
        for k in range(rng.randint(*turns)):
            if k:
                ctx += rng.randint(*tool_len)           # 追加工具结果
            out = rng.randint(*out_len)
            # 会话的上下文只追加、不修改，所以第 i 个块的内容在各轮之间不变，用 (会话, 块号) 作为键
            reqs.append(Request(rid, t, ctx, out, tenant=s, prefix_id=s, prefix_len=ctx // BLOCK * BLOCK))
            rid += 1
            ctx += out                                  # 模型的输出进入上下文
            t += 0.03 * out + rng.uniform(*tool_time)   # 粗略的生成时间 + 工具执行时间
    reqs.sort(key=lambda r: r.arrival)
    return reqs


def main():
    print("4 个 H100 实例，Qwen3-8B；60 个 Agent 会话，每个 8 到 30 轮，系统提示词 4000 token\n")
    print(f"{'策略':<12}{'命中率':>8}{'TTFT P50':>10}{'TTFT P99':>10}{'ITL P99':>9}")
    for name, router in [("轮询", RoundRobin()), ("最少负载", LeastLoad()), ("会话亲和", PrefixHash(4)), ("缓存感知", CacheAware())]:
        reqs = agent_sessions()
        res = Cluster([Instance(i) for i in range(4)], router).run(reqs)
        print(f"{name:<10}{res['命中率']:>9.2f}{res['TTFT_P50'] * 1e3:>8.0f} ms{res['TTFT_P99'] * 1e3:>8.0f} ms{res['ITL_P99'] * 1e3:>6.0f} ms")
    total = sum(r.input_len for r in agent_sessions())
    outs = sum(r.output_len for r in agent_sessions())
    print(f"\n输入 token 与输出 token 之比：{total / outs:.0f} : 1")


if __name__ == "__main__":
    main()
