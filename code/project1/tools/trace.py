"""逐步打印调度器（和引擎）的状态，用来理解与调试自己的实现。

输出各列：phase 为 prefill（本步算完 prompt）、chunk（prompt 只算了一部分）、decode；
n 为本步处理的 token 数；computed 为本步前后的 num_computed_tokens；len 为本步结束时的总 token 数（含新采样的 token）。

两种模式：
  只看调度器（不需要模型，完成第 1~6 步后即可使用）：
    python tools/trace.py --prompts 6,6,20 --budget 10 --blocks 12

  跑真实引擎（完成第 7~8 步后使用），额外显示每步采样出的 token：
    python tools/trace.py --model /tmp/tiny --prompts 6,6,20 --budget 10 --blocks 12

  --impl solution 可以先看参考实现的行为，再和自己的输出对比。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="starter", choices=["starter", "solution"])
    ap.add_argument("--model", default="", help="模型目录；不给则只模拟调度器")
    ap.add_argument("--prompts", default="6,6,20", help="各请求的 prompt 长度，逗号分隔")
    ap.add_argument("--arrive", default="", help="各请求在第几步到达，逗号分隔；默认都在第 1 步")
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--budget", type=int, default=10, help="max_num_batched_tokens")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--policy", default="continuous", choices=["continuous", "static"])
    ap.add_argument("--no-chunk", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, a.impl))

    from miniserve.sequence import SamplingParams, Sequence

    lens = [int(x) for x in a.prompts.split(",")]
    arrive = [int(x) for x in a.arrive.split(",")] if a.arrive else [1] * len(lens)
    params = SamplingParams(max_tokens=a.max_tokens, ignore_eos=True)
    names = {}

    if a.model:
        from miniserve import EngineConfig, LLMEngine
        eng = LLMEngine(EngineConfig(model=a.model, device="cpu", num_gpu_blocks=a.blocks, block_size=a.block_size,
                                     max_model_len=max(lens) + a.max_tokens + a.block_size,
                                     max_num_batched_tokens=a.budget, max_num_seqs=a.max_seqs,
                                     policy=a.policy, chunked_prefill=not a.no_chunk))
        sch = eng.scheduler
        add = lambda ids: eng.add_request(ids, params)
    else:
        from miniserve.kv_cache import BlockAllocator
        from miniserve.scheduler import Scheduler
        sch = Scheduler(BlockAllocator(a.blocks), a.block_size, a.max_seqs, a.budget, a.policy, not a.no_chunk)

        def add(ids):
            s = Sequence(ids, params)
            sch.add(s)
            return s

    pending = sorted(zip(arrive, range(len(lens))))
    for step in range(1, a.steps + 1):
        while pending and pending[0][0] <= step:
            _, i = pending.pop(0)
            s = add(list(range(10 + 100 * i, 10 + 100 * i + lens[i])))
            names[s.seq_id] = chr(ord("A") + i)
        if not sch.has_unfinished() and not pending:
            print("全部完成。空闲块", sch.allocator.num_free, "/", a.blocks)
            return
        before = {s.seq_id: s.num_computed_tokens for s in list(sch.running) + list(sch.waiting)}

        if a.model:
            outs = eng.step()
            scheduled = [(o.seq, None) for o in outs]
            sampled = {o.seq.seq_id: o.new_token_id for o in outs}
            ns = {o.seq.seq_id: o.seq.num_computed_tokens - before.get(o.seq.seq_id, 0) for o in outs}
            preempted = None
        else:
            out = sch.schedule()
            scheduled, preempted = out.scheduled, out.num_preempted
            ns = {s.seq_id: n for s, n in scheduled}
            sampled = {}
            for s, n in scheduled:                       # 模拟引擎：推进已算 token，算完就"生成"一个
                s.num_computed_tokens += n
                if s.num_computed_tokens == s.num_tokens:
                    s.token_ids.append(0)
                    sampled[s.seq_id] = 0
                    if s.num_output_tokens >= params.max_tokens:
                        sch.finish(s, "length")

        if not scheduled and not pending:
            print(f"\n第 {step} 步：没有任何请求被调度，但仍有请求在等待：{[names[x.seq_id] for x in sch.waiting]}。")
            print("  常见原因：关闭切块时 prompt 长于 token 预算（--budget），或空闲块不足以放下队头请求。")
            return
        used = sum(ns.values())
        print(f"\n第 {step} 步  本步 token {used}/{a.budget}  空闲块 {sch.allocator.num_free}/{a.blocks}"
              + (f"  抢占 {preempted} 个" if preempted else ""))
        print(f"  {'seq':4}{'phase':9}{'n':>4}{'computed':>11}{'len':>5}  block_table")
        for s, _ in scheduled:
            n = ns[s.seq_id]
            b = before.get(s.seq_id, 0)
            phase = "decode" if n == 1 and b >= s.prompt_len else ("prefill" if b + n >= s.prompt_len else "chunk")
            tok = f"  -> 采样 {sampled[s.seq_id]}" if s.seq_id in sampled and sampled[s.seq_id] is not None else ""
            done = "  [完成]" if s.is_finished else ""
            print(f"  {names[s.seq_id]:4}{phase:9}{n:>4}{f'{b}->{b + n}':>11}{s.num_tokens:>5}  {s.block_table}{tok}{done}")
        waiting = [names[s.seq_id] + ("(被抢占)" if s.num_preemptions else "") for s in sch.waiting]
        print(f"  running={[names[s.seq_id] for s in sch.running]}  waiting={waiting}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"\n还没有实现：{e}。完成对应步骤后再运行，或先用 --impl solution 观察参考行为。")
        sys.exit(1)
