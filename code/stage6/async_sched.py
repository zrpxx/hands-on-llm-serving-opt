"""同步调度与异步调度的对比：用一个线程模拟 GPU，观察 CPU 工作能否被藏到 GPU 计算的背后。对应 6.8 节。

decode 的每一步分三段：
- 调度与准备输入（CPU）：选出这一步的请求，分配 KV 块，构造 input_ids、positions、block table
- 前向计算（GPU）：这里用 sleep 代替，sleep 时会释放 GIL，与真实的 GPU 一样可以和 CPU 并行
- 后处理（CPU）：把采样结果拷回 CPU，检查停止条件，增量 detokenize，发给客户端

同步调度三段依次执行。异步调度在 GPU 计算第 N 步时，CPU 已经在准备第 N+1 步、并处理第 N-1 步的结果。
代价是：调度第 N+1 步时还不知道第 N 步有哪些请求结束了，这些请求会多算一步，结果被丢弃。

    python code/stage6/async_sched.py
"""
import argparse
import queue
import random
import threading
import time


class FakeGPU:
    def __init__(self, gpu_ms):
        self.gpu_ms = gpu_ms
        self.jobs, self.done = queue.Queue(), queue.Queue()
        self.busy = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            step, batch = self.jobs.get()
            t = time.perf_counter()
            time.sleep(self.gpu_ms / 1000)
            self.busy += time.perf_counter() - t
            self.done.put((step, batch))


def make_requests(n, rng):
    return {i: rng.randint(20, 200) for i in range(n)}      # 每个请求还要生成多少个 token


def run(mode, args):
    rng = random.Random(args.seed)
    remaining = make_requests(args.requests, rng)
    gpu = FakeGPU(args.gpu_ms)
    cpu = lambda ms: time.sleep(ms / 1000)
    useful = wasted = 0
    pending = list(remaining)          # 等待加入的请求
    running = set()

    def schedule():
        while pending and len(running) < args.max_batch:
            running.add(pending.pop())
        cpu(args.sched_ms)
        return list(running)

    def postprocess(batch):
        nonlocal useful, wasted
        cpu(args.post_ms)
        for r in batch:
            if remaining[r] <= 0:
                wasted += 1            # 异步时多算的一步
                continue
            useful += 1
            remaining[r] -= 1
            if remaining[r] == 0:
                running.discard(r)

    t0 = time.perf_counter()
    step = 0
    if mode == "sync":
        while running or pending:
            batch = schedule()
            gpu.jobs.put((step, batch)); gpu.done.get()
            postprocess(batch)
            step += 1
    else:
        batch = schedule()
        gpu.jobs.put((step, batch))
        while True:
            # GPU 在算第 step 步；CPU 提前调度第 step+1 步。此时第 step 步的结果还没出来，
            # 本步会结束的请求仍在 running 里，下一步它们会被多算一次
            nxt = schedule() if (running or pending) else None
            _, done_batch = gpu.done.get()
            if nxt is not None:
                gpu.jobs.put((step + 1, nxt))
            postprocess(done_batch)
            step += 1
            if nxt is None:
                break
            if not running and not pending:
                gpu.done.get()
                postprocess(nxt)
                break
    elapsed = time.perf_counter() - t0
    return elapsed, gpu.busy / elapsed, useful, wasted, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-ms", type=float, default=8.0, help="每步 GPU 前向的耗时")
    ap.add_argument("--sched-ms", type=float, default=2.5, help="每步调度与准备输入的耗时")
    ap.add_argument("--post-ms", type=float, default=2.0, help="每步后处理的耗时")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"每步：调度 {args.sched_ms} ms + GPU {args.gpu_ms} ms + 后处理 {args.post_ms} ms；"
          f"{args.requests} 个请求，batch 上限 {args.max_batch}\n")
    print(f"{'方式':8s}{'总耗时':>10s}{'GPU 利用率':>12s}{'每步':>10s}{'有效 token':>12s}{'多算的 token':>14s}")
    for mode, name in [("sync", "同步"), ("async", "异步")]:
        el, util, useful, wasted, steps = run(mode, args)
        print(f"{name:8s}{el:>8.2f} s{util:>11.0%}{el / steps * 1000:>8.1f} ms{useful:>12d}{wasted:>14d}")
    print("\n异步调度把 CPU 的两段工作藏到了 GPU 计算的背后，每一步的耗时接近纯 GPU 时间。")
    print("多算的 token 来自每个请求结束时多出的那一步，请求越长，这部分占比越小。")
    print("如果 CPU 的工作比 GPU 还慢（试试 --gpu-ms 2），异步也藏不住，这时要减少 CPU 工作本身。")


if __name__ == "__main__":
    main()
