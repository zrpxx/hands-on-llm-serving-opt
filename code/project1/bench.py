"""压测客户端：向任意 OpenAI 兼容服务（MiniServe、vLLM、SGLang、HF 基线）发送流式请求并统计指标。

prompt 直接用随机 token id 构造，保证输入长度精确；配合 ignore_eos 固定输出长度。

示例：
    # 开环：泊松到达，平均 8 req/s
    python bench.py --base-url http://localhost:8000 --model qwen --num-prompts 200 \
        --input-len 512 --output-len 128 --request-rate 8 --vocab-size 32000

    # 闭环：固定 16 个并发
    python bench.py ... --request-rate inf --max-concurrency 16

    # 所有请求共享 1024 token 的前缀（用于 1.7 节的前缀缓存实验）
    python bench.py ... --shared-prefix-len 1024
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import httpx


@dataclass
class RequestResult:
    input_len: int
    output_len: int = 0
    ttft: float = 0.0
    e2e: float = 0.0
    itl: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def tpot(self) -> float:
        return (self.e2e - self.ttft) / (self.output_len - 1) if self.output_len > 1 else 0.0


async def one_request(client, url, model, prompt, max_tokens, sem) -> RequestResult:
    res = RequestResult(input_len=len(prompt))
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    async with sem:
        t0 = last = time.perf_counter()
        chunks = 0
        try:
            async with client.stream("POST", url, json=body) as r:
                if r.status_code != 200:
                    res.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                    return res
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if data.get("usage") and not data.get("choices"):
                        res.output_len = data["usage"]["completion_tokens"]
                        continue
                    if not data.get("choices"):
                        continue
                    now = time.perf_counter()
                    if chunks == 0:
                        res.ttft = now - t0
                    else:
                        res.itl.append(now - last)
                    last = now
                    chunks += 1
            res.e2e = time.perf_counter() - t0
            res.output_len = res.output_len or chunks
        except Exception as e:  # noqa: BLE001
            res.error = repr(e)
    return res


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def summarize(results: List[RequestResult], duration: float, slo_ttft: float, slo_tpot: float) -> dict:
    ok = [r for r in results if r.error is None]
    out_tokens = sum(r.output_len for r in ok)
    in_tokens = sum(r.input_len for r in ok)
    ttft = [r.ttft * 1000 for r in ok]
    tpot = [r.tpot * 1000 for r in ok if r.output_len > 1]
    itl = [x * 1000 for r in ok for x in r.itl]
    e2e = [r.e2e * 1000 for r in ok]
    good = [r for r in ok if r.ttft * 1000 <= slo_ttft and r.tpot * 1000 <= slo_tpot]
    s = {
        "completed": len(ok), "failed": len(results) - len(ok), "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": out_tokens / duration,
        "total_token_throughput": (in_tokens + out_tokens) / duration,
        "goodput_req_s": len(good) / duration,
        "slo": {"ttft_ms": slo_ttft, "tpot_ms": slo_tpot},
    }
    for name, xs in (("ttft_ms", ttft), ("tpot_ms", tpot), ("itl_ms", itl), ("e2e_ms", e2e)):
        s[name] = {"mean": statistics.fmean(xs) if xs else 0.0, "p50": pct(xs, 50), "p90": pct(xs, 90),
                   "p99": pct(xs, 99), "max": max(xs) if xs else 0.0}
    return s


def print_summary(s: dict):
    print(f"完成 {s['completed']} 个请求，失败 {s['failed']}，耗时 {s['duration_s']:.1f} s")
    print(f"请求吞吐   {s['request_throughput']:.2f} req/s")
    print(f"输出吞吐   {s['output_throughput']:.1f} tok/s")
    print(f"总吞吐     {s['total_token_throughput']:.1f} tok/s")
    print(f"Goodput    {s['goodput_req_s']:.2f} req/s  (TTFT ≤ {s['slo']['ttft_ms']} ms 且 TPOT ≤ {s['slo']['tpot_ms']} ms)")
    print(f"{'':10}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>10}")
    for k in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        v = s[k]
        print(f"{k:10}{v['mean']:>10.1f}{v['p50']:>10.1f}{v['p90']:>10.1f}{v['p99']:>10.1f}{v['max']:>10.1f}")


async def run(a) -> dict:
    rng = random.Random(a.seed)
    lo = max(1, int(a.input_len * (1 - a.range_ratio)))
    hi = int(a.input_len * (1 + a.range_ratio))
    prefix = [rng.randrange(10, a.vocab_size) for _ in range(a.shared_prefix_len)]
    prompts = [prefix + [rng.randrange(10, a.vocab_size) for _ in range(rng.randint(lo, hi))]
               for _ in range(a.num_prompts)]
    out_lo = max(1, int(a.output_len * (1 - a.range_ratio)))
    out_hi = int(a.output_len * (1 + a.range_ratio))
    out_lens = [rng.randint(out_lo, out_hi) for _ in range(a.num_prompts)]

    url = a.base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(a.max_concurrency or 10**9)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        tasks = []
        t0 = time.perf_counter()
        for p, n in zip(prompts, out_lens):
            tasks.append(asyncio.create_task(one_request(client, url, a.model, p, n, sem)))
            if a.request_rate != float("inf"):
                await asyncio.sleep(rng.expovariate(a.request_rate))     # 泊松过程：间隔服从指数分布
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0
    s = summarize(results, duration, a.slo_ttft_ms, a.slo_tpot_ms)
    s["config"] = {k: (str(v) if v == float("inf") else v) for k, v in vars(a).items()}
    errors = [r.error for r in results if r.error]
    if errors:
        print("部分请求失败，示例：", errors[0])
    if a.save:
        json.dump({"summary": s, "requests": [asdict(r) for r in results]}, open(a.save, "w"), indent=1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="default")
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--range-ratio", type=float, default=0.0, help="长度在 [1-r, 1+r] 倍之间均匀随机")
    ap.add_argument("--shared-prefix-len", type=int, default=0)
    ap.add_argument("--request-rate", type=float, default=float("inf"))
    ap.add_argument("--max-concurrency", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=32000, help="随机 token id 的上界，避开特殊 token")
    ap.add_argument("--slo-ttft-ms", type=float, default=1000)
    ap.add_argument("--slo-tpot-ms", type=float, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    print_summary(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
