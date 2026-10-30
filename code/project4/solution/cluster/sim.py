"""离散事件模拟：请求到达、实例的每一步结束、PD 分离时的 KV 传输完成，按时间顺序处理。"""
import heapq
import itertools

from .workload import percentile


class Cluster:
    """聚合部署：每个实例都做 prefill 和 decode，路由器为每个请求选一个实例。"""

    def __init__(self, instances, router):
        self.instances, self.router = instances, router
        self.events = []
        self._seq = itertools.count()

    def push(self, t, kind, payload):
        heapq.heappush(self.events, (t, next(self._seq), kind, payload))

    def _kick(self, inst, now):
        if inst.busy_until is None:
            t = inst.start_step(now)
            if t is not None:
                self.push(t, "step", inst)

    def on_arrival(self, req, now):
        inst = self.instances[self.router.choose(req, self.instances)]
        inst.enqueue(req, now)
        self._kick(inst, now)

    def on_step(self, inst, now):
        for req in inst.finish_step(now):
            self.on_prefill_done(req, now)
        self._kick(inst, now)

    def on_prefill_done(self, req, now):
        raise RuntimeError("聚合部署中不应该出现 prefill 交接")

    def on_event(self, kind, payload, now):
        raise ValueError(kind)

    def run(self, requests):
        for r in requests:
            self.push(r.arrival, "arrive", r)
        now = 0.0
        while self.events:
            now, _, kind, payload = heapq.heappop(self.events)
            if kind == "arrive":
                self.on_arrival(payload, now)
            elif kind == "step":
                self.on_step(payload, now)
            else:
                self.on_event(kind, payload, now)
            for inst in self.all_instances():           # 有新的 decode 请求等待时唤醒空闲实例
                if inst.busy_until is None and inst.waiting:
                    self._kick(inst, now)
        return summarize(requests, self.all_instances(), now)

    def all_instances(self):
        return self.instances


def summarize(requests, instances, end_time):
    done = [r for r in requests if r.finish is not None]
    ttft = [r.first_token - r.arrival for r in done]
    gaps = [b - a for r in done for a, b in zip(r.token_times, r.token_times[1:])]
    tpot = [(r.finish - r.first_token) / max(1, len(r.token_times) - 1) for r in done]
    second = [r.token_times[1] - r.token_times[0] for r in done if len(r.token_times) > 1]
    hits = sum(r.cached_tokens for r in done)
    total_in = sum(r.input_len for r in done)
    busy = [inst.busy_time / max(end_time, 1e-9) for inst in instances]
    return {
        "完成": len(done), "总数": len(requests),
        "TTFT_P50": percentile(ttft, 50), "TTFT_P99": percentile(ttft, 99),
        "ITL_P50": percentile(gaps, 50), "ITL_P99": percentile(gaps, 99),
        "TPOT_P50": percentile(tpot, 50),
        "第二个token_P99": percentile(second, 99),
        "命中率": hits / max(1, total_in),
        "忙碌比例": busy,
        "吞吐": sum(len(r.token_times) for r in done) / max(end_time, 1e-9),
        "_requests": done,
    }


def goodput(result, ttft_slo: float, tpot_slo: float) -> float:
    """第 5 步：同时满足 TTFT 和 TPOT 目标的请求所占的比例（0.8 节）。"""
    reqs = result["_requests"]
    ok = [r for r in reqs if r.first_token - r.arrival <= ttft_slo
          and (r.finish - r.first_token) / max(1, len(r.token_times) - 1) <= tpot_slo]
    return len(ok) / max(1, result["总数"])
