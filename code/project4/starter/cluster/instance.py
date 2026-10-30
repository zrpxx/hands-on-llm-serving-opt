"""一个推理实例：连续批处理 + chunked prefill + 前缀缓存。对应 1.2、1.4 节。

role="both"：普通的实例，prefill 和 decode 在一起（聚合部署）。
role="prefill"：只做 prefill，算完第一个 token 后把请求交给集群（PD 分离）。
role="decode"：只做 decode，接收已经传来 KV 的请求。
"""
from .costmodel import H100, QWEN3_8B, kv_capacity_tokens, step_time
from .prefix_cache import PrefixCache
from .workload import BLOCK


class _Seq:
    __slots__ = ("req", "keys", "computed", "need", "gen", "reserved")

    def __init__(self, req, keys, computed, reserved):
        self.req, self.keys, self.computed, self.reserved = req, keys, computed, reserved
        self.need = req.input_len          # prefill 要算到的长度
        self.gen = 0                       # 已生成的 token 数


class Instance:
    def __init__(self, iid: int, hw=H100, model=QWEN3_8B, role: str = "both", max_batch_tokens: int = 2048,
                 max_seqs: int = 256, cache_frac: float = 0.4, prefix_caching: bool = True):
        self.iid, self.hw, self.model, self.role = iid, hw, model, role
        self.max_batch_tokens, self.max_seqs = max_batch_tokens, max_seqs
        cap = kv_capacity_tokens(hw, model)
        self.kv_capacity = int(cap * (1 - cache_frac))       # 留给运行中请求独有部分的 KV
        self.cache = PrefixCache(int(cap * cache_frac) // BLOCK) if prefix_caching else None
        self.waiting, self.running = [], []
        self.kv_used = 0
        self.busy_until = None              # 当前这一步结束的时间；None 表示空闲
        self._pending = None                # 当前这一步的计划
        self.busy_time = 0.0
        self.steps = 0

    # ------------------------------------------------------------------ 供路由器使用的负载信号
    def queued_tokens(self) -> int:
        """还没算完的 prefill token 数，加上运行中序列还要生成的 token 数。"""
        pf = sum(s.need - s.computed for s in self.running) + sum(r.input_len for r in self.waiting)
        dec = sum(s.req.output_len - s.gen for s in self.running) + sum(r.output_len for r in self.waiting)
        return pf + dec

    def num_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def cached_prefix_tokens(self, req) -> int:
        """不改变 LRU 顺序地估计这个请求能命中多少 token，供路由器打分。"""
        if self.cache is None:
            return 0
        n = 0
        for k in req.block_keys():
            if k not in self.cache:
                break
            n += 1
        return n * BLOCK

    # ------------------------------------------------------------------ 接收请求
    def enqueue(self, req, now: float):
        req.instance = self.iid
        self.waiting.append(req)

    def enqueue_decode(self, req, now: float):
        """PD 分离时，decode 实例接收一个已经有 KV、已经产出第一个 token 的请求。"""
        req.instance = self.iid
        req._decode_ready = True
        self.waiting.append(req)

    # ------------------------------------------------------------------ 一步
    def _admit_decode_ready(self):
        """decode 实例：接收 KV 已经传到的请求，只要 KV 空间够。按到达顺序，不插队。"""
        while self.waiting and getattr(self.waiting[0], "_decode_ready", False) and len(self.running) < self.max_seqs:
            req = self.waiting[0]
            need_kv = req.input_len + req.output_len
            if self.kv_used + need_kv > self.kv_capacity:
                break
            self.waiting.pop(0)
            s = _Seq(req, [], req.input_len, need_kv)
            s.gen = 1
            self.kv_used += need_kv
            self.running.append(s)

    def _admit_new(self, budget: int):
        """接收新请求做 prefill，直到这一步的 token 预算用完。返回新加入的序列。"""
        new = []
        while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
            req = self.waiting[0]
            if getattr(req, "_decode_ready", False):
                break
            keys = req.block_keys()
            hit = self.cache.match(keys) * BLOCK if self.cache is not None else 0
            hit = min(hit, req.input_len - 1)                   # 至少算最后一个 token，才能得到 logits
            need_kv = req.input_len - hit + (req.output_len if self.role != "prefill" else 0)
            if self.kv_used + need_kv > self.kv_capacity:
                break
            self.waiting.pop(0)
            if self.cache is not None:
                self.cache.acquire(keys)
            req.cached_tokens = hit
            self.kv_used += need_kv
            s = _Seq(req, keys, hit, need_kv)
            self.running.append(s)
            new.append(s)
            budget -= s.need - s.computed
        return new

    def start_step(self, now: float):
        """如果有活干，规划这一步并返回结束时间；否则返回 None。"""
        if self.busy_until is not None:
            return self.busy_until
        self._admit_decode_ready()
        budget = self.max_batch_tokens
        decoders = [] if self.role == "prefill" else [s for s in self.running if s.computed >= s.need]
        budget -= len(decoders)
        prefills = []
        for s in self.running:                                  # 先继续已经开始的 prefill
            if s.computed < s.need and budget > 0:
                c = min(budget, s.need - s.computed)
                prefills.append((s, c))
                budget -= c
        for s in self._admit_new(budget):                       # 再接收新请求
            c = min(budget, s.need - s.computed)
            if c > 0:
                prefills.append((s, c))
                budget -= c
        if not decoders and not prefills:
            return None
        kv = sum(s.computed + s.gen for s, _ in prefills) + sum(s.computed + s.gen for s in decoders)
        dt = step_time(self.hw, self.model, sum(c for _, c in prefills), len(decoders), kv)
        self._pending = (decoders, prefills)
        self.busy_until = now + dt
        self.busy_time += dt
        self.steps += 1
        return self.busy_until

    def finish_step(self, now: float):
        """应用这一步的结果。返回 prefill 实例上算完了 prefill、需要交给 decode 实例的请求列表。"""
        decoders, prefills = self._pending
        self._pending, self.busy_until = None, None
        handoff = []
        for s in decoders:
            s.gen += 1
            s.req.token_times.append(now)
        for s, c in prefills:
            s.computed += c
            if s.computed >= s.need:
                s.gen = 1
                s.req.first_token = now
                s.req.token_times.append(now)
                if self.role == "prefill":
                    handoff.append(s.req)
        done = [s for s in self.running if (s.gen >= s.req.output_len and s.computed >= s.need)
                or (self.role == "prefill" and s.req in handoff)]
        for s in done:
            self.running.remove(s)
            self.kv_used -= s.reserved
            if self.cache is not None and s.keys:
                self.cache.release(s.keys)
            if self.role != "prefill":
                s.req.finish = now
        return handoff
