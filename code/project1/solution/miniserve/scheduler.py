"""调度器：每一步决定哪些请求参与前向、各处理多少个 token。

三种模式，便于做对比实验：
- continuous + chunked_prefill（默认）：与 vLLM V1 相同的思路。先给 RUNNING 请求分配 token，
  再用剩余预算接纳 WAITING 请求；长 prompt 按预算切块，可以和 decode 混在同一步。
- continuous + 不切块：有新请求能接纳时，这一步只做 prefill（不带 decode）；否则只做 decode。
  这是早期 vLLM 的行为，会让正在 decode 的请求在别人 prefill 时停顿。
- static：只有当前批全部结束后才接纳下一批，用来对照 continuous batching 的收益。

显存不足时，抢占最晚到达的 RUNNING 请求：释放它的全部 KV 块，放回等待队列头部，之后重新 prefill。
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from .kv_cache import BlockAllocator
from .sequence import Sequence, SeqStatus


@dataclass
class SchedulerOutput:
    scheduled: List[Tuple[Sequence, int]]     # (请求, 本步处理的新 token 数)
    num_preempted: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.scheduled)


class Scheduler:
    def __init__(self, allocator: BlockAllocator, block_size: int, max_num_seqs: int,
                 max_num_batched_tokens: int, policy: str = "continuous", chunked_prefill: bool = True):
        assert policy in ("continuous", "static")
        self.allocator = allocator
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.policy = policy
        self.chunked_prefill = chunked_prefill
        self.waiting: deque = deque()
        self.running: List[Sequence] = []
        self.total_preemptions = 0

    # ---------- 队列操作 ----------
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence, reason: str):
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        self.allocator.free(seq.block_table)
        seq.block_table = []
        if seq in self.running:
            self.running.remove(seq)

    def abort(self, seq_id: str):
        for seq in list(self.running) + list(self.waiting):
            if seq.seq_id == seq_id:
                if seq in self.waiting:
                    self.waiting.remove(seq)
                self.finish(seq, "abort")
                return seq
        return None

    # ---------- 块管理 ----------
    def _blocks_needed(self, seq: Sequence, num_new: int) -> int:
        total = (seq.num_computed_tokens + num_new + self.block_size - 1) // self.block_size
        return max(0, total - len(seq.block_table))

    def _can_allocate(self, seq: Sequence, num_new: int) -> bool:
        return self._blocks_needed(seq, num_new) <= self.allocator.num_free

    def _allocate(self, seq: Sequence, num_new: int):
        for _ in range(self._blocks_needed(seq, num_new)):
            seq.block_table.append(self.allocator.allocate())

    def _preempt(self, seq: Sequence):
        self.allocator.free(seq.block_table)
        seq.block_table = []
        seq.num_computed_tokens = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.total_preemptions += 1

    # ---------- 主入口 ----------
    def schedule(self) -> SchedulerOutput:
        if self.policy == "static":
            return self._schedule_static()
        if not self.chunked_prefill:
            return self._schedule_no_chunk()
        return self._schedule_continuous()

    def _schedule_running(self, budget: int, max_tokens_per_seq=None):
        """给 RUNNING 请求分配 token；显存不足时从队尾抢占。返回 (scheduled, 剩余预算, 抢占数)。"""
        scheduled, preempted, i = [], 0, 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            if max_tokens_per_seq is not None:
                n = min(n, max_tokens_per_seq)
            while not self._can_allocate(seq, n):
                victim = self.running[-1]
                self._preempt(victim)
                preempted += 1
                if victim is seq:
                    break
            if seq.status != SeqStatus.RUNNING:
                break
            self._allocate(seq, n)
            scheduled.append((seq, n))
            budget -= n
            i += 1
        return scheduled, budget, preempted

    def _admit(self, budget: int, allow_partial: bool):
        """从 WAITING 队首依次接纳请求。"""
        scheduled = []
        while self.waiting and len(self.running) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            n = seq.num_tokens - seq.num_computed_tokens
            if n > budget and not allow_partial:
                break
            n = min(n, budget)
            if not self._can_allocate(seq, n):
                break
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            self._allocate(seq, n)
            scheduled.append((seq, n))
            budget -= n
        return scheduled

    def _schedule_continuous(self) -> SchedulerOutput:
        scheduled, budget, preempted = self._schedule_running(self.max_num_batched_tokens)
        if not preempted:      # 刚发生过抢占说明显存紧张，这一步不再接纳新请求
            scheduled += self._admit(budget, allow_partial=True)
        return SchedulerOutput(scheduled, preempted)

    def _schedule_no_chunk(self) -> SchedulerOutput:
        admitted = self._admit(self.max_num_batched_tokens, allow_partial=False)
        if admitted:           # 本步只做 prefill
            return SchedulerOutput(admitted)
        scheduled, _, preempted = self._schedule_running(self.max_num_batched_tokens)
        return SchedulerOutput(scheduled, preempted)

    def _schedule_static(self) -> SchedulerOutput:
        if not self.running:   # 上一批全部完成，才接纳下一批
            return SchedulerOutput(self._admit(self.max_num_batched_tokens, allow_partial=False))
        scheduled, _, preempted = self._schedule_running(self.max_num_batched_tokens)
        return SchedulerOutput(scheduled, preempted)
