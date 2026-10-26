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
        # TODO(1-3)：
        # 该请求在已计算 num_computed_tokens 的基础上再处理 num_new 个 token，还需要新分配几个块。
        raise NotImplementedError("TODO(1-3): Scheduler._blocks_needed")

    def _can_allocate(self, seq: Sequence, num_new: int) -> bool:
        return self._blocks_needed(seq, num_new) <= self.allocator.num_free

    def _allocate(self, seq: Sequence, num_new: int):
        # TODO(1-3)：
        # 按 _blocks_needed 的数量分配块，追加到 seq.block_table。
        raise NotImplementedError("TODO(1-3): Scheduler._allocate")

    def _preempt(self, seq: Sequence):
        # TODO(1-3)：
        # 抢占：释放全部块、清零 num_computed_tokens、状态改回 WAITING、放回等待队列头部，并更新计数。
        raise NotImplementedError("TODO(1-3): Scheduler._preempt")

    # ---------- 主入口 ----------
    def schedule(self) -> SchedulerOutput:
        if self.policy == "static":
            return self._schedule_static()
        if not self.chunked_prefill:
            return self._schedule_no_chunk()
        return self._schedule_continuous()

    def _schedule_running(self, budget: int, max_tokens_per_seq=None):
        """给 RUNNING 请求分配 token；显存不足时从队尾抢占。返回 (scheduled, 剩余预算, 抢占数)。"""
        # TODO(1-2 / 1-3)：
        # 按顺序给 RUNNING 请求分配 token（decode 为 1 个，未完成的 prefill 为剩余部分，受预算限制）；
        # 块不够时从队尾抢占，直到够用或轮到自己被抢占。返回 (scheduled, 剩余预算, 抢占数)。
        raise NotImplementedError("TODO(1-2 / 1-3): Scheduler._schedule_running")

    def _admit(self, budget: int, allow_partial: bool):
        """从 WAITING 队首依次接纳请求。"""
        # TODO(1-2)：
        # 从 WAITING 队首依次接纳请求，直到达到 max_num_seqs、预算耗尽或块不够。
        # allow_partial=True 时允许只处理 prompt 的一部分（chunked prefill）。
        raise NotImplementedError("TODO(1-2): Scheduler._admit")

    def _schedule_continuous(self) -> SchedulerOutput:
        # TODO(1-2)：
        # 先调度 RUNNING，再用剩余预算接纳 WAITING；本步发生过抢占时不接纳新请求。
        raise NotImplementedError("TODO(1-2): Scheduler._schedule_continuous")

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
