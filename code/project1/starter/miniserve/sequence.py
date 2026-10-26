import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

_ids = itertools.count()


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0          # 0 表示贪心解码
    top_p: float = 1.0
    ignore_eos: bool = False          # 压测时用来固定输出长度
    stop_token_ids: List[int] = field(default_factory=list)


class Sequence:
    """一个生成请求在引擎内部的全部状态。"""

    def __init__(self, prompt_token_ids: List[int], params: SamplingParams, seq_id: Optional[str] = None):
        self.seq_id = seq_id or f"seq-{next(_ids)}"
        self.params = params
        self.prompt_len = len(prompt_token_ids)
        self.token_ids: List[int] = list(prompt_token_ids)   # 输入 + 已生成
        self.status = SeqStatus.WAITING
        self.num_computed_tokens = 0          # 已写入 KV Cache 的 token 数
        self.block_table: List[int] = []      # 逻辑块 i -> 物理块号
        self.finish_reason: Optional[str] = None
        self.num_preemptions = 0
        self.arrival_time = time.perf_counter()
        self.first_token_time: Optional[float] = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def output_token_ids(self) -> List[int]:
        return self.token_ids[self.prompt_len:]

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    def __repr__(self):
        return (f"Sequence({self.seq_id}, {self.status.value}, tokens={self.num_tokens}, "
                f"computed={self.num_computed_tokens}, blocks={len(self.block_table)})")
