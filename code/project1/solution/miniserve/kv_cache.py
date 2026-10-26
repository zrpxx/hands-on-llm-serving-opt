from collections import deque
from typing import List, Tuple

import torch

from .config import ModelConfig


class BlockAllocator:
    """管理物理 KV 块的空闲列表。块号从 0 到 num_blocks-1。"""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("没有空闲的 KV 块")
        return self._free.popleft()

    def free(self, blocks: List[int]):
        self._free.extend(blocks)


def allocate_kv_cache(cfg: ModelConfig, num_blocks: int, block_size: int,
                      dtype: torch.dtype, device: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """每层一对 [num_blocks * block_size, n_kv, head_dim] 的张量。

    把块和块内偏移展平成"槽位"（slot = block * block_size + offset），
    这样写入和读取都可以用一次高级索引完成。
    """
    shape = (num_blocks * block_size, cfg.num_kv_heads, cfg.head_dim)
    return [(torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
            for _ in range(cfg.num_layers)]
