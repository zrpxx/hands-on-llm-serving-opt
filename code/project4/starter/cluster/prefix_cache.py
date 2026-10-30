"""第 1 步：按块管理的前缀缓存，LRU 淘汰。对应 1.4 节。

缓存里存的是"已经算好的 KV 块"的键。一个请求到来时，从第一块开始逐块查找，遇到第一个不在缓存里的块就停止，
命中的是一个前缀。正在被运行中的请求使用的块（引用计数大于 0）不能淘汰。

数据结构与 vLLM 相同：ref 记录每个块的引用计数；free 只放引用计数为 0 的块，按最近使用的顺序排列，
淘汰时从最久未使用的一端取。这样淘汰一个块是 O(1) 的。
"""
from collections import OrderedDict


class PrefixCache:
    def __init__(self, capacity_blocks: int):
        self.capacity = capacity_blocks
        self.ref = {}                    # key -> 引用计数
        self.free = OrderedDict()        # 引用计数为 0 的块；越靠后越是最近使用的
        self.hits = 0                    # 累计命中的块数
        self.queries = 0                 # 累计查询的块数

    def __len__(self):
        return len(self.ref)

    def __contains__(self, key):
        return key in self.ref

    def match(self, keys) -> int:
        """返回从头开始连续命中的块数。命中的空闲块移到 free 的末尾（最近使用）。"""
        # TODO(第 1 步)：从第一块开始逐块查找，遇到不在 ref 中的块就停止；命中的空闲块在 free 中移到末尾。
        # 别忘了更新 self.hits 和 self.queries（查询数按 len(keys) 计）。
        raise NotImplementedError("TODO(第 1 步): match")

    def acquire(self, keys):
        """请求开始使用这些块：已有的块引用计数加一，不存在的块插入（空间不够时先淘汰）。"""
        # TODO(第 1 步)：已有的块：如果引用计数为 0，先从 free 中移除；引用计数加一。
        # 不存在的块：先 self.evict(1)，再以引用计数 1 插入。
        raise NotImplementedError("TODO(第 1 步): acquire")

    def release(self, keys):
        """请求结束：引用计数减一，减到 0 的块放进 free 的末尾。块仍然留在缓存里，供之后的请求命中。"""
        # TODO(第 1 步)：引用计数减一；减到 0 时放进 free 的末尾（值可以是 None）。
        raise NotImplementedError("TODO(第 1 步): release")

    def evict(self, need: int):
        """为 need 个新块腾出空间：从 free 中最久未使用的块开始淘汰。正在使用的块不淘汰，
        所以所有块都在使用时，缓存可能暂时超出容量。"""
        # TODO(第 1 步)：当 len(self.ref) + need 超过容量、且 free 不为空时，弹出 free 最前面的块并从 ref 中删除。
        raise NotImplementedError("TODO(第 1 步): evict")

    def hit_rate(self) -> float:
        return self.hits / max(1, self.queries)
