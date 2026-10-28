# 项目 1 分步指南

> 配合 [项目 1　MiniServe](stage1/boss-miniserve.md) 使用。项目页说明"做什么"，这一页说明"按什么顺序、怎么一步步做出来"。

## 这份指南怎么用

整个项目拆成 11 步，每一步只改一两个函数，并且**有一组只针对这一步的测试**。做完一步、测试变绿，再进入下一步。

| 步 | 内容 | 文件 | 验证命令 | 参考用时 |
|---|---|---|---|---|
| 0 | 准备环境，跟着一个请求读一遍代码 | — | `MINISERVE_IMPL=solution pytest -q tests` | 45 分钟 |
| 1 | 物理块分配器 | `kv_cache.py` | `pytest tests/test_steps.py -k step01` | 10 分钟 |
| 2 | 计算并分配一个请求需要的块 | `scheduler.py` | `-k step02` | 20 分钟 |
| 3 | 从等待队列接纳新请求 | `scheduler.py` | `-k step03` | 30 分钟 |
| 4 | 给运行中的请求分配 token | `scheduler.py` | `-k step04` | 20 分钟 |
| 5 | 把两者组合成一次调度 | `scheduler.py` | `-k step05` | 10 分钟 |
| 6 | 显存不足时抢占 | `scheduler.py` | `-k step06` | 30 分钟 |
| 7 | 把一批请求展平成模型输入 | `engine.py` | `-k step07` | 60 分钟 |
| 8 | 分页注意力 | `model.py` | `-k step08` | 60 分钟 |
| 9 | 端到端正确性 | — | `pytest tests/test_correctness.py tests/test_scheduler.py` | 视调试情况 |
| 10 | 后台引擎循环与 token 分发 | `server.py` | `-k step10`，然后 `pytest tests/test_server.py` | 30 分钟 |
| 11 | 在真实模型上跑起来并压测 | — | `bench.py` | 60 分钟 |

所有命令都在 `code/project1/` 目录下执行。第 0~10 步只需要 CPU。

每一步的结构相同：

1. **目标**：一句话说清楚这个函数做什么。
2. **先想清楚**：动手前回答两三个问题，答案折叠在下面。答不上来就回到对应章节。
3. **约定**：函数的输入、输出和副作用，测试按这个约定检查。
4. **三级提示**：思路 → 伪代码 → 参考代码，都默认折叠。建议卡住 15 分钟以上再打开下一级。

**两个调试工具，整个项目都会用到：**

```bash
# 1) 逐步打印调度器状态。先用参考实现看"应该是什么样"，再用自己的实现对比
python tools/trace.py --impl solution --prompts 6,6,20 --budget 10 --blocks 12
python tools/trace.py --prompts 6,6,20 --budget 10 --blocks 12

# 2) 只跑某一步的测试，失败时显示完整的断言信息
pytest tests/test_steps.py -k step03 -x
```

---

## 第 0 步：准备环境，读一遍代码

### 0.1 准备环境

```bash
cd code/project1
pip install -r requirements.txt
python tools/make_tiny_model.py --out /tmp/tiny --layers 4 --hidden 128

MINISERVE_IMPL=solution pytest -q tests     # 参考实现：应全部通过，确认环境没问题
pytest -q tests/test_steps.py               # 你的 starter：此时应全部失败（NotImplementedError）
```

### 0.2 先看参考实现在做什么

在写任何代码之前，先观察目标行为。运行：

```bash
python tools/trace.py --impl solution --prompts 6,6,20 --budget 10 --blocks 12 --steps 6
```

```text
第 1 步  本步 token 10/10  空闲块 9/12
  seq phase       n   computed  len  block_table
  A   prefill     6       0->6    7  [0, 1]  -> 采样 0
  B   chunk       4       0->4    6  [2]
  running=['A', 'B']  waiting=['C']

第 2 步  本步 token 10/10  空闲块 6/12
  seq phase       n   computed  len  block_table
  A   decode      1       6->7    8  [0, 1]  -> 采样 0
  B   prefill     2       4->6    7  [2, 3]  -> 采样 0
  C   chunk       7       0->7   20  [4, 5]
  running=['A', 'B', 'C']  waiting=[]
```

逐行读懂它：

- 第 1 步预算 10：A 的 6 个 prompt token 全部处理，剩下的 4 个给 B，B 只处理了一半（chunk）。C 一个都没分到，留在等待队列。
- A 处理完 prompt，立即采样出第 1 个输出 token，总长从 6 变为 7。B 没处理完，不采样。
- 第 2 步先照顾运行中的请求：A 在 decode，分 1 个；B 还剩 2 个 prompt token；剩下的 7 个预算接纳 C。
- 块表：块大小为 4，A 的 6 个 token 需要 2 块，B 的 4 个 token 需要 1 块，第 2 步 B 到第 6 个 token 时再要 1 块。

改几个参数再观察，直到你能**提前预测**每一步的输出：

```bash
python tools/trace.py --impl solution --prompts 6,6,20 --budget 10 --blocks 12 --no-chunk   # 不切块：C 永远进不来，想想为什么
python tools/trace.py --impl solution --prompts 6,6,20 --budget 32 --blocks 12 --no-chunk   # 预算足够时再看一次
python tools/trace.py --impl solution --prompts 8,8 --budget 100 --blocks 6 --max-tokens 12  # 块不够，会抢占
python tools/trace.py --impl solution --prompts 4,4,4 --arrive 1,1,3 --policy static         # 静态批处理
```

### 0.3 跟着一个请求读代码

按下面的顺序读 `starter/miniserve/`，每读一处回答右边的问题：

| 顺序 | 位置 | 要搞清楚的 |
|---|---|---|
| 1 | `sequence.py` 的 `Sequence` | 有哪些字段？`token_ids` 里放的是什么？`num_tokens` 和 `num_computed_tokens` 分别是什么意思？ |
| 2 | `engine.py` 的 `add_request` | 请求进来后放到了哪里？ |
| 3 | `engine.py` 的 `step` | 一步分为哪几个阶段？哪里调用了调度器、哪里调用了 `_prepare_inputs`、哪里更新了 `num_computed_tokens`？ |
| 4 | `scheduler.py` 开头的说明与 `schedule` | 三种模式分别调用哪个函数？ |
| 5 | `scheduler.py` 的 `finish` | 请求结束时做了哪些清理？ |
| 6 | `model.py` 的 `AttentionMetadata` | 注意力需要哪些信息？为什么 prefill 和 decode 分成两组？ |

<details>
<summary>参考：一个请求的完整生命周期</summary>

1. `add_request` 创建 `Sequence`，`token_ids` 为 prompt，`num_computed_tokens = 0`，放入 `scheduler.waiting`。
2. 某一步 `schedule()` 中，`_admit` 把它从 `waiting` 移到 `running`，分给它 $n$ 个 token，并分配足够的 KV 块。
3. `step()` 中 `_prepare_inputs` 把这 $n$ 个 token 和其他请求的 token 拼在一起，算出每个 token 写入 KV 的槽位。
4. 模型前向，把这 $n$ 个 token 的 K、V 写入它的块。
5. `step()` 把 `num_computed_tokens` 加上 $n$。如果等于 `num_tokens`（所有已知 token 都在 KV 里了），就采样一个新 token 追加到 `token_ids`，此时 `num_tokens` 加 1。
6. 下一步它还在 `running` 中，差值为 1，就是 decode。
7. 生成够了，`finish` 释放它的块，把它从 `running` 移除。

整个调度的核心就是一个差值：**本步要处理的 token 数 = `num_tokens - num_computed_tokens`**，再受预算限制。新请求、切块中的 prefill、decode、被抢占后的恢复，都由这一个式子统一表达。

</details>

---

## 第 1 步：物理块分配器

**目标**：管理一个空闲块列表，能分出一块、能归还一批。

**文件**：`kv_cache.py` 中的 `BlockAllocator.allocate`、`BlockAllocator.free`

**先想清楚**

1. 构造函数里已经有 `self._free = deque(range(num_blocks))`。为什么用 `deque` 而不是 `list`？
2. 空闲块不够时，`allocate` 应该返回什么？

<details>
<summary>答案</summary>

1. 从左边取、往右边放，`deque` 两端操作都是 O(1)；`list.pop(0)` 是 O(n)。
2. 抛出 `RuntimeError`。调用方（调度器）应该先用 `_can_allocate` 检查，所以真的不够时属于程序错误，应该让它暴露出来。

</details>

**约定**

| 函数 | 输入 | 输出 / 副作用 |
|---|---|---|
| `allocate()` | 无 | 返回一个 `int` 块号，并从空闲列表移除；没有空闲块时抛 `RuntimeError` |
| `free(blocks)` | 块号列表 | 把这些块放回空闲列表 |

**验证**：`pytest tests/test_steps.py -k step01`

<details>
<summary>提示 1：思路</summary>

两个函数都只有一两行：`allocate` 从 `self._free` 左边弹出一个，`free` 把列表整体追加到右边。

</details>

<details>
<summary>参考代码</summary>

```python
def allocate(self) -> int:
    if not self._free:
        raise RuntimeError("没有空闲的 KV 块")
    return self._free.popleft()

def free(self, blocks: List[int]):
    self._free.extend(blocks)
```

</details>

---

## 第 2 步：计算并分配一个请求需要的块

**目标**：一个请求在已经算了 `num_computed_tokens` 个 token 的基础上，再处理 `num_new` 个，算出还需要几个新块，并把它们分配到 `seq.block_table`。

**文件**：`scheduler.py` 中的 `_blocks_needed`、`_allocate`

**先想清楚**

1. 块大小为 4。一个请求已算 5 个 token、已有 2 块，再算 1 个，需要新块吗？再算 4 个呢？
2. 为什么函数的参数是"再算多少个"，而不是"总共多少个"？

<details>
<summary>答案</summary>

1. 再算 1 个：总共 6 个，需要 $\lceil 6/4 \rceil = 2$ 块，已有 2 块，不需要。再算 4 个：总共 9 个，需要 3 块，再要 1 块。
2. 调度器知道的是"这一步给它分多少 token"，用增量最直接；总量可以从 `num_computed_tokens + num_new` 算出。

</details>

**约定**

| 函数 | 输出 / 副作用 |
|---|---|
| `_blocks_needed(seq, num_new)` | 返回还需要新分配的块数（不小于 0）。不修改任何状态 |
| `_allocate(seq, num_new)` | 分配 `_blocks_needed` 个块，依次追加到 `seq.block_table` |

已经写好的 `_can_allocate(seq, num_new)` 会调用你的 `_blocks_needed`，判断空闲块是否够用。

**验证**：`pytest tests/test_steps.py -k step02`

<details>
<summary>提示 1：思路</summary>

总 token 数 $T$ = `seq.num_computed_tokens + num_new`。需要的总块数是 $\lceil T / B \rceil$，用整数写作 `(T + B - 1) // B`。减去已有的 `len(seq.block_table)`，再和 0 取最大值。

`_allocate` 循环调用 `self.allocator.allocate()` 即可。

</details>

<details>
<summary>参考代码</summary>

```python
def _blocks_needed(self, seq: Sequence, num_new: int) -> int:
    total = (seq.num_computed_tokens + num_new + self.block_size - 1) // self.block_size
    return max(0, total - len(seq.block_table))

def _allocate(self, seq: Sequence, num_new: int):
    for _ in range(self._blocks_needed(seq, num_new)):
        seq.block_table.append(self.allocator.allocate())
```

</details>

---

## 第 3 步：从等待队列接纳新请求

**目标**：从 `waiting` 队头开始，依次把请求移入 `running`，并分给它本步要处理的 token 数，直到某个条件不满足。

**文件**：`scheduler.py` 中的 `_admit(budget, allow_partial)`

**先想清楚**

1. 有哪几个条件会让接纳停止？
2. `allow_partial=True` 和 `False` 分别对应什么？队头的请求放不下时，能不能跳过它去看下一个？
3. 预算为 10，等待队列里有两个 6 token 的 prompt，`allow_partial=True` 时结果是什么？`False` 呢？

<details>
<summary>答案</summary>

1. 等待队列空了；`running` 已有 `max_num_seqs` 个；预算用完；空闲块不够。
2. `True` 对应 chunked prefill：放不下整个 prompt 就先处理一部分。`False` 时整个 prompt 必须一次放下，放不下就停止。不跳过：按到达顺序（FCFS）接纳，跳过队头会让长请求一直饿着。
3. `True`：第一个处理 6 个，第二个处理 4 个。`False`：只接纳第一个，第二个留在队列里。

</details>

**约定**

- 返回本步新接纳的 `[(seq, n), ...]`。
- 被接纳的请求：从 `self.waiting` 移除，`status` 设为 `SeqStatus.RUNNING`，追加到 `self.running`，并为这 `n` 个 token 分配好块。
- 本步要处理的 token 数：`n = seq.num_tokens - seq.num_computed_tokens`，允许切块时再与 `budget` 取最小值。
- 块不够时**停止接纳**，不要抛异常。

**验证**：`pytest tests/test_steps.py -k step03`

<details>
<summary>提示 1：思路</summary>

一个 `while` 循环，循环条件写上三个"还能继续"的条件（队列非空、未达并发上限、预算大于 0）。循环体里先看队头 `self.waiting[0]`，算出 `n`；不允许切块且放不下就 `break`；块不够也 `break`；都没问题再真正弹出、改状态、分配块、扣预算。

关键是"先检查，再修改"：不满足条件时，队列和请求的状态都不能被改动。

</details>

<details>
<summary>提示 2：伪代码</summary>

```text
scheduled = []
while 等待队列非空 and running 数 < max_num_seqs and budget > 0:
    seq = 队头（先不弹出）
    n = seq 还差的 token 数
    if n > budget and 不允许切块: break
    n = min(n, budget)
    if 块不够: break
    弹出队头；状态改为 RUNNING；加入 running；分配块
    scheduled 追加 (seq, n)；budget -= n
return scheduled
```

</details>

<details>
<summary>参考代码</summary>

```python
def _admit(self, budget: int, allow_partial: bool):
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
```

</details>

**常见错误**：先 `popleft()` 再检查条件，检查失败后请求就从队列里"消失"了。

---

## 第 4 步：给运行中的请求分配 token

**目标**：按顺序遍历 `running`，给每个请求分配本步要处理的 token 数，直到预算用完。

**文件**：`scheduler.py` 中的 `_schedule_running(budget, max_tokens_per_seq=None)`

这一步**先不处理抢占**：块不够时直接停止循环。第 6 步再回来把它改成抢占。`max_tokens_per_seq` 参数用不到，忽略即可。

**先想清楚**

1. 运行中的请求有哪两种状态？各分几个 token？
2. 为什么运行中的请求要比等待中的请求优先？

<details>
<summary>答案</summary>

1. decode（差值为 1）和 prompt 还没算完的切块 prefill（差值为剩余 prompt 长度）。用同一个差值公式即可，不需要区分。
2. 它们已经占着 KV 块，用户已经在等后续 token；优先保证它们继续，TPOT 才稳定，已开始的请求不会饿死（1.6 节第 3 部分）。

</details>

**约定**

- 返回一个**三元组** `(scheduled, 剩余预算, 抢占数)`。这一步抢占数恒为 0。
- 对每个被调度的请求，要先为它的 `n` 个 token 分配好块。

**验证**：`pytest tests/test_steps.py -k step04`

<details>
<summary>提示 1：思路</summary>

和第 3 步很像，但遍历的是 `self.running`，而且请求不需要移动队列。用下标 `i` 遍历（不要用 `for seq in self.running`），因为第 6 步加入抢占后，`running` 会在循环中被修改。

</details>

<details>
<summary>参考代码（不含抢占的版本）</summary>

```python
def _schedule_running(self, budget: int, max_tokens_per_seq=None):
    scheduled, preempted, i = [], 0, 0
    while i < len(self.running) and budget > 0:
        seq = self.running[i]
        n = min(seq.num_tokens - seq.num_computed_tokens, budget)
        if not self._can_allocate(seq, n):
            break                      # 第 6 步把这里改成抢占
        self._allocate(seq, n)
        scheduled.append((seq, n))
        budget -= n
        i += 1
    return scheduled, budget, preempted
```

</details>

---

## 第 5 步：组合成一次调度

**目标**：先调度运行中的请求，再用剩余预算接纳新请求。

**文件**：`scheduler.py` 中的 `_schedule_continuous()`

**约定**：返回 `SchedulerOutput(scheduled, num_preempted)`。允许切块。**这一步如果发生过抢占，就不再接纳新请求**（说明显存已经紧张）。

**验证**：`pytest tests/test_steps.py -k step05`，然后运行追踪工具，和第 0 步看到的参考输出对比：

```bash
python tools/trace.py --prompts 6,6,20 --budget 10 --blocks 12 --steps 6
```

到这里，`--no-chunk` 和 `--policy static` 两种模式也能用了，因为它们复用了你写的 `_admit` 和 `_schedule_running`。

<details>
<summary>参考代码</summary>

```python
def _schedule_continuous(self) -> SchedulerOutput:
    scheduled, budget, preempted = self._schedule_running(self.max_num_batched_tokens)
    if not preempted:
        scheduled += self._admit(budget, allow_partial=True)
    return SchedulerOutput(scheduled, preempted)
```

</details>

---

## 第 6 步：显存不足时抢占

**目标**：运行中的请求需要新块而空闲块不够时，从 `running` 的**队尾**选请求抢占，释放它的块，直到够用。

**文件**：`scheduler.py` 中的 `_preempt(seq)`，以及修改第 4 步的 `_schedule_running`

**先想清楚**

1. 为什么选队尾？
2. 抢占后要把哪些状态恢复成"刚到达"的样子？`token_ids` 要不要清空？
3. 如果队尾恰好就是当前正在处理的这个请求，怎么办？

<details>
<summary>答案</summary>

1. `running` 按接纳顺序排列，队尾是最晚开始的请求，已投入的计算最少，抢占它的损失最小，也符合先来先服务。
2. 释放全部块并清空 `block_table`，`num_computed_tokens` 清零，`status` 改回 `WAITING`，从 `running` 移除，放回 `waiting` 的**头部**（它比后来的请求更早到达）。`token_ids` 不清空：已经生成的 token 要保留，恢复时连同 prompt 一起重新 prefill，用户看到的输出不受影响。
3. 抢占它自己，然后停止本步对运行中请求的调度。

</details>

**约定**

`_preempt(seq)`：完成上面第 2 题中的全部状态变化，并把 `seq.num_preemptions` 和 `self.total_preemptions` 各加 1。

`_schedule_running` 中：块不够时循环抢占队尾，直到够用或被抢占的就是当前请求；后者发生时停止循环。返回值中的抢占数为实际抢占的个数。

**验证**：`pytest tests/test_steps.py -k step06`，然后 `pytest tests/test_scheduler.py` 应全部通过。用追踪工具观察一次抢占：

```bash
python tools/trace.py --prompts 8,8 --budget 100 --blocks 6 --max-tokens 12
```

<details>
<summary>提示 1：思路</summary>

把第 4 步中的 `if not can_allocate: break` 换成一个内层循环：

```text
while 当前请求的块不够:
    victim = running 的最后一个
    抢占 victim
    if victim 就是当前请求: 跳出内层循环
if 当前请求已经不是 RUNNING: 跳出外层循环
```

</details>

<details>
<summary>参考代码</summary>

```python
def _preempt(self, seq: Sequence):
    self.allocator.free(seq.block_table)
    seq.block_table = []
    seq.num_computed_tokens = 0
    seq.status = SeqStatus.WAITING
    seq.num_preemptions += 1
    self.running.remove(seq)
    self.waiting.appendleft(seq)
    self.total_preemptions += 1

def _schedule_running(self, budget: int, max_tokens_per_seq=None):
    scheduled, preempted, i = [], 0, 0
    while i < len(self.running) and budget > 0:
        seq = self.running[i]
        n = min(seq.num_tokens - seq.num_computed_tokens, budget)
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
```

</details>

**常见错误**：忘记把 `num_computed_tokens` 清零。恢复时调度器以为前面的 token 还在 KV 里，只处理最后一个 token，而它的 `block_table` 已经是空的，结果是越界或输出错误。

到这里调度器已经完整了。

---

## 第 7 步：把一批请求展平成模型输入

这是整个项目里最需要耐心的一步。先在纸上把下面的例子算一遍，再写代码。

**目标**：把调度结果 `[(seq, n), ...]` 变成一次前向需要的全部张量。

**文件**：`engine.py` 中的 `_prepare_inputs(sched)`

### 7.1 为什么要展平

一批里有的请求本步处理 2 个 token，有的处理 1 个，有的处理 3 个。不做 padding，而是把所有新 token 首尾相接成一维：`input_ids` 的形状是 `[T]`，`T` 是本步 token 总数。线性层、归一化、MLP 都直接在 `[T, hidden]` 上计算；只有注意力需要知道每一行属于哪个请求、它的上下文在哪里。这就是 `AttentionMetadata` 的作用。

### 7.2 手算一个例子

块大小 $B = 4$。本步调度了三个请求：

| 请求 | `token_ids` | 已算 | 本步 n | `block_table` | 情况 |
|---|---|---|---|---|---|
| A | `[11, 12, 13, 14, 15, 16]`（prompt 6 个） | 4 | 2 | `[3, 7]` | 切块 prefill 的第 2 块 |
| B | `[21, 22, 23, 24, 25]`（prompt 3 个 + 已生成 2 个） | 4 | 1 | `[5, 2]` | decode |
| C | `[31, …, 37]`（prompt 7 个） | 0 | 3 | `[9]` | 新请求，只处理前 3 个 |

一个位置 $p$ 的 KV 存放在第 `block_table[p // B]` 块的第 `p % B` 格，展平后的槽位号是：

$$
\text{slot}(p) = \text{block\_table}[\lfloor p / B \rfloor] \times B + (p \bmod B)
$$

按顺序计算：

| 行 | 请求 | token | 位置 $p$ | 槽位 |
|---|---|---|---|---|
| 0 | A | 15 | 4 | `block_table[1]=7` → 7×4+0 = 28 |
| 1 | A | 16 | 5 | 7×4+1 = 29 |
| 2 | B | 25 | 4 | `block_table[1]=2` → 2×4+0 = 8 |
| 3 | C | 31 | 0 | `block_table[0]=9` → 36 |
| 4 | C | 32 | 1 | 37 |
| 5 | C | 33 | 2 | 38 |

于是：

```text
input_ids    = [15, 16, 25, 31, 32, 33]
positions    = [4, 5, 4, 0, 1, 2]
slot_mapping = [28, 29, 8, 36, 37, 38]
```

**注意力还需要每个请求的完整上下文**（本步之后，从位置 0 到 `已算 + n - 1`）：

- A：上下文长度 6，槽位 `[12, 13, 14, 15, 28, 29]`（块 3 的 4 格 + 块 7 的前 2 格）。A 本步有 2 个 token，走 prefill 组。
- C：上下文长度 3，槽位 `[36, 37, 38]`。走 prefill 组。
- B：本步只有 1 个 token 且不是第一个 token，走 decode 组。上下文长度 5，槽位 `[20, 21, 22, 23, 8]`。decode 组要把所有请求的槽位 padding 到同样长度拼成矩阵：B 有 2 块，矩阵为 `[[20, 21, 22, 23, 8, 9, 10, 11]]`，`mask` 为 `[[True]*5 + [False]*3]`，后 3 格是 padding。

**哪些请求要采样**：本步结束后 `已算 + n == num_tokens` 的请求，即所有已知 token 都进了 KV。A：4+2 = 6 = 6，要采样，取它的最后一行（第 1 行）。B：4+1 = 5 = 5，要采样，第 2 行。C：0+3 = 3 < 7，prompt 没算完，不采样。所以 `sample_rows = [1, 2]`。

测试 `test_step07` 用的正是这个例子，你可以把上面的每个数和测试里的断言一一对应。

### 7.3 约定

返回 `(input_ids, positions, meta, sample_seqs, sample_rows)`：

| 名称 | 类型 | 内容 |
|---|---|---|
| `input_ids`、`positions` | `[T]` 的 long 张量 | 见上例 |
| `meta.slot_mapping` | `[T]` 的 long 张量 | 每个新 token 的写入槽位 |
| `meta.prefill_q_starts`、`prefill_q_lens` | `list[int]` | prefill 组中每个请求在展平后的起始行与 token 数 |
| `meta.prefill_ctx_slots` | `list[Tensor]` | prefill 组中每个请求完整上下文的槽位 |
| `meta.decode_rows` | `[D]` 张量或 `None` | decode 组各请求所在的行 |
| `meta.decode_slots` | `[D, max_blocks × B]` 张量或 `None` | padding 后的槽位矩阵 |
| `meta.decode_mask` | `[D, max_blocks × B]` 布尔张量或 `None` | 有效位置为 `True` |
| `sample_seqs`、`sample_rows` | `list` | 要采样的请求，以及各自取 hidden 的行号 |

**归入哪一组**：`n == 1 and 已算 > 0` 归 decode 组，其他都归 prefill 组。没有 decode 请求时，三个 decode 字段都为 `None`。所有张量放在 `self.cfg.device` 上。

**验证**：`pytest tests/test_steps.py -k step07`

<details>
<summary>提示 1：思路</summary>

用一个变量 `row` 记录当前写到了展平后的第几行。对每个 `(seq, n)`：

1. `start = seq.num_computed_tokens`，`ctx = start + n`。
2. 把 `seq.token_ids[start:ctx]` 和 `range(start, ctx)` 分别追加到两个列表。
3. 对 `range(start, ctx)` 中每个位置算槽位，追加到 `slot_mapping` 列表。
4. 判断进 decode 组还是 prefill 组，记录对应信息。
5. 如果 `ctx == seq.num_tokens`，记下 `seq` 和行号 `row + n - 1`。
6. `row += n`。

循环结束后再把列表转成张量。上下文槽位可以用张量运算一次算出：`(block_table[:, None] * B + arange(B)).reshape(-1)[:ctx]`，它先列出每个块的全部槽位，再截取前 `ctx` 个。

</details>

<details>
<summary>提示 2：decode 组的 padding</summary>

先找出 decode 组中最长的 `block_table` 长度 `max_blocks`，把每个 `block_table` 用 0 补到这个长度（0 是一个合法块号，读出来的数据会被 mask 屏蔽，所以无所谓）。得到 `[D, max_blocks]` 的张量后，用和上下文槽位一样的方法展开成 `[D, max_blocks × B]`。`mask[i, j] = j < ctx_i`。

</details>

<details>
<summary>参考代码</summary>

```python
def _prepare_inputs(self, sched: SchedulerOutput):
    bs, dev = self.cfg.block_size, self.cfg.device
    ids, pos, slot_mapping = [], [], []
    pf_starts, pf_lens, pf_slots = [], [], []
    dec_rows, dec_tables, dec_ctx = [], [], []
    sample_seqs, sample_rows = [], []
    row = 0
    for seq, n in sched.scheduled:
        start = seq.num_computed_tokens
        ctx = start + n
        ids.extend(seq.token_ids[start:ctx])
        pos.extend(range(start, ctx))
        table = seq.block_table
        slot_mapping.extend(table[p // bs] * bs + p % bs for p in range(start, ctx))
        if n == 1 and start > 0:
            dec_rows.append(row)
            dec_tables.append(table)
            dec_ctx.append(ctx)
        else:
            pf_starts.append(row)
            pf_lens.append(n)
            bt = torch.tensor(table, device=dev)
            pf_slots.append((bt[:, None] * bs + torch.arange(bs, device=dev)).reshape(-1)[:ctx])
        if ctx == seq.num_tokens:
            sample_seqs.append(seq)
            sample_rows.append(row + n - 1)
        row += n

    dec_rows_t = dec_slots = dec_mask = None
    if dec_rows:
        max_blocks = max(len(t) for t in dec_tables)
        padded = torch.tensor([t + [0] * (max_blocks - len(t)) for t in dec_tables], device=dev)
        dec_slots = (padded[:, :, None] * bs + torch.arange(bs, device=dev)).reshape(len(dec_rows), -1)
        dec_mask = (torch.arange(dec_slots.shape[1], device=dev)[None, :]
                    < torch.tensor(dec_ctx, device=dev)[:, None])
        dec_rows_t = torch.tensor(dec_rows, device=dev)
    meta = AttentionMetadata(torch.tensor(slot_mapping, device=dev), pf_starts, pf_lens, pf_slots,
                             dec_rows_t, dec_slots, dec_mask)
    return (torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), meta, sample_seqs, sample_rows)
```

</details>

**常见错误**：采样行写成 `row`（该请求的第一行）而不是 `row + n - 1`（最后一行）；上下文只截到 `start` 而漏掉了本步的新 token。

---

## 第 8 步：分页注意力

**目标**：在 `Attention.forward` 中，把新 token 的 K、V 写进分页池，然后按第 7 步准备好的槽位读出上下文，算注意力。

**文件**：`model.py` 中 `Attention.forward` 的 TODO 部分。投影、QK 归一化、RoPE 都已经写好了，此时 `q` 的形状是 `[T, n_q, d]`，`k`、`v` 是 `[T, n_kv, d]`。

**先想清楚**

1. 为什么要先写入新 K、V，再读上下文？
2. A 是切块 prefill：本步 2 个新 token，上下文长度 6。掩码的形状是什么？第 0 行能看到哪几列？
3. decode 组为什么不需要因果掩码，只需要 padding 掩码？

<details>
<summary>答案</summary>

1. 新 token 也要关注自己和本步的其他新 token，它们的 K、V 必须先在池里。
2. 形状 `[2, 6]`。第 0 行是位置 4 的 token，能看到列 0~4；第 1 行是位置 5，能看到列 0~5。一般地，第 $i$ 行的位置是 `ctx - n + i`，允许的列是 `j <= ctx - n + i`。如果直接用 `torch.tril(ones(n, ctx))`，第 0 行只能看到第 0 列，前面 4 个历史 token 都被错误地挡住了。
3. decode 的那一个 token 就是当前最新的位置，上下文里的所有 token 都在它之前，全部可见；只需要挡住 padding。

</details>

**约定**：结果写入 `out`（形状 `[T, n_q, d]`）的对应行，然后由后面已经写好的 `o_proj` 投影。

| 部分 | 做法 |
|---|---|
| 写入 | `k_cache[meta.slot_mapping] = k`，`v` 同理 |
| prefill 组 | 对每个请求：取出 `q[start:start+n]`、`k_cache[slots]`、`v_cache[slots]`，转成 `[heads, len, d]`，构造上面第 2 题的掩码，调用 `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=True)`，结果写回 `out[start:start+n]` |
| decode 组 | `q[decode_rows]` 形状 `[D, n_q, d]` → `[D, n_q, 1, d]`；`k_cache[decode_slots]` 形状 `[D, L, n_kv, d]` → `[D, n_kv, L, d]`；掩码扩展为 `[D, 1, 1, L]`；一次调用 `scaled_dot_product_attention`；结果写回 `out[decode_rows]` |

`enable_gqa=True` 让 `n_kv` 个 KV 头自动对应到 `n_q` 个 Q 头，不需要手动 `repeat_interleave`。布尔掩码中 `True` 表示可以关注。

**验证**：`pytest tests/test_steps.py -k step08`。`step08a` 只测一次完整 prefill 的 logits，`step08b` 测切块 prefill 加 decode 的生成结果。

<details>
<summary>提示：按测试定位问题</summary>

- `step08a` 失败：问题在写入或 prefill 组。先确认 `out` 每一行都被写到了（可以用 `torch.full_like(q, float("nan"))` 代替 `empty_like`，没写到的行会是 NaN）。
- `step08a` 通过、`step08b` 第 1 个 token 就不同：切块时掩码的偏移错了。
- `step08b` 前几个 token 相同、后面不同：decode 组的问题，检查 `permute` 的维度顺序和掩码的形状。

</details>

<details>
<summary>参考代码</summary>

```python
        k_cache[meta.slot_mapping] = k
        v_cache[meta.slot_mapping] = v

        out = torch.empty_like(q)
        for start, n, slots in zip(meta.prefill_q_starts, meta.prefill_q_lens, meta.prefill_ctx_slots):
            ctx = slots.numel()
            qs = q[start:start + n].transpose(0, 1)                 # [n_q, n, d]
            ks = k_cache[slots].transpose(0, 1)                     # [n_kv, ctx, d]
            vs = v_cache[slots].transpose(0, 1)
            q_pos = torch.arange(ctx - n, ctx, device=x.device)
            mask = torch.arange(ctx, device=x.device)[None, :] <= q_pos[:, None]
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, scale=self.scale, enable_gqa=True)
            out[start:start + n] = o.transpose(0, 1)

        if meta.decode_rows is not None:
            qd = q[meta.decode_rows].unsqueeze(2)                   # [D, n_q, 1, d]
            kd = k_cache[meta.decode_slots].permute(0, 2, 1, 3)     # [D, n_kv, L, d]
            vd = v_cache[meta.decode_slots].permute(0, 2, 1, 3)
            mask = meta.decode_mask[:, None, None, :]               # [D, 1, 1, L]
            o = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask, scale=self.scale, enable_gqa=True)
            out[meta.decode_rows] = o.squeeze(2)
```

注意 starter 中 `out = torch.empty_like(q)` 后面有一行 `raise NotImplementedError`，写完后删掉它。

</details>

---

## 第 9 步：端到端正确性

不需要写新代码。运行：

```bash
pytest -q tests/test_correctness.py tests/test_scheduler.py
```

`test_correctness.py` 用五个长度不同的 prompt，分别在四种配置下（连续批处理 + 切块、频繁抢占、不切块、静态批处理）生成 20 个 token，要求与 Hugging Face 的贪心结果逐个相同，并且结束后所有 KV 块都已归还。

如果前面每一步的测试都通过了，这里通常也会通过。没通过时：

| 失败的用例 | 优先检查 |
|---|---|
| 只有 `test_with_preemption` 失败 | 第 6 步：抢占后的状态是否完全复位；恢复时是否重新分配了块 |
| 结束后 `num_free` 不等于总块数 | 块是否在某条路径上没有归还（抢占、完成） |
| 只有 `test_no_chunked_prefill` 失败 | 第 3 步 `allow_partial=False` 的分支 |

用追踪工具跑同样的配置，和参考实现逐步对比，通常能很快找到第一处不同：

```bash
python tools/trace.py --model /tmp/tiny --prompts 5,3,60,17,3 --budget 32 --blocks 40 --max-tokens 20 > mine.txt
python tools/trace.py --model /tmp/tiny --prompts 5,3,60,17,3 --budget 32 --blocks 40 --max-tokens 20 --impl solution > ref.txt
diff mine.txt ref.txt | head
```

---

## 第 10 步：后台引擎循环与 token 分发

**目标**：让同步的 `LLMEngine` 在后台线程里不断执行 `step()`，并把每个新 token 送到对应 HTTP 请求的 `asyncio.Queue`。

**文件**：`server.py` 中的 `AsyncEngine._dispatch`、`AsyncEngine._loop`。先读一遍已经写好的 `__init__`、`_handle`、`generate`，弄清楚 `inbox` 里放的是什么命令、`self.streams` 存的是什么。

**先想清楚**

1. 后台线程能不能直接调用 `q.put_nowait(item)`？
2. 没有任何请求时，后台线程应该做什么？
3. 什么样的 `StepOutput` 不需要投递？

<details>
<summary>答案</summary>

1. 不能。`asyncio.Queue` 只能在它所属事件循环的线程里操作。要用 `loop.call_soon_threadsafe(q.put_nowait, item)`，让事件循环线程去执行（1.4 节第 2.3 部分）。
2. 阻塞在 `self.inbox.get()` 上等待新命令，而不是空转反复调用 `step()`。
3. `new_token_id` 为 `None` 的（切块 prefill 中间步，没有采样）；以及 `self.streams` 中找不到的（请求已被中止）。

</details>

**约定**

`_dispatch(out)`：找到 `self.streams[out.seq.seq_id]` 中的 `(q, loop)`，投递 `(out.new_token_id, out.finished, out.seq.finish_reason)`；如果 `out.finished`，从 `self.streams` 中删除这个请求。

`_loop()`：无限循环。每一轮：

1. 如果引擎没有未完成的请求，阻塞等待 `inbox` 的第一条命令；否则不阻塞。
2. 把 `inbox` 中当前所有命令都取出来，逐条交给 `self._handle(cmd)`。
3. 执行一次 `self.engine.step()`，把每个输出交给 `self._dispatch`。

**验证**：`pytest tests/test_steps.py -k step10`（只测 `_dispatch`），然后 `pytest tests/test_server.py`。

<details>
<summary>提示：取空队列的写法</summary>

`queue.Queue.get(block=False)` 在队列为空时抛出 `queue.Empty`。可以这样写"第一条按需阻塞，之后不阻塞，取到空为止"：

```python
block = not self.engine.has_unfinished()
try:
    while True:
        cmd = self.inbox.get(block=block)
        block = False
        self._handle(cmd)
except queue.Empty:
    pass
```

</details>

<details>
<summary>参考代码</summary>

```python
def _loop(self):
    while True:
        block = not self.engine.has_unfinished()
        try:
            while True:
                cmd = self.inbox.get(block=block)
                block = False
                self._handle(cmd)
        except queue.Empty:
            pass
        for out in self.engine.step():
            self._dispatch(out)

def _dispatch(self, out: StepOutput):
    entry = self.streams.get(out.seq.seq_id)
    if entry is None or out.new_token_id is None:
        return
    q, loop = entry
    loop.call_soon_threadsafe(q.put_nowait, (out.new_token_id, out.finished, out.seq.finish_reason))
    if out.finished:
        self.streams.pop(out.seq.seq_id, None)
```

</details>

**常见错误**：`_loop` 里抛出的异常只会结束后台线程，HTTP 请求则会一直等待。`test_server.py` 设置了 60 秒超时；手动调试时，在 `_loop` 外包一层 `try/except` 并打印 traceback 会更容易发现问题。

全部测试通过：

```bash
pytest -q tests
```

---

## 第 11 步：在真实模型上跑起来

### 11.1 先在 CPU 上跑通服务

```bash
PYTHONPATH=starter python -m miniserve.server --model /tmp/tiny --device cpu --num-gpu-blocks 1024 --port 8000
```

另开一个终端：

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"hello","max_tokens":8,"ignore_eos":true}'
python bench.py --base-url http://localhost:8000 --num-prompts 32 --input-len 64 --output-len 32 \
  --max-concurrency 8 --vocab-size 400
curl -s localhost:8000/metrics
```

微型模型的权重是随机的，生成的文字没有意义，这是正常的。

### 11.2 换成 GPU 和真实模型

```bash
# 下载模型到本地目录（旧版 huggingface_hub 用 huggingface-cli download）
hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B

PYTHONPATH=starter python -m miniserve.server --model ~/models/Qwen3-0.6B --port 8000 --max-model-len 4096
```

启动日志中会打印显存测算结果和 KV 块数，记下来，项目页的实验 D 会用到。用聊天接口确认输出是正常的中文或英文：

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话解释 KV Cache"}],"max_tokens":64}'
```

### 11.3 第一组对比

用同一组参数分别压测 HF 基线和 MiniServe：

```bash
python baseline_hf.py --model ~/models/Qwen3-0.6B --port 8001     # 另开一个终端启动基线
python bench.py --base-url http://localhost:8001 --num-prompts 64 --input-len 512 --output-len 128 --max-concurrency 16 --vocab-size 32000
python bench.py --base-url http://localhost:8000 --num-prompts 64 --input-len 512 --output-len 128 --max-concurrency 16 --vocab-size 32000
```

输出吞吐的比值就是项目完成标准中的"相对 HF 基线的倍数"。接下来回到 [项目 1 页面](stage1/boss-miniserve.md) 完成四组实验和报告。

---

## 卡住时

1. **先缩小范围**：用 `pytest tests/test_steps.py -k stepNN -x` 只看第一个失败的断言。断言信息会说明期望值和实际值。
2. **再看行为**：用 `tools/trace.py` 对比自己的实现和参考实现，找到第一处不同。
3. **最后看代码**：按顺序打开当前步骤的三级提示。看过参考代码后，关掉它自己重写一遍，确认真的理解了。
