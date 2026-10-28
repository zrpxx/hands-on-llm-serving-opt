# 1.6 vLLM 架构与调度器

## 学习目标

完成本节学习后，你将能够：
- **说清楚传统连续 KV Cache 预分配造成的三种显存浪费**（预留浪费、内部碎片、外部碎片），明白为什么早期系统有超过 60% 的显存被白白浪费。
- **用操作系统“虚拟内存分页”的思想手绘出 PagedAttention 的映射过程**：掌握逻辑块、`block_table`、物理块池与槽位映射（`slot_mapping`）的精确关系。
- **读懂 vLLM V1 引擎的极简统一调度公式**：理解 `num_new_tokens = num_tokens - num_computed_tokens` 如何用一行减法统一处理 Prefill、Chunked Prefill、Decode、前缀缓存与抢占恢复。

---

## 1. PagedAttention 之前的显存浪费

在 `1.3` 节我们学会了用 KV Cache 给自回归生成加速，但在早期推理系统中，要把一个请求的 KV Cache 存进显存，面临一个极其棘手的难题：
- 底层深度学习框架（如 PyTorch）要求一个张量在物理显存上必须是**一块连续的内存地址**。
- 可是当一个用户发来请求时，**没人知道它最后会生成 10 个词还是 4096 个词**。

于是，早期系统只能采取最笨的办法：**只要一个请求进来，不管三七二十一，先在显存里按最大可能长度（如 `max_len = 2048`）给它圈一块连续的领地。**

这种做法立刻导致了三种触目惊心的显存浪费（vLLM 论文实测表明，当时系统中**真正存了有效数据的 KV 显存只有 20.4%–38.2%**，其余超过 60% 全在装空气。）：

```mermaid
flowchart TB
  subgraph Bad["传统连续预分配：三大显存浪费（利用率 < 35%）"]
    direction LR
    R1["请求 A (最大预留 2048 格)<br/>[已算 100 词] + [预留等待区 400 格] + [内部碎片：提前 EOS 永远用不上的 1548 格]"]
    GAP["外部碎片：A 和 B 之间夹着 600 格空隙<br/>放不下要求连续 1024 格的新请求 C。"]
    R2["请求 B (最大预留 1024 格)"]
  end
```

| 浪费类型 | 它是怎么产生的？ | 形象比喻 |
|---|---|---|
| **① 预留浪费（Reservation Waste）** | 请求未来可能会用到，但当前这一步还没生成到那儿，空间却已经提前霸占住了 | 客人说“我最多可能吃 20 盘菜”，餐厅一进门就先给他摆满 20 个空盘子占住桌子 |
| **② 内部碎片（Internal Fragmentation）** | 按最大长度 2048 预留了空间，结果模型生成了 50 个词就吐出 `EOS` 结束了，剩下 1998 个格子的预留彻底浪费 | 客人吃饱走了，那 19 个从来没盛过菜的盘子白占了整场 |
| **③ 外部碎片（External Fragmentation）** | 不同长度的连续内存块频繁分配和释放后，显存里到处是东一块 300 MB、西一块 500 MB 的零碎空隙，拼不出新请求要求的连续 1 GB | 停车场里到处是半个车位的空隙，加起来能停 5 辆车，但偏偏塞不进任何 1 辆完整的车 |

---

## 2. PagedAttention：借鉴操作系统的分页

怎么在不事先知道长度的情况下，彻底消灭上面三种浪费？
vLLM 的作者向诞生了半个世纪的**操作系统虚拟内存分页（Virtual Memory Paging）**借了一招：

> **核心思想**：不再要求一个请求的 KV Cache 在物理显存里连续存放。而是把整个 GPU 的 KV 显存池切成一格一格大小完全相同的**物理块（Physical Blocks，比如每块固定存 $B = 16$ 个 Token）**。请求每生成几个新词，需要用多少块，就当场从空闲池里领一块，用一张**块表（`block_table`）**记住映射关系即可。

```mermaid
flowchart LR
  subgraph Logical["请求 A 眼里的逻辑 KV（看起来是连续增长的）"]
    L0["逻辑块 0 (词 0~3)<br/>'Alan', 'Turing', 'was', 'a'"]
    L1["逻辑块 1 (词 4~7)<br/>'British', 'mathematician', 'who', 'is'"]
    L2["逻辑块 2 (词 8~9，未满)<br/>'famous', 'for', [空], [空]"]
  end
  subgraph BT["请求 A 的 Block Table（登记簿）"]
    T0["逻辑块 0 物理块 #7"]
    T1["逻辑块 1 物理块 #1"]
    T2["逻辑块 2 物理块 #3"]
  end
  subgraph Phys["GPU 物理 KV 块池（物理上东一块西一块，完全不连续。）"]
    P1["物理块 #1 (已满)"]
    P3["物理块 #3 (用 2/4)"]
    P7["物理块 #7 (已满)"]
  end
  L0 --> T0 --> P7
  L1 --> T1 --> P1
  L2 --> T2 --> P3
```

看看这一下解决了多少问题：
1. **零预留浪费**：按需领块，写满第 0 块（第 0~3 个词）后，第 4 个词生成时才去领第 1 块。
2. **内部碎片最多不到 1 个块**：就算请求随时结束，最多也只浪费**最后那 1 个物理块里没填满的几个槽位**（比如块大小为 16 时，平均只浪费 7.5 个 Token，占几千长文的不到 0.2%。）。
3. **零外部碎片**：因为所有物理块大小一模一样（都是 $B=16$），任何释放出来的物理块都能立刻分给任何其他请求使用，永远不会出现“空隙塞不进”的情况。
4. **顺手白送“跨请求共享”能力**：如果两个请求有相同的开头（如公共 System Prompt），只需要让它们各自 `block_table` 的第 0 项同时指向同一个物理块 `#7` 即可。这就是下一节（`1.7 Prefix Caching`）的地基。

---

## 3. 组件：块分配与抢占

点击下方模拟器的三个状态标签，观察多个请求如何从同一个物理块池里按需领块，以及当所有物理块被领光时，调度器是如何通过**抢占（Preemption）**自救的：

<div class="llm-widget" data-widget="paged-attention-sim"></div>

### 动手试一试 & 算清槽位寻址公式（项目 1 必考。）
在上面的组件中，假设每块大小 `block_size` $B = 4$，请求 A 的 `block_table = [1, 4, 7]`。
请问：当请求 A 生成到它的第 **6** 个词（从 0 开始数，即位置 `pos = 5`）时，这个词的 $K, V$ 应该写到物理池的第几个具体槽位（`slot`）里？
- **第 1 步（算逻辑块号）**：`logical_block = pos // B = 5 // 4 = 1`（属于逻辑第 1 块）。
- **第 2 步（查表得物理块号）**：`physical_block = block_table[1] = 4`（对应物理块 `#4`）。
- **第 3 步（算块内偏移）**：`offset = pos % B = 5 % 4 = 1`（物理块 `#4` 里的第 1 个格子）。
- **合并成一行公式**：
  $$\text{slot} = \text{block\_table}[\text{pos} \mathbin{//} B] \times B + (\text{pos} \bmod B) = 4 \times 4 + 1 = \mathbf{17}$$
这行公式就是你在 **项目 1 MiniServe 的 `_prepare_inputs()`** 里要亲手写出来的核心寻址代码。

---

## 4. vLLM V1 的架构与调度循环

理解了 PagedAttention，我们再来看看工业界标杆 **vLLM（以当前默认的 V1 架构为准）** 是如何把各个模块组织起来的。

### 4.1 vLLM 的核心组件与启动时的“显存实测（Profiling）”

```mermaid
flowchart LR
  U["客户端请求"] --> P["Processor<br/>(输入校验与 CPU 分词)"]
  P --> EC["EngineCore (独立进程)"]
  subgraph Core["EngineCore 核心循环"]
    SCH["Scheduler 调度器<br/>(管队列 · 分 Token 预算 · 管 Block Table)"]
    EX["ModelExecutor / GPUWorker<br/>(管多卡通信 · CUDA Graph · 跑 PagedAttention)"]
    SCH <-->|"SchedulerOutput"| EX
  end
  EC --> OP["Output Processor<br/>(增量反分词与流式返回)"]
```

在 `0.3` 节你做实验时可能注意到了：vLLM 启动时并不是用公式去“猜”激活值占多少显存，而是做了一次真实的 **显存测算（Memory Profiling）**：
1. Worker 进程先把模型权重加载进 GPU。
2. 构造一个达到 `max_num_batched_tokens` 上限的假输入（Dummy Batch），**真刀真枪跑一次前向传播**，实测出峰值中间激活张量吃掉了多少显存。
3. 把 `总显存 × gpu_memory_utilization - 权重显存 - 实测激活峰值` 算出来的剩余空间，除以单块大小，**一次性全部切分成 `num_gpu_blocks` 个物理 KV 块**交由 `BlockPool` 管理。（MiniServe 的 `_profile_num_blocks` 同样复刻了这一经典机制。）

> [!NOTE]
> **教材版本勘误提醒**：原书第 8 章在介绍模型加载时提到 `Qwen2.5-7B` 会匹配到 `Qwen3NextForCausalLM`，且示例代码里写了 `Qwen/Qwen3-7B-Instruct`。实际上 `Qwen2.5` 的 `config.json` 中 `architectures` 为 `Qwen2ForCausalLM`；而官方 Qwen3 稠密模型家族的尺寸规格为 `0.6B / 1.7B / 4B / 8B / 14B / 32B`（并无 7B 版本）。阅读源码或跑实验时以模型仓库实际的 `config.json` 为准。

---

### 4.2 调度器的灵魂：用一行减法统一所有阶段。

很多初学者以为调度器里要写无数个 `if is_prefill: ... elif is_chunked: ... elif is_decode: ...`。
翻开 vLLM V1 调度器源码（以及我们的 MiniServe `scheduler.py`），你会惊讶地发现：**调度器根本不区分什么 Prefill 还是 Decode。它只为每个请求维护两个整数，并用一行减法统管全局**：

- **`seq.num_tokens`**：这个请求到目前为止**已知的全部 Token 总数**（= 初始 Prompt 词数 + 已经生成出来的词数）。
- **`seq.num_computed_tokens`**：这个请求到目前为止**已经算完并存进 KV Cache 里的 Token 个数**。

那么，在当前这一步，这个请求**还需要算多少个新 Token**？就是两者的差值（再和当前剩余的单步预算 `token_budget` 取个最小值）：

$$
\text{num\_new\_tokens} = \min\!\Big(\text{token\_budget},\ \text{seq.num\_tokens} - \text{seq.num\_computed\_tokens}\Big)
$$

让我们看看这一个减法式子是如何神奇地统一五种场景的：

| 场景 | `num_tokens` | `num_computed_tokens` | 差值（本步要算的词数） | 算完本步后发生了什么？ |
|---|---|---|---|---|
| **① 全新短请求到达（Prompt=100 词，预算=512）** | $100$ | $0$ | $\min(512, 100 - 0) = \mathbf{100}$ | 一次算完 100 词，`computed` 变为 $100$，并采样出第 1 个新词（`num_tokens` 变为 $101$）。 |
| **② 超长请求切块 Prefill（Prompt=1000 词，预算=512）** | $1000$ | $0$ | $\min(512, 1000 - 0) = \mathbf{512}$ | 本步只算前 512 词，`computed` 变为 $512$（还没追上 $1000$，所以本步不采样，留着下步继续算剩下的 $488$ 词）。 |
| **③ 正常 Decode 步（昨天已算完 100 词并采样出第 101 词）** | $101$ | $100$ | $\min(512, 101 - 100) = \mathbf{1}$ | 自然而然算出差值为 **1**。本步只算最新那 1 个词。 |
| **④ 前缀缓存命中 80 词的新请求（Prompt=100 词）** | $100$ | 直接设为 $\mathbf{80}$ | $\min(512, 100 - 80) = \mathbf{20}$ | 自动跳过已命中的前 80 词，只算末尾新增的 20 词。 |
| **⑤ 被抢占（Preempt）后重新恢复的请求** | $150$ | 被清零为 $\mathbf{0}$ | $\min(512, 150 - 0) = \mathbf{150}$ | 把原来的 Prompt 连同已生成的词一起当成 150 词重新做一次 Prefill。 |

### 4.3 调度每一步的固定顺序：为什么必须“先保老员工（RUNNING），再招新员工（WAITING）”？

在每一步 `schedule()` 中，调度器严格遵守两阶段顺序：
1. **第一阶段：遍历 `running` 队列（已经在跑的老请求）**：
   - 它们已经占着物理 KV 块，屏幕前的用户正在盯着它吐字。
   - 检查每个老请求本步新增 Token 后需不需要新领物理块；如果空闲块池空了，就从 `running` **队尾（最晚进来的最低优先级请求）触发抢占（`_preempt`）**：释放队尾请求的全部物理块、把它的 `num_computed_tokens` 清零塞回 `waiting` 队头，用腾出来的块保住排在前面的老请求继续生成。
2. **第二阶段：如果刚才没发生抢占，且预算和席位还有富余，再从 `waiting` 队列接纳新请求**。

---

## 5. 实验：阅读调度器源码并复现抢占

### 实验 1：对照阅读 MiniServe 的百行调度器
打开本仓库的 [`code/project1/solution/miniserve/scheduler.py`](https://github.com/zrpxx/hands-on-llm-serving-opt/blob/main/code/project1/solution/miniserve/scheduler.py)（总共只有 160 行，结构与 vLLM V1 的 `vllm/v1/core/sched/scheduler.py` 完全同构），亲眼看看 `_schedule_running`、`_preempt` 和 `_admit` 是怎么写的。

### 实验 2：在 GPU 上故意制造显存不足，观察抢占（Preemption）监控指标
启动 vLLM 时故意把 `--gpu-memory-utilization` 压低（只给很少的 KV 块），同时发一批长输出请求，在另一个终端实时抓取 `/metrics`：

```bash
curl -s http://localhost:8000/metrics | grep -E "num_preemptions|gpu_cache_usage|num_requests_(running|waiting)"
```

当你看到 `num_preemptions_total` 开始上涨时，立刻观察压测客户端报告的 **TPOT P99** 是否同步出现了明显的跳升。

---

## 6. 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "把 PagedAttention 的 block_size 设得很小（例如 1）或很大（例如 512），分别有什么问题？", "options": ["两者都没有问题，block_size 只影响显存统计的粒度", "很小时 block_table 很长，注意力 kernel 寻址次数多、访存不连续；很大时每个序列最后一块平均浪费约一半，碎片又回来了，前缀缓存的命中粒度也变粗", "很小时会浪费显存，很大时会降低 KV 的数值精度", "很小时无法使用 GQA，很大时无法使用 chunked prefill"], "answer": 1, "explain": "块大小是寻址开销与内部碎片之间的折中。16 是常见的默认值，FlashInfer 等后端在部分场景用更大的块。块大小与数值精度、GQA 都无关。"}, {"q": "某请求的 prompt 有 600 个 token，单步 token 预算为 256，开启 chunked prefill。第一步调度结束后，这个序列的状态是什么？", "options": ["<code>num_computed_tokens = 600</code>，并采样出第一个输出 token", "<code>num_computed_tokens = 256</code>，这一步不采样，下一步继续算后面的 prompt", "<code>num_computed_tokens = 256</code>，并用第 256 个位置的 logits 采样一个 token", "<code>num_computed_tokens = 0</code>，因为 prompt 超过预算，要等到预算足够时一次算完"], "answer": 1, "explain": "只有当一个序列的已计算长度追上全部 prompt 时，最后一个位置的 logits 才是“下一个 token”的分布。中途的块不采样。按 256、256、88 分三步，第三步结束时才产生第一个输出 token。"}, {"q": "调度器每一步为什么先给 RUNNING 队列分配预算和块，最后才用剩余资源接纳 WAITING 中的新请求？", "options": ["WAITING 队列中的请求还没有经过 tokenizer，无法立即调度", "运行中的请求已经占着 KV 块、用户正在接收输出；先接纳新请求可能导致块不够，只能抢占老请求，已做的计算被浪费，输出也会停顿", "这样可以保证每一步的 token 数恰好等于预算", "新请求的 prefill 必须在 decode 之后执行，否则注意力掩码会出错"], "answer": 1, "explain": "优先保证已有请求能继续前进，是减少抢占和保持 ITL 稳定的关键。新请求只用剩下的预算与空闲块。prefill 与 decode 混合在同一批里时顺序与掩码正确性无关，元数据会区分它们。"}]'></div>

---

## 延伸阅读

- Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023
- Aleksa Gordić, [*Inside vLLM: Anatomy of a High-Throughput LLM Inference System*](https://www.aleksagordic.com/blog/vllm)
- Chi Wang & Peiheng Hu, *Hands-On LLM Serving and Optimization*, O'Reilly, Chapter 6 (*PagedAttention*) & Chapter 8 (*vLLM*)
