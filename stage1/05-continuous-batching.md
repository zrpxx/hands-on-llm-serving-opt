# 1.5 Continuous Batching 与 Chunked Prefill

## 学习目标

完成本节学习后，你将能够：
- **从静态批处理、动态批处理一路推演到连续批处理（Continuous Batching）**：用“渡船过河”的直觉讲清楚为什么传统批处理在大模型下会造成严重的 GPU 空转。
- **区分并联合调优两个最核心的调度参数**：`--max-num-seqs`（最大并发请求数）与 `--max-num-batched-tokens`（单步最大 Token 预算）。
- **彻底理解 Chunked Prefill（分块预填充）的设计权衡**：明白它是如何把长 Prompt 切碎填入 Decode 步的算力空隙里，从而消灭 ITL 卡顿尖峰的。

---

## 1. 传统批处理为什么不适合 LLM

在 `0.4 Roofline` 节我们已经算过一笔铁账：单请求 Decode 的算术强度只有 $I \approx 1$，GPU 99% 的算力都在空转；只有把 $b$ 个用户的请求合批（Batching）在一起算，让所有人共用同一次权重读取，算术强度才能提升到 $I \approx b$，吞吐才能翻几十倍。

但在传统图像分类（如 ResNet）或推荐模型里做批处理很容易——因为每张图片都是固定的 `224x224` 大小，跑 1 次前向传播统统只要 `10 ms`，同一批进去的 10 张图一定在同一毫秒一起出来。

而到了生成式大模型里，**同一批进来的 4 个用户，生成的回答长度可能天差地别**：
- 用户 1 问：`"今天星期几？"` $\to$ 生成 **5 个词** 就吐出 `EOS` 结束了（耗时 0.1 秒）。
- 用户 2 问：`"写首绝句"` $\to$ 生成 **28 个词** 结束（耗时 0.5 秒）。
- 用户 3 问：`"请帮我写一篇 2000 字的年度述职报告"` $\to$ 要生成 **2000 个词**（耗时 30 秒）。

如果你用传统的批处理方式：**用户 1 和用户 2 虽然不到半秒钟就算完了，但它们的 GPU 槽位必须原地干等用户 3 的 2000 个字全部写完（足足等 29.5 秒。），才能把整批释放掉、去接门外排队的第 4 个用户。**

---

## 2. 三代批处理：静态、动态、连续

我们用码头开渡船的比喻，把三代批处理技术对比得明明白白：

```mermaid
flowchart TB
  subgraph S1["第一代：静态批处理 (Static Batching)"]
    A1["规则：船必须凑满 10 个人才开船；且必须等船上走得最慢的人下船，才接下一船"]
  end
  subgraph S2["第二代：动态批处理 (Dynamic Batching，如传统 Triton Server)"]
    A2["改进开船条件：凑满 10 人【或】最多等 50ms 就开船；但开船后依然要整船同进同出。"]
  end
  subgraph S3["第三代：连续批处理 (Continuous Batching / Iteration-level Scheduling)"]
    A3["革命性改变：不再按'整船完成'调度，而是按'每迈一步(1个Token)'调度。<br/>每走一步，到站的人立刻下船释放座位，岸上排队的人立刻跳上空座位。"]
  end
  S1 --> S2 --> S3
```

| 批处理代际 | 调度粒度 | 解决了什么问题？ | 还有什么致命缺陷？ |
|---|---|---|---|
| **① 静态批处理** | 整批请求（Batch-level） | 离线跑固定数据集时能跑满 GPU | 在线流量低时，前 9 个人要无限期干等第 10 个人到来才能开跑 |
| **② 动态批处理** | 整批请求 + 超时窗口（如 `max_wait=10ms`） | 解决了“开船前无限干等”的问题 | **开船后依然要整批绑定到底**。短请求算完后槽位一直空转（形成巨大的红色空泡），新请求进不来 |
| **③ 连续批处理（Orca 提出）** | **单步迭代（Iteration-level，即每生成 1 个词）** | **彻底消灭尾部空泡，也不再需要 `max_wait` 等待窗口。** 短请求第 5 步一吐出 EOS，第 6 步新请求立刻顶替它的位置 | 当新加入的请求需要做超长 Prefill 时，会把船上正在 Decode 的老乘客全部卡停（需要下面的 **Chunked Prefill** 来救场） |

---

## 3. 模拟：三种策略下同一组请求的时间线

下面的模拟器用与项目 1 MiniServe 相同的调度规则，逐步模拟 5 个请求（最多同时运行 3 个）。代价模型很简单：每一步固定耗时 1 个单位（每步都要读一遍权重），每个 prefill token 再加 1/8 个单位（prefill 受算力限制，token 越多越慢）。右侧两列是每个请求的 TTFT 和最大 ITL。

<div class="llm-widget" data-widget="batching-timeline-sim"></div>

### 按顺序观察

1. **静态批处理**：R1 很快结束，但 R4、R5 要等前一批三个请求全部完成才能开始，TTFT 分别达到 19.1 和 16.1。这就是第 2 节说的"整船绑定"。
2. **连续批处理，不切块**：R4 一到就被接纳，TTFT 降到 8.4。但 R4 的 32 个 prompt token 在一步内算完，这一步耗时 5 个单位，正在 decode 的 R2、R3 在这一步完全停顿（红框），最大 ITL 从 1.0 升到 6.0。
3. **连续批处理 + 切块**：R4 的 prompt 被切成每步 12 个 token 左右，与 R2、R3 的 decode 同步进行，最大 ITL 降到 2.3；代价是 R4 的 TTFT 升到 11.4。
4. 把**每步 token 预算**从 4 拖到 32，观察最大 ITL 与平均 TTFT 的此消彼长；再把 **R4 的 prompt 长度**调到 64，不切块时停顿会更明显。

这个取舍正是第 4 节要讨论的：token 预算越小，decode 越平稳，但长 prompt 的首 token 越晚。

---

## 4. 两个预算与 Chunked Prefill

在现代推理引擎（如 vLLM、SGLang 以及我们的项目 1 MiniServe）中，连续批处理调度器靠**两把尺子**来决定每一步（Step）到底放多少工作量上卡：

### 4.1 两把控制尺子：`max_num_seqs` vs `max_num_batched_tokens`

为什么不能只设一个“最大 Batch Size”？
因为 **10 个正在 Decode 的请求（一共只送入 10 个 Token）** 和 **10 个刚进来要做 8K 长文 Prefill 的请求（一共送入 80,000 个 Token）**，对 GPU 算力和激活显存的压力相差 **8,000 倍**。因此调度器必须同时卡住两个上限：

| 参数名称（vLLM / MiniServe） | 物理含义 | 主要卡住哪个阶段？ | 设太小会怎样？设太大会怎样？ |
|---|---|---|---|
| **`--max-num-seqs`** | 同时处于运行状态（RUNNING）的**最大请求个数** | **Decode 阶段**（每个请求每步只出 1 个 Token，并发请求数直接决定 Decode 批大小 $b$） | 设太小：Decode 算术强度低，吞吐上不去；<br/>设太大：KV Cache 块不够分，频繁触发抢占（Preemption） |
| **`--max-num-batched-tokens`** | 一步 `engine.step()` 前向传播中，所有请求加起来的**最大 Token 总数预算** | **Prefill 阶段 & Chunked Prefill 切块大小** | 设太小（如 `< 256`）：每步喂给 GPU 的 Token 太少，Prefill 算力吃不满，TTFT 变慢；<br/>设太大（如 `> 16384`）：单步耗时过长，正在 Decode 的请求被阻塞，ITL P99 尖峰飙升 |

### 4.2 Chunked Prefill（分块预填充）到底是怎么算的？

假设你把单步 Token 预算设为 `max_num_batched_tokens = 512`。
现在系统里有 **12 个正在 Decode 的老请求**，同时门外新来了 **1 个长度为 1,500 Token 的长 Prompt 请求 R_new**。调度器在这一步是怎么分配预算的？

```mermaid
flowchart LR
  B["本步总预算<br/>token_budget = 512"] --> D["① 先喂饱老请求 Decode<br/>12 个老请求 × 1 token = 12 tokens"]
  D --> R["② 剩余预算 = 512 - 12 = 500 tokens<br/>把 R_new 的前 500 个词切出来作为第 1 块。"]
  R --> STEP["③ 合并成 [512, hidden] 张量<br/>做 1 次 GPU 前向传播。"]
```

1. **第 1 步**：12 个老请求各算 1 个 Decode 词（消耗 12 预算），`R_new` 算它的第 `0 ~ 499` 个词并写入 KV Cache（消耗 500 预算），刚好凑满 512。这一步只花约 `15 ms`，12 个老请求丝滑拿到了新词，完全没感觉到卡顿。
2. **第 2 步**：12 个老请求继续各算 1 个词，`R_new` 接着算第 `500 ~ 999` 个词（它的 Q 会去和前 500 个词的 KV Cache 加上这 500 个词的 KV 算因果注意力）。
3. **第 3 步**：12 个老请求各算 1 个词，`R_new` 算完最后的第 `1000 ~ 1499` 个词，并终于吐出它的第 1 个生成词，正式转为 Decode 状态。

**这就是 Chunked Prefill（由 Sarathi-Serve 论文系统提出）的一石二鸟之计**：
- **对老请求（Decode）**：再也不会被别人的万字长文一把卡死几百毫秒，**ITL P99 毛刺被彻底削平**。
- **对 GPU 硬件**：原本 12 个请求的 Decode 算术强度只有 $12$（严重访存受限，算力空置 90%）；现在塞进了 500 个 Prefill Token 一起过线性层，**把原本白白浪费的空闲 Tensor Core 算力顺手白嫖利用了起来**，总吞吐反而更高。
- **唯一的微小代价**：`R_new` 自己的 TTFT 略微增加了一点点（因为分成了 3 步才算完 Prompt）。

---

## 5. 动手实验：用实测数据验证四种调度策略的差距

在项目 1 MiniServe 中，我们已经内置了这几种调度模式。即使在纯 CPU 上跑微型模型（64 个长短混合请求，泊松到达 30 req/s，SLO 要求 `TTFT ≤ 500ms, TPOT ≤ 30ms`），也能测出教科书般清晰的对比数据：

| 调度策略 | 输出吞吐 (tok/s) | TTFT P50 / P99 (ms) | 单步间隔 ITL P99 / Max (ms) | 有效吞吐 Goodput (req/s) | 核心现象解读 |
|---|---|---|---|---|---|
| **① 静态批处理** | 1,185 | 759 / 1,692 | 7.3 / 9.2 | **5.3** | 新请求在门外排长队等上一批结束，TTFT 爆炸，Goodput 垫底 |
| **② 连续批处理（不切块）** | 1,418 | 264 / 742 | 27.4 / **62.7** | 15.6 | 排队大幅缓解，Goodput 翻 3 倍；但长 Prefill 造成了 **62.7 ms** 的严重卡顿尖峰 |
| **③ 连续批处理 + 切块（预算 2048）** | **1,481** | **150 / 644** | 22.7 / **24.1** | **16.9** | **甜点配置**：卡顿尖峰从 62.7ms 削平到 24.1ms，且吞吐与 Goodput 均达到最高 |
| **④ 连续批处理 + 切块（预算过小 128）** | 1,384 | 242 / 807 | **15.0** / 27.9 | 13.6 | 切得太碎（每步最多 128 词），ITL 最稳，但长 Prompt 要切十几次才算完，拖慢了 TTFT |

如果你有 GPU，可以用 vLLM 调整 `--max-num-batched-tokens`（分别设为 `512`、`2048`、`8192`）跑一遍同样的长短混合压测，亲手找出你手里显卡的黄金预算值：

```bash
vllm serve Qwen/Qwen3-8B --max-model-len 16384 --max-num-batched-tokens 2048

python code/project1/bench.py --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
  --num-prompts 300 --input-len 4096 --output-len 256 --range-ratio 0.9 \
  --request-rate 4 --vocab-size 32000
```

---

## 6. 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "静态批处理（以及传统的动态批处理）为什么通常要配置一个凑批等待时间，而连续批处理不需要？", "options": ["连续批处理使用了更大的批，等待时间可以省略", "静态批处理一旦开始要等整批结束才能接新请求，开批前多等几毫秒能多装几个；连续批处理每一步都能接纳新请求，不必等待", "等待时间是为了让 GPU 预热，连续批处理使用 CUDA Graph 所以不需要预热", "连续批处理把等待时间转移到了 tokenizer 线程"], "answer": 1, "explain": "静态批处理错过一次开批就要等一整批的生成时间，所以值得多等一会儿凑满。连续批处理以迭代为调度粒度，新请求最多等一步，凑批的收益消失了。"}, {"q": "某服务的请求输入很短（平均 50 token）、输出很长（平均 1500 token），并发很高。哪个限制最先成为瓶颈？", "options": ["<code>max_num_batched_tokens</code>，因为每一步的 prefill token 太多", "<code>max_num_seqs</code> 或 KV Cache 容量：几乎所有请求都在 decode，每步每个序列只贡献 1 个 token，token 预算用不满", "tokenizer 的吞吐，因为输出 token 太多", "HTTP 连接数，因为长输出会让连接长时间保持"], "answer": 1, "explain": "decode 为主的负载里，每步的 token 数约等于批中的序列数，远小于 token 预算。限制并发的是序列数上限，以及每个序列不断增长的 KV 占用。输入长、输出短的负载才会先碰到 token 预算。"}, {"q": "开启 chunked prefill 后，把 <code>max_num_batched_tokens</code> 从 2048 调小到 64，会产生什么效果？", "options": ["TTFT 和 ITL 都会改善，没有代价", "ITL 更平稳，但长 prompt 要切成很多块，TTFT 明显变长，每步的固定开销占比升高，总吞吐下降", "ITL 变差，因为每步都要处理 prefill", "没有影响，token 预算只在不开 chunked prefill 时生效"], "answer": 1, "explain": "token 预算决定每步最多夹带多少 prefill token。预算越小，decode 被打断得越轻，但 4000 token 的 prompt 需要 60 多步才能算完，而且每步的调度、启动开销都摊不开。可以在 1.5 的模拟器中拖动预算滑块观察这个取舍。"}]'></div>

---

## 延伸阅读

- Gyeong-In Yu et al., [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022（首次提出 Iteration-level Scheduling / Continuous Batching）
- Amey Agrawal et al., [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310), OSDI 2024（系统提出 Chunked Prefill 消除 Prefill-Decode 干扰）
- Chi Wang & Peiheng Hu, *Hands-On LLM Serving and Optimization*, O'Reilly, Chapter 6 (*Request Batching and Scheduling-Level Optimizations*)
