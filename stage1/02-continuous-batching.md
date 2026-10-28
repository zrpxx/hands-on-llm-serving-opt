# 1.2 连续批处理与 Chunked Prefill

> 对应原书第 6 章 *Request Batching and Scheduling-Level Optimizations*。

0.7 节说明了批处理为什么对 decode 至关重要：权重读一遍，可以同时算 $b$ 个请求的 token，吞吐几乎随批大小线性增长。上一节的第二版服务实现了最朴素的批处理，也暴露了它的问题。这一节从静态批处理讲到连续批处理，再讲 chunked prefill 怎样消除长 prompt 带来的停顿，最后在 MiniServe 上测量几种策略的差别。

## 1. 生成任务为什么难以批处理

图像分类模型的批处理很简单：每张图片的输入大小相同，前向一次就得到结果，同一批的图片同时开始、同时结束。

生成任务不一样。同一批里的请求，输出长度可能相差很大：

- "今天星期几"，几个 token 就结束了；
- "写一首绝句"，几十个 token；
- "写一份两千字的年终总结"，两千多个 token。

如果整批一起开始、一起结束，前两个请求很快就生成完了，但它们在批里的位置要一直空着，等第三个请求写完两千个 token 才能释放。新来的请求也只能等这一批结束。

## 2. 三种批处理方式

| | 调度单位 | 解决了什么 | 剩下的问题 |
|---|---|---|---|
| 静态批处理 | 整批 | 离线处理固定数据集时能用满 GPU | 在线场景要等凑满一批才开始；整批同进同出 |
| 动态批处理 | 整批，加等待超时 | 凑满一批或等待超时就开始，不会无限等待 | 开始之后仍然整批同进同出，短请求的位置空转，新请求进不来 |
| 连续批处理 | 每一步 | 每生成一个 token 都重新组批：结束的序列立刻离开，新请求立刻加入 | 新请求的 prefill 很长时，会拖慢同一步里其他请求的 decode |

**动态批处理**是 Triton Inference Server 等传统服务框架的标准做法，原书第 3 章的第二版服务也是这样实现的。它设置两个参数：最大批大小和最长等待时间。等待时间是一个折中：等得越久，批越大，吞吐越高，但每个请求的延迟也越长。

**连续批处理**由 Orca 论文（OSDI 2022）提出，论文中叫迭代级调度（iteration-level scheduling）。它把调度的粒度从"一批请求"缩小到"一步"。每一步开始前，调度器检查：哪些序列已经结束，把它们移出去；有没有新请求，有空位就加进来。一个序列在第 5 步结束，第 6 步它的位置就被新请求占用了。

因为每一步都能接收新请求，新请求最多等一步，也就不再需要凑批的等待时间。

## 3. 模拟：同一组请求，三种策略

下面的模拟器按 MiniServe 的调度规则，逐步运行 5 个请求，最多同时运行 3 个。代价模型很简单：每一步固定耗时 1 个单位（每步都要读一遍权重），每个 prefill token 再加 1/8 个单位（prefill 受算力限制，token 越多越慢）。

<div class="llm-widget" data-widget="batching-timeline"></div>

依次切换三种策略：

1. **静态批处理。** R1 很快结束，位置却一直空着。R4、R5 要等前三个请求全部结束才能开始，TTFT 很长。
2. **连续批处理，不切块。** R4 一到就被接收，TTFT 短了很多。但它的 32 个 prompt token 在一步内算完，这一步花了 5 个单位，正在 decode 的 R2、R3 在这一步没有输出（红色），它们的最大 ITL 从 1 变成了 6。
3. **连续批处理加切块。** R4 的 prompt 被切成几块，和 R2、R3 的 decode 一起进行，最大 ITL 降下来了；代价是 R4 自己的 TTFT 变长了一些。

再拖动 token 预算和 R4 的 prompt 长度，观察这个取舍的变化。第 4 节解释背后的机制。

## 4. Chunked Prefill

### 4.1 两个上限

连续批处理的调度器每一步要决定放多少工作进去。只限制序列数不够：10 个 decode 中的序列，这一步只有 10 个 token；10 个刚到的 8K 长 prompt，这一步有 8 万个 token，两者的计算量和激活显存相差几千倍。所以调度器同时用两个上限：

| 参数（vLLM 与 MiniServe 同名） | 含义 | 主要影响 | 设得太小 | 设得太大 |
|---|---|---|---|---|
| `max_num_seqs` | 同时运行的序列数上限 | decode 的批大小 | decode 的批小，吞吐上不去 | KV Cache 不够用，频繁抢占（1.3 节） |
| `max_num_batched_tokens` | 一步里所有序列加起来的 token 数上限 | prefill 每一步能处理多少 | 长 prompt 要很多步才算完，TTFT 变长 | 一步耗时太长，同一步里 decode 的序列出现停顿 |

### 4.2 一个 prompt 怎样分几步算完

设 `max_num_batched_tokens = 512`。现在有 12 个序列在 decode，又来了一个 1500 个 token 的 prompt，记作 R。调度器这样分配：

```mermaid
flowchart LR
  B["这一步的预算：512"] --> D["先给 12 个 decode 序列<br/>各 1 个 token，用掉 12"]
  D --> R["剩下 500<br/>切出 R 的前 500 个 token"]
  R --> S["一共 512 个 token<br/>做一次前向计算"]
```

- 第 1 步：12 个序列各生成一个 token；R 计算第 0～499 个位置，写入 KV Cache，不产生输出。
- 第 2 步：12 个序列继续；R 计算第 500～999 个位置。它们的注意力要和缓存里前 500 个位置，以及这一块内部的位置做因果注意力。
- 第 3 步：R 计算第 1000～1499 个位置，得到第一个输出 token，之后转入 decode。

这种做法叫 **chunked prefill**，由 Sarathi-Serve 论文系统地提出，现在是 vLLM V1 的默认行为。它有两个好处：

- **decode 不再被长 prompt 打断。** 每一步的 token 数被限制在预算以内，耗时也就有了上限，ITL 的尾部被削平。
- **GPU 用得更充分。** 12 个序列的 decode，算术强度只有 12 左右，算力大量闲置；夹带 500 个 prefill token 一起经过线性层，几乎不增加读取权重的时间，却多做了很多有用的计算。

代价是 R 自己的 TTFT 变长了，原来一步算完，现在要三步。

在实现上，切块带来一个细节：第二块及以后的块，起始位置不是 0。构造因果掩码时，本块第 $i$ 行对应的绝对位置是"已缓存的长度 + $i$"，它能看到所有已缓存的位置，以及本块中位置不超过自己的部分。项目 1 的第 8 步就要处理这个问题。

### 4.3 预算怎么选

预算是 TTFT 和 ITL 之间的取舍。几个经验：

- decode 为主的负载（输入短、输出长）几乎碰不到 token 预算，限制并发的是 `max_num_seqs` 和 KV Cache 容量。
- prefill 为主的负载（输入长）对预算敏感。预算小，ITL 平稳但 TTFT 长；预算大则相反。vLLM 的默认值随版本和硬件变化，在 2048 到 8192 之间。
- 预算太小还有一个副作用：每一步都有固定的开销（读权重、调度、启动 kernel），预算小意味着同样的 prefill 要更多步，这些开销被重复支付，总吞吐会下降。

## 5. 实验：在 MiniServe 上比较四种配置

MiniServe 实现了这几种策略，用 `--policy`、`--no-chunked-prefill` 和 `--max-num-batched-tokens` 切换。下面是在 CPU 上用微型模型测得的结果：64 个请求，输入长度 51～972（平均 512），输出 6～121（平均 64），按每秒 30 个的泊松过程到达，SLO 为 TTFT ≤ 500 ms、TPOT ≤ 30 ms。

| 配置 | 输出吞吐（token/s） | TTFT P50 / P99（ms） | ITL P99 / 最大（ms） | goodput（请求/s） |
|---|---|---|---|---|
| 静态批处理 | 489 | 4443 / 6842 | 5.5 / 9.2 | 0 |
| 连续批处理，不切块 | 1243 | 56 / 127 | 204 / 396 | 10.7 |
| 连续批处理，切块，预算 2048 | 1256 | 97 / 209 | 81 / 84 | 12.3 |
| 连续批处理，切块，预算 128 | 966 | 1232 / 2125 | 18 / 20 | 4.3 |

几点观察：

- 静态批处理的 ITL 很低，因为批一旦开始就没有新的 prefill 插进来；但新请求要等整批结束，TTFT 达到几秒，没有一个请求满足 SLO。
- 不切块时 TTFT 最短，但长 prompt 进入时，正在 decode 的序列会停顿，ITL 的最大值接近 400 ms。
- 切块、预算 2048 时，ITL 的尾部被限制住了，goodput 最高。
- 预算降到 128，ITL 最平稳，但 prompt 要切很多块，TTFT 变长，吞吐也下降了。

CPU 上的绝对数字没有参考意义，不同机器也会有差异，但几种策略之间的关系和 GPU 上是一致的。复现步骤：

```bash
cd code/project1
python tools/make_tiny_model.py --out /tmp/tiny --layers 4 --hidden 128

# 每次启动一种配置，压测完关掉再换下一种
PYTHONPATH=solution python -m miniserve.server --model /tmp/tiny --device cpu --num-gpu-blocks 4096 --port 8000 \
    --policy static --max-num-batched-tokens 4096          # 或 --no-chunked-prefill --max-num-batched-tokens 4096
                                                            # 或 --max-num-batched-tokens 2048 / 128

python bench.py --base-url http://localhost:8000 --model /tmp/tiny --num-prompts 64 \
    --input-len 512 --output-len 64 --range-ratio 0.9 --request-rate 30 \
    --vocab-size 400 --slo-ttft-ms 500 --slo-tpot-ms 30 --seed 1
```

微型模型的词表只有 400 多个 token，所以 `--vocab-size` 要设成 400，压测工具生成的随机 token id 才不会越界。

有 GPU 的话，可以用 vLLM 做同样的实验：用较长的输入（比如 4096）压测，分别把 `--max-num-batched-tokens` 设为 512、2048、8192，比较 TTFT 和 ITL 的 P99。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "动态批处理通常要设置一个凑批的等待时间，连续批处理为什么不需要？", "options": ["连续批处理的批更大，等待时间可以忽略", "连续批处理每一步都能接收新请求，新请求最多等一步", "连续批处理用 CUDA Graph 预热，不需要等待", "连续批处理把等待转移到了分词阶段"], "answer": 1, "explain": "整批同进同出时，错过一次开批就要等一整批生成完，所以值得多等一会儿凑满。连续批处理按步调度，凑批的收益消失了。"}, {"q": "某服务的请求输入很短（平均 50 个 token），输出很长（平均 1500 个），并发很高。哪个限制最先成为瓶颈？", "options": ["<code>max_num_batched_tokens</code>", "分词器的处理速度", "<code>max_num_seqs</code> 或 KV Cache 容量", "HTTP 连接数"], "answer": 2, "explain": "这类负载几乎全在 decode，每步每个序列只贡献 1 个 token，token 预算远远用不满。限制并发的是序列数上限，以及不断增长的 KV Cache。"}, {"q": "开启 chunked prefill 后，把 <code>max_num_batched_tokens</code> 从 2048 调到 64，会怎样？", "options": ["TTFT 和 ITL 都改善，没有代价", "ITL 更平稳，但长 prompt 要切很多块，TTFT 变长，吞吐下降", "ITL 变差，因为每一步都有 prefill", "没有变化，切块时这个参数不起作用"], "answer": 1, "explain": "预算决定每步最多夹带多少 prefill token。预算越小，decode 被拖慢得越少，但 4000 个 token 的 prompt 要切成 60 多块，每一步的固定开销被重复支付。"}, {"q": "一个 1500 个 token 的 prompt 被切成三块计算。计算第二块（位置 500～999）时，它的注意力需要用到哪些 K、V？", "options": ["只用第二块自己的 500 个位置", "缓存里的前 500 个位置，加上本块中不晚于自己的位置", "全部 1500 个位置", "只用缓存里的前 500 个位置"], "answer": 1, "explain": "切块不改变计算结果。本块第 &#36;i&#36; 行的绝对位置是 &#36;500 + i&#36;，它要看到所有更早的位置：缓存中的 0～499，以及本块中 500 到 &#36;500 + i&#36;。第三块的内容此时还没算，也不应被看到。"}]'></div>

## 延伸阅读

- 原书第 6 章 *Why Batching in Real-Time Serving*、*Dynamic Batching*、*Continuous Batching*、*Continuous Batching with Chunked Prefill*。
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022。提出迭代级调度。
- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024。系统地提出 chunked prefill。
- Anyscale 博客 [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)。
