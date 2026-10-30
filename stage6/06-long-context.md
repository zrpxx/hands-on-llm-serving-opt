# 6.6 长上下文：稀疏注意力与上下文并行

> 书外内容。3.1 节讨论了 MQA、GQA、MLA 与滑动窗口，4.4 节讨论了 KV Cache 的卸载与分层存储。本节讨论 128K 到 1M 上下文下的三个问题：KV 怎样变小、注意力怎样少算、一条序列怎样切到多张卡上。

Agent、代码库问答、长文档分析都需要很长的上下文，主流模型的上下文窗口已经到了 128K 到 1M。长上下文对推理的压力来自三个方面：KV Cache 的显存、decode 时每步读取的 KV、prefill 时与长度平方成正比的计算。

## 1. KV 有多大

```bash
python code/stage6/long_context.py
```

```text
1) 每条序列的 KV Cache
   模型                                每 token      128K        1M
   Llama-3.1-8B（GQA，BF16）           128.0 KB   16.0 GB  128.0 GB
   Qwen3-235B-A22B（GQA，BF16）        188.0 KB   23.5 GB  188.0 GB
   DeepSeek-V3（MLA，BF16）             68.6 KB    8.6 GB   68.6 GB
   DeepSeek-V4.1-Flash（全局 KV，FP4）    0.9 KB    0.1 GB    0.9 GB
```

用 GQA 的模型，一条 1M 的序列就需要上百 GB 的 KV，一张卡放不下。降低 KV 的手段可以叠加：MLA 把 KV 压缩成低维的潜在向量（3.1 节）；KV 量化到 FP8 甚至 FP4（2.5 节）；跨层共享（YOCO 一类的结构，6.1 节）；以及只为少数层保留全局的 KV，其余层只用滑动窗口。DeepSeek-V4.1-Flash 把这些都用上了，每个 token 不到 1 KB，比 MLA 又小了约 80 倍。

## 2. 每一步要读多少

KV 能放下之后，decode 每生成一个 token，完整的注意力仍要把这条序列的全部 KV 读一遍。128K 的序列在 DeepSeek-V3 上每步要读 8.6 GB，在 H800 上约 2.6 ms，而且每条序列都要读自己的那一份。

减少读取量的办法是**只看一部分位置**：

- **滑动窗口**：只看最近的 $w$ 个 token。简单，但窗口外的信息完全看不到，通常与少数全局注意力层交替使用（如 Gemma、gpt-oss）。
- **原生稀疏注意力（NSA）**：DeepSeek 在 2025 年初提出，每个 query 同时走三条分支：把远处的块压缩成摘要、根据摘要挑选最相关的几个块做细粒度注意力、以及一个滑动窗口。
- **DeepSeek Sparse Attention（DSA）**：DeepSeek-V3.2 采用。一个很轻的**闪电索引器**（64 个头，FP8，用 ReLU 代替 softmax）为每个历史位置打分，每个 query 只对得分最高的 2048 个位置做完整的 MLA 注意力。按 DeepSeek 的估算，128K 上下文的 decode 成本降到约六分之一。

`long_context.py` 的第二部分用一个人为构造的例子演示了二者的区别：32K 个位置中有 8 个与当前 query 高度相关，都在最近 4096 个之外。

```text
2) 稀疏注意力：32768 个位置，8 个关键位置都在最近 4096 个之外
   方式                          参与计算的位置   与完整注意力的相对误差
   完整注意力                          32768        0.000
   滑动窗口                             4096        0.989
   精确 top-2048                        2048        0.103
   16 维索引器选出的 top-2048            2048        0.132
   索引器选中了 100% 的关键位置。
```

滑动窗口完全错过了关键信息；按相关性挑选的稀疏注意力只算 6% 的位置，误差约为滑动窗口的十分之一。即使索引器只用 16 维的随机投影打分，也选中了全部关键位置。

注意 DSA 并没有消除与长度成正比的部分：索引器仍要给每个位置打分，只是每个位置读的字节少得多。它也没有减少 KV 的存储，全部 KV 仍要留着，因为下一个 query 可能选中任何位置。

<div class="llm-widget" data-widget="longctx-kv"></div>

## 3. 一条序列切到多张卡：上下文并行

prefill 一条 1M 的序列，计算量与长度的平方成正比，单卡要算很久。**上下文并行**（Context Parallelism，CP）把一条序列沿长度切成几段，分给几张卡，各自计算自己那一段的 query，KV 在卡之间传递（Ring Attention 就是一种实现：KV 块沿环形依次传给下一张卡，传输与计算重叠）。

因果注意力带来一个负载问题：第 $i$ 个 token 要与前面 $i$ 个 token 计算，越靠后的 token 越贵。按顺序切分时，最后一张卡算得最多：

```text
3) 上下文并行：131072 个 token 的 prefill 切到 8 张卡
   切分方式                                    最忙 / 平均
   按顺序切成 8 段                                1.87 倍
   切成 16 段，第 i 张卡拿第 i 段和倒数第 i 段      1.00 倍
```

首尾配对（也称 zigzag 切分）之后，每张卡的计算量几乎相同。DeepInfer 的"均衡的上下文并行"（balanced context parallelism）指的就是这类做法。

decode 时也可以用 CP：把一条长序列的 KV 分散存在几张卡上，每张卡算自己那部分的注意力，再合并（用 3.2 节在线 softmax 的方法）。这让单条序列的 KV 不再受一张卡的显存限制。

## 4. 结构与系统的配合

长上下文的优化越来越多地发生在模型结构上，推理系统随之调整：

| 结构上的做法 | 对推理系统的要求 |
|---|---|
| 滑动窗口与全局层交替 | KV Cache 管理器要支持不同层有不同的缓存长度（vLLM 的混合分配器） |
| 跨层共享 KV、编码器与解码器分开 | prefill 时上层可以只算最后一段（6.1 节的 bounded replay） |
| 稀疏注意力的索引器 | 需要 top-k 等专门的算子，还要为索引键单独分配缓存 |
| FP4 的 KV | 注意力 kernel 要直接读 4 位的 KV |

## 5. 实验

- **CPU**：把 `long_context.py` 中索引器的维度从 16 降到 4，召回率和误差怎样变化？
- **CPU**：把关键位置的数量从 8 增加到 4000（超过 top-k 的 2048），稀疏注意力的误差怎样变化？这说明 top-k 应该怎样选？
- **组件**：对 DeepSeek-V3.2，上下文从 4K 增加到 1M，DSA 每步读取量的增长比完整注意力慢多少？
- **GPU**：用 vLLM 部署一个支持 128K 的模型，在 8K、32K、128K 输入下测 TTFT 和 TPOT，并用组件估算 KV 的读取时间占 TPOT 的比例。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "DSA 每步只对 2048 个位置做完整的注意力。它是否消除了与上下文长度成正比的开销？", "options": ["完全消除了", "没有：索引器仍要为每个位置打分，只是每个位置读的字节少得多；KV 也仍要全部存着", "只在 prefill 时消除", "消除了 KV 的存储，但没有减少计算"], "answer": 1, "explain": "下一个 query 可能选中任何位置，所以 KV 不能丢。上下文越长，节省越明显。"}, {"q": "上下文并行中，为什么按顺序把序列切成 N 段会导致负载不均？", "options": ["最后一段的 token 最多", "因果注意力中越靠后的 token 要看的历史越多，最后一张卡算得最多", "第一张卡要处理系统提示词", "KV 传输只能单向进行"], "answer": 1, "explain": "切成 2N 段、第 i 张卡拿第 i 段和倒数第 i 段，各卡的计算量几乎相同。"}, {"q": "一条关键信息出现在 10 万个 token 之前。以下哪种注意力一定看不到它？", "options": ["完整注意力", "窗口为 4096 的滑动窗口", "DSA", "NSA"], "answer": 1, "explain": "滑动窗口只看最近的 w 个 token，所以通常与少数全局层交替使用。稀疏注意力按相关性挑选位置，能看到远处的内容。"}]'></div>

## 延伸阅读

- Yuan et al., [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089), 2025。
- DeepSeek-AI, [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556), 2025。DSA 部分。
- Liu et al., [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889), 2023。
