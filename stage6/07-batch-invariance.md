# 6.7 批不变推理与确定性采样

> 书外内容。原书第 10 章引用了 Thinking Machines 的文章，强调确定性对 RL 的重要性（5.6 节），但没有讨论原理。本节的 CPU 实验和组件是书外内容。

很多人以为，温度设为 0，同一个提示词的输出就一定相同。实际并非如此。Thinking Machines 在 2025 年 9 月的文章中做了一个实验：用 vLLM 部署 Qwen3-235B-A22B，温度为 0，同一个提示词请求 1000 次，得到了 80 种不同的输出。前 102 个 token 完全相同，从第 103 个开始分岔。

## 1. 原因不是"GPU 的并行有随机性"

一种常见的解释是：GPU 上的线程并发执行，浮点加法的顺序不确定，于是结果不确定。这个解释不完整。实际上，LLM 前向中的大多数 kernel 本身是**确定的**：同样的输入、同样的形状，跑多少次结果都一样。

真正的原因是**批不变性**（batch invariance）的缺失：一个请求的计算结果，取决于它和哪些请求被放在同一个 batch 里。

- 浮点加法不满足结合律，加法的顺序不同，结果的最后几位就不同。
- 为了在各种形状下都跑得快，kernel 会根据 batch 大小选择不同的计算策略：batch 小时把归约维度切成多段并行（split-K），batch 大时用另一种分块方式。
- 对用户来说，他的请求和谁在同一个 batch 里是随机的，于是归约的顺序也是随机的。

在 CPU 上就能复现：

```bash
python code/stage6/batch_invariance.py
```

```text
2) PyTorch 矩阵乘法：第 0 行单独计算，与放在不同大小的 batch 里计算（torch.bfloat16）
   batch     1：与单独计算的最大差 0.00e+00
   batch     4：与单独计算的最大差 0.00e+00
   batch    16：与单独计算的最大差 7.81e-03
   batch    64：与单独计算的最大差 3.05e-05
   batch  1024：与单独计算的最大差 7.81e-03

3) split-K：分几段由 batch 大小决定时会变，固定分段就不变
   batch    1：分段数随 batch 变（16 段），差 0.0e+00；固定分 8 段，差 0.0e+00
   batch   64：分段数随 batch 变（ 4 段），差 6.1e-05；固定分 8 段，差 0.0e+00
   batch  256：分段数随 batch 变（ 1 段），差 1.4e-04；固定分 8 段，差 0.0e+00
```

同一行数据，在不同大小的 batch 里计算，结果的最后几位不同。差异虽小，但只要某一步的两个候选 token 的 logits 足够接近，选出的 token 就会不同，之后的生成就完全分岔。

<div class="llm-widget" data-widget="fp-order"></div>

## 2. 让每个算子都批不变

Thinking Machines 的做法是逐个改写涉及归约的算子，**让每个元素的归约顺序与 batch 大小无关**：

- **RMSNorm**：每一行由一个线程块完成归约，不论 batch 多大都用同样的策略。代价是 batch 很小时，线程块数少于 SM 数，GPU 用不满。
- **矩阵乘法**：固定一种分块配置，不用 split-K。小 batch 时损失一部分性能，文章中的例子约慢 20%。
- **注意力**：最复杂。decode 时常用 split-KV（把一条序列的 KV 切成几段并行计算，再合并），段数通常根据 batch 和序列长度决定。改为**固定每段的长度**而不是固定段数，每个位置的归约顺序就与 batch 无关了。另外，prefill 的分块和 decode 读 KV Cache 的方式要统一，保证同一个 token 无论在 prefill 中还是 decode 中计算，顺序都相同。

性能代价：文章中用 Qwen3-8B 生成 1000 条序列，vLLM 默认用时 26 秒，未优化的确定性版本 55 秒，改进注意力 kernel 后为 42 秒。

## 3. 推理框架中的支持

- **vLLM**：设置环境变量 `VLLM_BATCH_INVARIANT=1`，目前为 beta。支持计算能力 8.0 及以上的 NVIDIA GPU、部分 AMD 和 Intel GPU，已验证 DeepSeek-V3、Qwen3、Llama 3 等模型。张量并行时，all-reduce 也要用固定的归约顺序。
- **SGLang**：提供 `--enable-deterministic-inference`。
- **DeepInfer**：`DEEPINFER_BATCH_INVARIANT=1` 开启批不变推理；要得到完全确定的采样，还要关闭投机解码、在 `SamplingParams` 中固定 `seq_seed`（仅 NVIDIA）。

为什么投机解码要关掉？草稿被接受多少，取决于这一步的计算结果；接受数量不同，下一步各序列的长度就不同，batch 的组合也就不同。即使每个算子都批不变，验证时多个草稿 token 一起计算的方式与逐个生成时也不同。

采样也要确定：温度大于 0 时，随机数要与请求绑定（每个请求一个种子，按 token 位置生成随机数），而不是全局共享一个随机数生成器。否则一个请求的随机数取决于它前面有几个请求被采样。

## 4. 谁需要确定性

- **RL 训练**（5.6 节）：采样时推理引擎给出每个 token 的对数概率，训练时训练框架重新计算。两者不一致，训练就变成了离策略的。文章中的实验：用批不变的推理并让训练前向与之数值一致，采样与训练之间的 KL 散度为 0，实现了真正的同策略训练；不做修正的对照组训练崩溃。
- **评测与调试**：同一个配置跑两次，分数应当相同；线上问题应当能够复现。
- **缓存与审计**：同样的请求得到同样的回答。

对普通的在线服务，大多数时候不需要为它付出性能代价。

## 5. 实验

- **CPU**：运行 `batch_invariance.py`，在你的机器上，哪些 batch 大小会产生差异？把线程数改为 1（`torch.set_num_threads(1)`）再试一次。
- **CPU**：修改第 3 部分，把固定分段数改为固定每段的长度（比如每段 256 个），验证它仍然批不变。
- **GPU**：用 vLLM 部署一个小模型，温度为 0，同一个提示词在并发 1 和并发 64（其余请求是不同的提示词）下各请求 100 次，统计不同输出的种数；再设置 `VLLM_BATCH_INVARIANT=1` 重复，比较种数和吞吐。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "温度为 0、同一个提示词，多次请求得到不同输出。最主要的原因是：", "options": ["GPU 线程并发导致每次 kernel 的结果随机", "请求和谁在同一个 batch 里是随机的，kernel 根据 batch 大小选择不同的归约策略，浮点加法的顺序随之改变", "采样器有 bug", "KV Cache 被其他请求覆盖"], "answer": 1, "explain": "大多数 kernel 在同样的形状下是确定的，缺的是批不变性。"}, {"q": "批不变的注意力 kernel 中，split-KV 应该怎样切分？", "options": ["固定切成 4 段", "固定每段的长度，而不是固定段数", "根据 batch 大小决定段数", "不切分"], "answer": 1, "explain": "固定每段的长度，每个位置所在的段和归约顺序就与 batch 和其他请求无关。"}, {"q": "批不变推理对 RL 训练的意义是：", "options": ["让采样更快", "让推理引擎给出的对数概率与训练框架重新计算的一致，实现真正的同策略训练", "减少显存占用", "提高奖励模型的准确率"], "answer": 1, "explain": "Thinking Machines 的实验中，采样与训练之间的 KL 散度降到 0；不做修正的对照组训练崩溃。"}]'></div>

## 延伸阅读

- Horace He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), 2025。
- [vLLM 文档：Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)。
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) README 中的 Batch-Invariant Inference 一节。
