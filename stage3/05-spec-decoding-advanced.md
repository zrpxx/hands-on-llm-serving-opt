# 3.5 投机解码方法与调参

> 对应原书第 7 章 *Tuning and Usage*、*Limitations*、*Hands-on Speculative Decoding*。MTP 和树形验证是书外内容。

上一节的公式说明，投机解码的效果取决于两个量：草稿被接受的概率 $\alpha$，以及起草的开销 $c$。各种投机解码方法的区别，主要就在于怎样在这两者之间取舍。这一节先介绍主流的几种方法，再讨论怎样选择草稿长度 $K$，最后看什么时候不该用投机解码。

## 1. 起草的几种方式

### 1.1 独立的小模型

最直接的做法是用同一系列的小模型作为草稿，比如用 Qwen3-0.6B 为 Qwen3-32B 起草。同一系列的模型共用分词器，预训练数据相近，接受率比较高。原书还建议对草稿模型做激进的量化，因为它出错了也有目标模型兜底。

如果能做一些训练，从目标模型**蒸馏**出一个草稿模型（2.8 节）通常比直接拿现成的小模型更好，它模仿的就是目标模型自己的分布。

这种做法的问题在于工程上：要在同一组 GPU 上运行两个模型，各自有 KV Cache，各自需要调度；草稿模型虽小，但 $K$ 次串行的前向也有不小的开销，小模型的 decode 同样受限于 kernel 启动这样的固定开销（3.3 节）。

### 1.2 自起草：Medusa 与 EAGLE

自起草的方法不用独立的模型，而是在目标模型上加一个很小的模块。

**Medusa** 在模型的最后一层之后加几个额外的预测头，第 $i$ 个头直接预测之后第 $i+1$ 个位置的 token。一次前向就能得到后面几个位置的候选，把每个位置的前几个候选组合起来，就是一棵候选树。原书图 7-2 中，一次前向产出了"It is difficult"三个 token。

**EAGLE** 的出发点是：直接预测几步之后的 token 很难，但预测下一步的隐藏状态比较容易。它训练一个只有一层 Transformer 的草稿头，以目标模型最后一层的特征和已经确定的 token 作为输入，自回归地预测后续的特征，再用目标模型的输出头把特征变成 token。之后的两个版本：

- **EAGLE-2** 根据草稿头的置信度动态地扩展候选树，容易预测的地方猜得更远，难预测的地方少猜几个。
- **EAGLE-3** 不再预测特征本身，而是直接预测 token，并且融合目标模型低、中、高三层的特征作为输入。训练时还模拟了推理时的多步起草，让草稿头适应自己的误差。原书写作时（2025 年底），EAGLE-3 是效果最好的自起草方法之一。

EAGLE 的草稿头需要针对每个目标模型单独训练。好在常见的开源模型都已经有了训练好的 EAGLE-3 草稿头，比如原书实验用的 `RedHatAI/Qwen3-32B-speculator.eagle3`。

### 1.3 MTP：训练时就带上草稿头

DeepSeek-V3 在预训练时加入了**多 token 预测**（Multi-Token Prediction，MTP）的目标：主模型之外有一个额外的模块，预测再往后一个位置的 token。它本来是为了改善训练，但推理时正好可以作为草稿。DeepSeek 报告，第二个 token 的接受率在 85% 到 90% 之间，decode 的速度提高到约 1.8 倍。

之后 GLM-4.5、Qwen3-Next 等模型也发布了 MTP 层。和 EAGLE 相比，MTP 是与模型一起预训练的，接受率更高，也不需要另外找草稿头。6.5 节会讨论 MTP 在大规模部署中的用法。

### 1.4 n-gram

n-gram 方法完全不需要模型：在已有的 token（包括 prompt）中查找与结尾几个 token 相同的片段，把它后面的内容作为草稿。原书表 7-1 举了一个例子：序列中前面出现过"a quick brown fox"，结尾又是"quick brown"，就提议"fox"。

它的开销几乎为零，在两类场景中效果很好：

- 生成的内容大量重复 prompt，比如改写文章、按模板填写、代码编辑；
- 结构化的输出，比如 JSON、SQL。

原书的建议是先试 n-gram，再考虑其他方法。

### 1.5 比较

| 方法 | 接受率 | 起草开销 | 需要额外准备 | 适合 |
|---|---|---|---|---|
| n-gram | 低，依赖内容的重复 | 几乎为零 | 不需要 | 改写、模板、代码编辑 |
| 独立小模型 | 中 | 较高 | 同系列的小模型 | 有合适的小模型时 |
| Medusa | 中 | 低 | 训练预测头 | 较早的方法，现在用得不多 |
| EAGLE-3 | 高 | 低 | 训练草稿头，常见模型有现成的 | 通用 |
| MTP | 很高 | 低 | 模型自带 | 模型发布了 MTP 层时首选 |

## 2. 树形验证

前面讨论的都是一条草稿链。如果第一个 token 有两个都很可能的候选，只选一个就浪费了机会。**树形验证**同时提出多条候选路径，组成一棵树，目标模型一次前向验证整棵树，选出被接受的最长路径。

实现上的关键是**树形注意力掩码**：把树中的所有节点排成一个序列送进目标模型，每个节点只能看到自己的祖先，看不到兄弟和其他分支。SGLang 的 EAGLE 实现就使用了树形验证，相关参数包括每步保留几个候选（`--speculative-eagle-topk`）和整棵树的节点数（`--speculative-num-draft-tokens`）。

树越大，被接受的期望长度越长，但验证的计算量也越大。和 $K$ 一样，树的大小需要按负载调节。

## 3. 选择 K

$K$ 是最重要的调节参数。原书的建议：

- 从较小的 $K$（2 或 4）开始，作为稳定的基线，再逐步增大；
- 很多情况下最优的 $K$ 在 4 到 8 之间；
- 生成的内容很容易预测时，比如结构化输出或者 Agent 的函数调用，$K$ 可以大到 16 甚至 32。

调参时最有用的信息是**每个位置的接受率**。vLLM 会在日志和监控指标中给出这个数据。原书举的例子是 $K = 6$ 时各位置的接受率为 $[0.8, 0.7, 0.6, 0.5, 0.1, 0.02]$：第 5、6 个位置几乎从不被接受，却要付出同样的起草开销，应该把 $K$ 降到 4。

注意这些数字的含义：第 $i$ 个位置的接受率是"前 $i$ 个草稿全部被接受"的比例，所以它一定是递减的。3.4 节的实验中，各位置的接受率是 0.49、0.28、0.12、0.05，第 3、4 个位置的收益已经很小了。

## 4. 什么时候不该用

原书列出了几种投机解码帮不上忙，甚至起反作用的情况：

- **它只加速 decode**。输入很长、输出很短的任务（比如分类、摘要），瓶颈在 prefill，投机解码几乎没有作用。
- **并发高时可能降低吞吐**。batch 大的时候，一步要处理的 token 已经让 GPU 接近计算瓶颈，再验证被拒绝的草稿就是浪费算力。这时投机解码也许还能改善单个请求的延迟，但整体吞吐会下降。
- **固定的 $K$ 不能适应变化的负载**。并发低时适合大 $K$，并发高时适合小 $K$ 甚至关闭。

下面的组件按 Roofline 估算了投机解码的吞吐加速比怎样随并发变化：

<div class="llm-widget" data-widget="spec-batch"></div>

可以观察到两点。上下文较短时，并发超过几十之后，线性层进入计算瓶颈，加速比迅速下降，最终低于 1。上下文较长时，注意力部分读 KV Cache 的时间占了大头，而验证 $K+1$ 个查询读的是同一份 KV，所以即使并发很高，投机解码仍然有收益。后一点在原书中没有提到，但有研究（如 MagicDec）专门讨论过长上下文、大 batch 下的投机解码。

这个估算没有包含起草和验证在调度、采样上的额外开销，实测的结果通常更差一些。原书的结论是：投机解码最适合延迟敏感、batch 较小、可以用一些吞吐和 TTFT 换取更低 ITL 的场景。

## 5. 动手：原书的实验

原书在 Qwen3-32B 上比较了四种配置：不用投机解码、n-gram、调整后的 n-gram、EAGLE-3。

```bash
# n-gram
vllm serve Qwen/Qwen3-32B --max-model-len 2048 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 6,
                         "prompt_lookup_min": 4, "prompt_lookup_max": 6}'

# 调整后的 n-gram：更短的最小匹配长度、更长的最大匹配长度、更小的 K
vllm serve Qwen/Qwen3-32B --max-model-len 2048 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4,
                         "prompt_lookup_min": 2, "prompt_lookup_max": 128}'

# EAGLE-3
vllm serve Qwen/Qwen3-32B --max-model-len 2048 \
  --speculative-config '{"method": "eagle3", "model": "RedHatAI/Qwen3-32B-speculator.eagle3",
                         "num_speculative_tokens": 3}'
```

`prompt_lookup_min` 和 `prompt_lookup_max` 是 n-gram 匹配长度的下限和上限。原书的命令中还有 `--disable-log-requests`，较新的 vLLM 默认就不记录请求日志，这个参数已经被弃用。

结果：

- **并发为 1**：原始的 n-gram 略有提升，调整后的 n-gram 提升约 16%；EAGLE-3 的吞吐几乎翻倍，从每秒 28.9 个 token 提高到 56.5 个。
- **并发为 16**：两种 n-gram 都比不用投机解码更慢，提议和验证的开销在高并发下变得显著；EAGLE-3 仍然更快，但优势缩小。EAGLE-3 的 TTFT 明显变长，原书认为这是草稿头的额外开销造成的。

这个实验印证了第 4 节的分析。部署前要在自己的数据和目标并发下实测，根据每个位置的接受率选择方法和 $K$。

## 6. 实验：在 CPU 上调 K

用 3.4 节的 `spec_demo.py`，可以观察 $K$ 和接受率的关系：

```bash
cd code/project3
FORGE_IMPL=solution python spec_demo.py --target /tmp/tiny-lm --draft /tmp/draft-lm --data ../project2/data
```

试着回答两个问题：

1. 根据输出中各位置的接受率，用 3.4 节的公式算出最优的 $K$，和实测的"估算加速比"一列对比。
2. 草稿模型如果更大（比如 2 层、隐藏维度 192），接受率会提高，$c$ 也会增大。重新训练一个草稿模型，看最优的 $K$ 和加速比怎样变化。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "让模型把一篇长文改得更通顺，输出大部分照抄原文。应该先尝试哪种投机解码？", "options": ["n-gram，从 prompt 中查找与结尾相同的片段作为草稿，开销几乎为零", "独立的小模型", "训练一个新的 EAGLE 草稿头", "不用投机解码"], "answer": 0, "explain": "输出大量重复输入时，n-gram 的接受率很高，而且不需要任何模型。原书也建议把它作为第一个尝试的方法。"}, {"q": "K = 6 时各位置的接受率是 [0.8, 0.7, 0.6, 0.5, 0.1, 0.02]。怎样调整？", "options": ["把 K 增大到 8", "把 K 降到 4，最后两个位置几乎从不被接受，却要付出同样的起草开销", "关闭投机解码", "不需要调整"], "answer": 1, "explain": "第 i 个位置的接受率是前 i 个草稿全部被接受的比例。后两个位置的期望收益只有 0.12 个 token，不值得两次起草。"}, {"q": "EAGLE 与 Medusa 的主要区别是：", "options": ["EAGLE 需要一个独立的大模型", "EAGLE 用一个小的草稿头自回归地预测目标模型的特征，再用输出头得到 token；Medusa 用多个头直接预测之后几个位置的 token", "Medusa 的接受率更高", "EAGLE 不需要训练"], "answer": 1, "explain": "直接预测几步之后的 token 很难，预测下一步的特征更容易。EAGLE-3 进一步融合了多层特征，改为直接预测 token。"}, {"q": "上下文很短、并发很高时，开启投机解码后总吞吐下降了。最可能的原因是：", "options": ["草稿模型的输出错了", "线性层已经接近计算瓶颈，验证被拒绝的草稿 token 消耗了算力，换不回等量的产出", "KV Cache 不够", "接受率随并发下降"], "answer": 1, "explain": "投机解码用空闲的算力换取更少的前向次数。并发高时算力已经不空闲了。上下文长时，注意力读 KV 的时间占大头，验证多个查询读的是同一份 KV，情况会好一些。"}]'></div>

## 延伸阅读

- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), ICML 2024。
- Li et al., [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077), ICML 2024；[EAGLE-2](https://arxiv.org/abs/2406.16858), EMNLP 2024；[EAGLE-3](https://arxiv.org/abs/2503.01840), 2025。
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 第 2.2 节与第 5.4.3 节。
- Sadhukhan et al., [MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding](https://arxiv.org/abs/2408.11049), ICLR 2025。
- [vLLM 文档：Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html)。
