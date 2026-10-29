# 3.1 注意力变体：MQA、GQA、MLA、滑动窗口

> 对应原书第 6 章 *Scalable Attention Mechanisms*。滑动窗口和混合注意力是书外内容。

0.4 节说过，decode 阶段每生成一个 token，都要把这个请求全部的 KV Cache 从显存读一遍。KV Cache 越大，每一步读得越多，能同时运行的请求越少，能支持的上下文越短。所以近几年的模型在结构上做了很多减小 KV Cache 的设计。这些设计是在训练时确定的，部署时没法改变，但它们决定了一个模型好不好部署，也决定了推理引擎需要什么样的 kernel。

这一节介绍四类设计：减少 KV 头的 MQA 和 GQA，压缩 KV 的 MLA，限制每层能看多远的滑动窗口，以及正在兴起的混合注意力。

## 1. 从 MHA 到 GQA

原始的多头注意力（MHA）中，每个查询头都有自己的一组 K、V。一层的 KV Cache 是每个 token $2 \times n_{\text{heads}} \times d_{\text{head}}$ 个数。

**MQA**（Multi-Query Attention，Shazeer，2019）走到了另一个极端：所有查询头共用一组 K、V，KV Cache 缩小为原来的 $1/n_{\text{heads}}$。32 个头的模型，KV Cache 只剩 1/32。代价是模型质量有明显的下降，训练也不太稳定。

**GQA**（Grouped-Query Attention，Ainslie 等，2023）取了折中：把查询头分成若干组，每组共用一组 K、V。Llama-3.1-8B 有 32 个查询头、8 个 KV 头，每 4 个查询头共用一组，KV Cache 是 MHA 的 1/4。GQA 论文还给出了把现有的 MHA 模型改成 GQA 的方法：把每组的 K、V 头取平均，再用原预训练约 5% 的计算量继续训练。现在主流的开源模型几乎都用 GQA。

判断一个模型用的是哪一种，看 `config.json` 里的两个字段：

```json
"num_attention_heads": 32,
"num_key_value_heads": 8
```

两者相等是 MHA，后者为 1 是 MQA，介于两者之间是 GQA。

GQA 对推理的意义不只是省显存。decode 时，一组 K、V 被同组的几个查询头共用，读一次 KV 可以做几倍的计算，算术强度（0.7 节）提高了。好的注意力 kernel 会把同一组的查询头放在一起处理，充分利用这一点。项目 1 和项目 3 的 kernel 中，第 $h$ 个查询头读第 $h / (n_{\text{heads}} / n_{\text{kv}})$ 个 KV 头，就是 GQA 的实现。

## 2. MLA

DeepSeek-V2 提出的 **MLA**（Multi-head Latent Attention）换了一种思路：不减少头数，而是压缩 K、V。

每个 token 的隐藏状态先经过一个降维矩阵，得到一个很短的隐向量 $c$（DeepSeek-V3 中是 512 维）。每个头的 K、V 由 $c$ 乘以各自的升维矩阵得到：

$$
c = h W^{DKV}, \qquad K_h = c\, W^{UK}_h, \qquad V_h = c\, W^{UV}_h
$$

KV Cache 只需要存 $c$。RoPE 需要按位置旋转 K，和这种压缩不兼容，所以 MLA 另外给每个 token 一个 64 维、所有头共用的 RoPE 分量。DeepSeek-V3 每层每个 token 只缓存 $512 + 64 = 576$ 个数。论文的说法是，它的 KV Cache 相当于只有 2.25 组的 GQA，效果却比 MHA 更好。

**矩阵吸收。** 如果 decode 时先把 $c$ 解压成 128 个头的 K、V 再计算，就失去了压缩的意义：每一步都要临时生成大量的 K、V。实际的做法是利用矩阵乘法的结合律，把 $W^{UK}_h$ 吸收到查询中：

$$
q_h \cdot K_h^\top = q_h \left(c\, W^{UK}_h\right)^\top = \left(q_h W^{UK\top}_h\right) c^\top
$$

先把查询变换到 512 维的隐空间，再直接与缓存的 $c$ 计算分数；$W^{UV}_h$ 同理吸收到输出投影中。这样，128 个查询头共用同一份 $c$，注意力的计算等价于一个 K、V 都是 512 维的 MQA。`code/stage3/mla_absorb.py` 验证了两种做法的结果相同：

```bash
python code/stage3/mla_absorb.py
```

```text
两种做法的最大差： 3.5762786865234375e-07
朴素做法要临时生成的 K、V：64 MiB（BF16）；吸收后只读 KV Cache 本身：1.0 MiB
吸收后注意力部分每读 1 字节 KV 的计算量：约 256 FLOPs，同样头数的 GQA（8 个 KV 头）约为 16
```

最后一行说明了 MLA 对推理的另一个影响：decode 时注意力的算术强度很高，在 H100 上已经接近计算瓶颈，而不像 GQA 那样完全受限于带宽。这也是 DeepSeek 要专门开源 FlashMLA 这样的 kernel 的原因。prefill 时情况相反，查询很多，吸收反而增加计算量，所以 prefill 通常还是先解压成普通的多头注意力来算。

## 3. 滑动窗口

另一种思路是限制每一层能看到的范围。**滑动窗口注意力**（Sliding Window Attention）中，每个 token 只关注最近的 $W$ 个 token。这一层的 KV Cache 最多只需要保存 $W$ 个 token，不随上下文增长。

只用滑动窗口会丢失远处的信息，所以现在的模型把滑动窗口层和全局注意力层交替使用：

| 模型 | 设计 |
|---|---|
| Mistral 7B | 所有层窗口 4096，靠多层叠加间接看到更远的内容 |
| Gemma 3 | 每 6 层中有 5 层是窗口 1024 的局部注意力，1 层是全局注意力 |
| gpt-oss | 全局注意力层与窗口 128 的局部注意力层交替，各占一半 |

上下文越长，这种设计省得越多。以 Gemma-3-27B 为例，128K 上下文时，52 个局部层每层只存 1024 个 token，KV Cache 主要来自 10 个全局层。

滑动窗口对推理引擎提出了新的要求：不同层的 KV Cache 长度不同，滑出窗口的块可以提前释放，前缀缓存的命中规则也不一样。vLLM 为此引入了混合 KV Cache 管理器，对不同类型的层分别分配块。

gpt-oss 还使用了 **attention sink**：每个头有一个可学习的偏置，加在 softmax 的分母中，相当于一个永远存在、但不贡献任何值的虚拟 token。StreamingLLM 的研究发现，模型倾向于把大量注意力分给开头的几个 token，把它们当作"无处可去"时的落脚点。滑动窗口去掉这些 token 会让效果变差，显式的 sink 解决了这个问题。

## 4. 比较

下面的组件按各模型的实际配置计算一个请求的 KV Cache 大小：

<div class="llm-widget" data-widget="attn-variants"></div>

几个值得注意的对比：

- Llama-2-7B 用的是 MHA，它的 KV Cache 比参数量大十倍的 Llama-3.1-70B 还大。
- DeepSeek-V3 有 6710 亿参数，每个 token 的 KV Cache 却只有约 69 KiB，比 8B 的 GQA 模型还小。如果用同样头数和维度的 MHA，会大约 70 倍。
- 上下文从 1K 增加到 128K 时，Gemma 3 和 gpt-oss 的 KV Cache 增长得比其他模型慢得多。

## 5. 混合注意力

最近还有一类模型把注意力层和**线性注意力**层混合使用。线性注意力（如 Mamba、DeltaNet 等）为每个序列维护一个固定大小的状态，而不是随长度增长的 KV Cache，计算量也与序列长度成线性关系。Qwen3-Next 就以 3:1 的比例混合了 Gated DeltaNet 层和普通的注意力层。

对推理引擎来说，这意味着同一个模型里有两种完全不同的缓存：注意力层的分页 KV Cache，以及线性层的定长状态。后者无法分页，前缀缓存也很难做，因为状态里混合了全部历史，不能按块复用。各引擎都在逐步支持这类模型，第 6 部分会再提到。

## 6. 对部署的影响

- **结构决定了显存的需求。** 估算 KV Cache（0.6 节）时，要用 `num_key_value_heads`、MLA 的 `kv_lora_rank` 和 `qk_rope_head_dim`，以及滑动窗口的配置，不能简单地用隐藏维度计算。
- **结构决定了需要什么 kernel。** MLA 需要专门的 kernel（FlashMLA、FlashInfer 的 MLA kernel），滑动窗口需要支持窗口掩码，attention sink 需要 kernel 支持额外的分母项。新模型发布后，推理引擎往往要过一段时间才能高效地支持它，这是选型时要考虑的。
- **结构影响并行方式。** MLA 只有一份隐向量，没有多个 KV 头可以在张量并行时切分，每张卡都要存完整的 KV Cache。DeepSeek 的部署因此对注意力部分用数据并行，第 4 部分会讨论。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "一个模型的 config.json 中 num_attention_heads 为 64，num_key_value_heads 为 8。它的 KV Cache 是同样结构的 MHA 的多少？", "options": ["1/64", "1/8", "1/4", "一样大"], "answer": 1, "explain": "KV Cache 的大小由 KV 头数决定。64 个查询头共用 8 组 K、V，每 8 个查询头一组，是 MHA 的 1/8。"}, {"q": "MLA 在 decode 时为什么要做矩阵吸收，而不是先把隐向量解压成每个头的 K、V？", "options": ["解压会损失精度", "解压要在每一步临时生成全部 token、全部头的 K、V，读写量远大于缓存本身，压缩就失去了意义", "吸收之后 KV Cache 变得更小", "吸收之后不需要 RoPE"], "answer": 1, "explain": "吸收后查询先变换到隐空间，直接与缓存的隐向量计算，所有头共用一份隐向量，相当于一个 512 维的 MQA，读的只是缓存本身。"}, {"q": "Gemma 3 每 6 层中有 5 层是窗口 1024 的局部注意力。上下文从 8K 增加到 128K 时，它的 KV Cache 大约怎样变化？", "options": ["增加 16 倍，和普通模型一样", "几乎不变", "增长得比普通模型慢得多，主要由占 1/6 的全局层随上下文增长", "变小"], "answer": 2, "explain": "局部层每层最多存 1024 个 token，与上下文长度无关；只有全局层的 KV 随上下文线性增长。上下文越长，这种设计节省得越多。"}, {"q": "关于 GQA 对 decode 性能的影响，下面哪个说法正确？", "options": ["GQA 只减少显存占用，对速度没有影响", "一组 K、V 被多个查询头共用，读一次 KV 能做几倍的计算，注意力的算术强度提高", "GQA 让 decode 变成计算瓶颈", "GQA 需要更多的 KV 头"], "answer": 1, "explain": "decode 的注意力是带宽瓶颈，减少读取量直接带来加速。kernel 把同组的查询头放在一起处理，才能充分利用这一点。"}]'></div>

## 延伸阅读

- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019。
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), EMNLP 2023。
- DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024，第 2.1 节 MLA。
- Gemma Team, [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), 2025。
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), ICLR 2024。
