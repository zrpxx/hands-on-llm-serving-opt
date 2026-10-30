# 4.4 KV Cache 进阶：长上下文与分层存储

> 对应原书第 7 章 *Advanced KV Caching*。KV 传输与加载的带宽估算是书外内容。

1.4 节的前缀缓存把算好的 KV 留在显存里，下次遇到相同的前缀直接复用。但显存很小：一张 H100 放完 Qwen3-8B 的权重，剩下的空间只够存三十多万个 token 的 KV。当应用需要反复使用很长的上下文，比如整个代码仓库、一个客户的全部文档、几十轮的对话历史，显存就不够用了。

这一节讨论两个问题：长上下文的应用在成本和延迟上怎样权衡；KV Cache 放不下显存时，怎样把它存到 CPU 内存、SSD 甚至远端，需要时再取回来。

## 1. RAG 与长上下文

处理大量知识的传统做法是 **RAG**（检索增强生成，原书第 4 章）：每次请求时检索出最相关的几段文档，拼进 prompt。prompt 不会太长，TTFT 可控。

另一种做法是原书所说的 **CAG**（缓存增强生成）：把全部相关的上下文都放进 prompt，作为固定的前缀，用前缀缓存保存它的 KV，每个请求只追加自己的问题。前缀缓存本身就可以看作 CAG 最朴素的实现。两个变化让这种做法越来越可行：模型的上下文窗口已经达到十万到百万 token，而且长上下文的效果在不断改善；KV Cache 可以卸载到 CPU 内存、SSD 或远端存储，容量不再受限于显存。

### 1.1 延迟

原书图 7-20 比较了两种做法需要 prefill 的 token 数。两者都有 500 个 token 的系统提示词，可以被缓存：

| | RAG | CAG |
|---|---|---|
| 系统提示词（已缓存） | 500 | 500 |
| 检索到的文档 | 10 段 × 500 = 5000，每次不同，无法缓存 | |
| 固定的长上下文（已缓存） | | 100,000 |
| 用户的问题 | 500 | 500 |
| 需要 prefill 的 token | 5500 | 500 |

CAG 只需要 prefill 约十分之一的 token。原书据此估计，如果 RAG 的 TTFT 是 5 秒，CAG 可以降到约 0.5 秒，而且还没算上 RAG 的向量检索时间。

> [!NOTE]
> 这个估计偏乐观。CAG 的 500 个新 token 在 prefill 时要与 10 万个缓存 token 计算注意力，注意力的计算量是 RAG 中 5500 个 token 的若干倍；decode 时每一步都要读 10 万个 token 的 KV，ITL 也会明显变长。前缀缓存省掉的是线性层和缓存部分自身的计算，不是注意力。上下文越长，这一点越明显。

### 1.2 成本

使用第三方 API 时，缓存的 token 按折扣价计费。原书表 7-2 列出了 2025 年底的价格（美元，每百万 token）：

| | GPT-5 | Gemini 2.5 Pro | Claude Sonnet 4 |
|---|---|---|---|
| 普通输入 | 1.25 | 1.25 | 3 |
| 缓存命中的输入 | 0.125 | 0.31，另加每小时 4.5 的存储费 | 0.3，写入缓存时另计 |

按 GPT-5 的价格计算：

$$
\text{RAG：} \frac{500 \times 0.125 + 5500 \times 1.25}{10^6} \approx 0.007 \text{ 美元}, \qquad
\text{CAG：} \frac{100{,}500 \times 0.125 + 500 \times 1.25}{10^6} \approx 0.013 \text{ 美元}
$$

尽管缓存的价格只有十分之一，CAG 每个请求的成本仍然是 RAG 的近两倍，因为缓存的 token 多了二十倍。

> [!NOTE]
> 原书的 CAG 算式写成了"100,500 regular input × 0.125"，标签与价格不一致；按它的本意，100,500 个是缓存的 token，500 个是普通输入，结果约为 0.013 美元，与原书相同。另外，Claude Sonnet 4 写入缓存（5 分钟有效）的价格是每百万 token 3.75 美元，是普通输入的 1.25 倍，原书写作 3.25。API 的价格经常调整，使用时以各家的官方价格页为准。

**自己部署时，账就不一样了。** 缓存的 KV 存在自己的 CPU 内存和 SSD 里，几乎不花额外的钱；省下的 prefill 计算则是实实在在的 GPU 时间。下面讨论怎样做到这一点。

## 2. 分层存储

原书的观点是：在长上下文的场景下，KV Cache 不再只是模型内部的实现细节，它要像模型权重和输入数据一样，被存储、调度、在设备之间移动和淘汰。原书图 7-21 给出了一台 AWS p5.48xlarge 上的存储层次：

| 层 | 容量（这台机器） | 到 GPU 的带宽（量级） |
|---|---|---|
| GPU 显存 | 8 × 80 GB | 直接访问 |
| CPU 内存 | 2 TB | 每张卡经 PCIe 5.0 约 50 GB/s |
| 本地 NVMe SSD | 8 × 3.84 TB | 几 GB/s |
| 远端存储（Redis、对象存储等） | 不限 | 取决于网络 |

越往下容量越大，速度越慢。原书说，把 KV 卸载到 CPU 内存，可缓存的空间约为显存的三倍；再加上 SSD，可以达到五十倍。

### 2.1 重算还是加载

把 KV 从下一层存储读回来，比重新 prefill 快吗？取决于前缀的长度和存储的带宽：

<div class="llm-widget" data-widget="kv-tier"></div>

两条规律：

- **前缀越长，越值得加载。** 重算的线性层部分与长度成正比，注意力部分与长度的平方成正比；加载的时间只与长度成正比。
- **KV 越小，越值得加载。** GQA、MLA、FP8 KV Cache 都让每个 token 的 KV 变小，加载更快，而重算的计算量不变。

### 2.2 LMCache

LMCache 是一个专门管理 KV Cache 的开源系统，可以作为 vLLM、SGLang 的插件使用。它提供：

- **分层卸载**：显存放不下的 KV 自动卸载到 CPU 内存、本地 SSD 或远端存储，命中时取回；
- **跨实例共享**：多个推理实例共用远端的 KV 存储，一个实例算过的前缀，其他实例也能用；
- **KV 压缩**：CacheGen 根据 KV 的数值分布把它编码成更紧凑的比特流，减少存储空间和传输时间；
- **非前缀的复用**：CacheBlend，见下文。

原书给出了在 vLLM 中启用 LMCache 的 CPU 卸载的方法：

```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=150.0        # CPU 内存上限，GB
vllm serve Qwen/Qwen3-14B --max-model-len 40960 \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'
```

LMCache 的环境变量和配置方式在各版本之间有过多次调整，使用时以它的文档为准。较新的 vLLM 也提供了不依赖 LMCache 的原生 KV 卸载，同样通过 KV connector 的机制接入。

### 2.3 原书的实验

原书在 Qwen3-14B 上对比了默认的 vLLM（只有显存中的前缀缓存）与开启 CPU 卸载的 LMCache：

- **冷启动**：所有请求都没有缓存时，LMCache 总是比默认的 vLLM 慢一些，这是管理 KV 的额外开销。原书的结论是，如果负载的前缀很短或者很随机，不应该开启 LMCache。
- **预热之后**：30 个不同前缀的请求依次发送。前 9 个在两种配置下都能命中显存中的缓存，表现相同；从第 10 个开始，默认 vLLM 的缓存已经被淘汰，延迟回到冷启动的 6 到 7 秒，而 LMCache 从 CPU 内存取回 KV，约 1 秒。
- **并发**：20 个不同的长前缀、5 个并发时，默认 vLLM 受限于显存中能缓存的前缀数，吞吐停在每秒约 4700 个 token；LMCache 的吞吐约为它的 16 倍。

## 3. 非前缀的复用：CacheBlend

前缀缓存只能复用**前缀**。RAG 的场景下，同样的几段文档以不同的顺序、和不同的问题组合出现，前缀不同，缓存就无法命中。

能不能把每段文档的 KV 分别缓存，用的时候直接拼起来？不行。原书图 7-23 解释了原因：自注意力中，后面的 token 要关注前面所有的 token。单独计算的一段文档的 KV，没有"看到"排在它前面的其他文档，直接拼接会丢失这些跨段的关系，结果与完整计算不同。

CacheBlend 的做法是：拼接各段缓存的 KV，再挑出一小部分 token（默认 15%，通过 `blend_recompute_ratios` 调节）重新计算，修复跨段的注意力关系。论文表明，这样能以很小的质量损失，省掉大部分的重算。

## 4. 显存之内的长上下文

上面讨论的是"把 KV 存在哪里"。单个请求的上下文本身就很长时（几十万到上百万 token），还要考虑怎样在显存中处理它：

- KV Cache 量化（2.5 节）和 MLA、滑动窗口等结构（3.1 节）减小每个 token 的 KV；
- 稀疏注意力只计算一部分 token 之间的注意力；
- Context Parallelism 把一个长序列切到多张卡上计算。

6.6 节会讨论后两者。

## 5. 实验

- **组件**：在 `kv-tier` 中选 Qwen3-8B 和 H100，找出从 CPU 内存加载开始快于重算的前缀长度。换成 Llama-2-7B（MHA，KV 大 4 倍），这个长度怎样变化？
- **GPU**：按原书的方法，用 `vllm bench serve` 发送 30 个不同的长前缀请求（每个约 8K token），先正序发一遍、再倒序发一遍，比较默认 vLLM 与开启 LMCache CPU 卸载的 TTFT。在倒序的那一遍，默认 vLLM 从第几个请求开始失去缓存？

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "用第三方 API 时，原书的例子中 CAG 每个请求的成本约是 RAG 的两倍，尽管缓存的价格只有十分之一。原因是：", "options": ["CAG 不能使用缓存", "CAG 每次都带着 10 万个缓存的 token，数量是 RAG 检索内容的约 20 倍，抵消了折扣", "RAG 的检索是免费的", "CAG 的输出更长"], "answer": 1, "explain": "0.125 × 100,500 已经超过了 1.25 × 5500。自己部署时情况不同：缓存的 KV 存在自己的内存和 SSD 中，几乎不花钱，省下的是 GPU 的 prefill 计算。"}, {"q": "长上下文的 CAG 命中了 10 万 token 的前缀缓存，只需 prefill 500 个新 token。下面哪个说法正确？", "options": ["TTFT 和 ITL 都与没有长上下文时相同", "新 token 仍要与 10 万个缓存 token 计算注意力，decode 每一步也要读 10 万 token 的 KV，TTFT 和 ITL 都会比短上下文时长", "前缀缓存让注意力的计算量变为零", "ITL 会变短"], "answer": 1, "explain": "前缀缓存省掉的是缓存部分自身的计算，新 token 对整个上下文的注意力仍然要算。上下文越长，这部分开销越大。"}, {"q": "什么情况下，从 CPU 内存加载 KV 比重新 prefill 更划算？", "options": ["前缀很短时", "前缀越长、每个 token 的 KV 越小（GQA、MLA、量化），加载相对重算越划算", "模型越小时", "永远不划算"], "answer": 1, "explain": "重算中的注意力部分随长度平方增长，加载只随长度线性增长；KV 越小，要搬运的数据越少，而重算的计算量不变。"}, {"q": "为什么不能把 RAG 中各段文档分别缓存的 KV 直接拼接起来使用？", "options": ["KV 的格式不兼容", "单独计算的一段文档的 KV 没有关注到排在它前面的其他文档，直接拼接会丢失跨段的注意力关系", "显存不够", "拼接太慢"], "answer": 1, "explain": "后面的 token 的 KV 依赖于它之前的全部内容。CacheBlend 拼接后重新计算约 15% 的 token，修复这些关系，以很小的质量损失省掉大部分重算。"}]'></div>

## 延伸阅读

- [LMCache 文档](https://docs.lmcache.ai/)。
- Liu et al., [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://arxiv.org/abs/2310.07240), SIGCOMM 2024。
- Yao et al., [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444), EuroSys 2025。
- Qin et al., [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079), FAST 2025。
