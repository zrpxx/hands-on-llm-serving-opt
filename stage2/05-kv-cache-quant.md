# 2.5 KV Cache 量化

> 对应原书第 6 章 *KV cache quantization and attention quantization*。KIVI、KVQuant 等低比特方法是书外内容。

前面几节量化的都是线性层的权重和激活。KV Cache 是另一块大头：并发高、上下文长的时候，它占用的显存可以超过模型权重本身。这一节讨论 KV Cache 量化能带来什么、FP8 KV Cache 怎样使用，以及量化到 4 位、2 位时 K 和 V 应该分别怎样处理。

## 1. 为什么要量化 KV Cache

以 Qwen3-8B 为例。它有 36 层，8 个 KV 头，每个头 128 维，BF16 下每个 token 的 KV Cache 是：

$$
2 \times 36 \times 8 \times 128 \times 2 \text{ 字节} = 144 \text{ KiB}
$$

32 个并发请求、每个 8K token，共约 26 万个 token，KV Cache 约 36 GiB，是权重（约 16 GB）的两倍多。量化 KV Cache 有两方面的收益：

- **容量**。FP8 让同样的显存多存一倍的 token，能支持更大的 batch 或更长的上下文，前缀缓存（1.4 节）也能保留更多内容。
- **带宽**。decode 阶段每一步都要把所有请求的 KV Cache 从显存读一遍。上面的例子中，每一步读 KV 的字节数已经超过了读权重的字节数。KV 的字节数减半，注意力部分的读取量也减半。

原书认为，KV Cache 量化通常不会明显降低延迟，因为注意力计算仍是高精度的，量化的 KV 读出来以后还要反量化。这个说法需要区分情况：如果注意力的 kernel 直接读取低精度的 KV，在寄存器里反量化，那么显存读取量确实减少了，长上下文的 decode 会变快；如果先把整个 KV 反量化到一块 BF16 的缓冲区再计算，收益就只剩容量了。vLLM 和 SGLang 的注意力 kernel 都属于前一种。

## 2. FP8 KV Cache

FP8 是目前最常用的 KV Cache 量化方式，精度损失通常很小：

```bash
vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8
```

启动日志中 KV Cache 能容纳的 token 数会变成原来的两倍左右。使用时有几个细节：

- **scale 从哪里来**。不做校准时，vLLM 把 K 和 V 的 scale 都设为 1.0，也就是直接把 BF16 的数值转换成 FP8。K、V 的数值一般远小于 E4M3 的最大值 448，所以通常没有问题。更稳妥的做法是用 LLM Compressor 在校准数据上统计每层 K、V 的范围，把 scale 存进 checkpoint，vLLM 加载时会自动使用。
- **粒度**。默认每层的 K、V 各一个 scale。部分注意力后端支持按 KV 头的 scale，需要校准得到。
- **注意力用什么精度算**。在 Hopper 上使用 FlashAttention 3 后端时，Q 也被量化到 FP8，注意力的矩阵乘法直接在 FP8 上进行；其他情况下，K、V 读出后转换回 BF16 再计算。
- **个别层更敏感**。比如滑动窗口注意力层，可以用 `--kv-cache-dtype-skip-layers` 让它们保持原精度。

校准 KV scale 的 recipe 大致如下，可以和权重量化放在同一个 recipe 里：

```python
recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      kv_cache_scheme:
        num_bits: 8
        type: float
        strategy: tensor
        dynamic: false
        symmetric: true
"""
```

原书推荐的顺序是：先做权重或权重加激活的量化，比如 FP8 W8A8；遇到长上下文显存不够，或者需要更高的 decode 吞吐时，再加上 FP8 KV Cache。

## 3. 更低的位数：K 和 V 不一样

FP8 之后，再往下是 INT4、INT2。到这个位数，量化的粒度就变得很重要了。KIVI 的作者统计了 KV Cache 的数值分布，发现 K 和 V 的特点完全不同：

- **K 有固定的离群通道**。少数几个通道在几乎所有 token 上数值都很大，和 2.4 节激活中的离群特征类似。如果按 token 量化，每个 token 的 scale 都被这几个通道决定。应该**按通道**量化，每个通道一个 scale。
- **V 没有明显的离群通道**。注意力的输出是各个 token 的 V 按注意力权重加权求和。如果按 token 量化，每个 token 的误差局限在它自己的那一行，权重小的 token 误差再大也影响不大。应该**按 token** 量化。

下面的组件模拟了这两种分布，可以比较不同方向的量化误差。

<div class="llm-widget" data-widget="kv-quant-axis"></div>

4 位时，K 按通道量化的误差比按 token 小约 20 倍；V 则相反，按 token 更好。到 2 位时，K 按 token 量化的相对均方误差接近 0.5，几乎已经不可用。

按通道量化 K 有一个工程上的问题：KV Cache 是逐个 token 追加的，而按通道的 scale 要在多个 token 上统计。KIVI 的做法是把 token 按每 32 个分成一组，一组凑满了再一起量化，最近还没凑满的 token 以及一段固定长度的最新 token 保持 FP16。论文在 2 位下精度几乎无损，峰值显存降低了 2.6 倍（含权重），batch 可以增大到原来的 4 倍，吞吐提高 2.35 到 3.47 倍。

KVQuant 进一步发现，K 的离群通道在 RoPE 之前更明显、更规整。RoPE 把每两个通道配成一对做旋转，会把一个通道的离群值分摊到它配对的通道上，而且旋转角度随位置变化。所以 KVQuant 在 RoPE 之前量化 K，计算注意力时再对反量化后的 K 应用 RoPE。

> [!NOTE]
> 在主流的服务引擎中，最成熟、最常用的仍是 FP8 KV Cache。INT4 等更低位数的支持在各个引擎之间差别较大，使用前要查看所用版本的文档。LMDeploy 较早支持 INT4、INT8 的 KV Cache。

## 4. 其他压缩 KV Cache 的办法

量化只是减小 KV Cache 的一种手段。还有两类方法从别的方向入手：

- **少存一些 token**。StreamingLLM 只保留开头几个 token 和最近的一段窗口；H2O、SnapKV 等方法按注意力分数淘汰不重要的 token。这些方法会改变模型看到的上下文，对需要精确回忆长文中某个细节的任务影响较大。
- **改变模型结构**。GQA 让多个查询头共享一组 KV（0.4 节）；DeepSeek 的 MLA 把 KV 压缩成一个低维的隐向量，3.1 节会介绍。这些需要在训练时就采用，不能用于现有的模型。

## 5. 实验

`add_kv_quant` 在每层的 K（Qwen3 的 `k_norm` 输出，即 RoPE 之前）和 V 上挂钩子做伪量化，V 固定按 token，K 可以选择方向：

```bash
cd code/project2
QUANTLAB_IMPL=solution python run.py --model /tmp/tiny-lm --data /tmp/qcorpus --kv-bits 4 --k-axis channel
QUANTLAB_IMPL=solution python run.py --model /tmp/tiny-lm --data /tmp/qcorpus --kv-bits 4 --k-axis token
```

小模型上的结果（不量化时困惑度 37.98）：

| KV 位数 | K 按 token | K 按通道 |
|---|---|---|
| 8 | 37.97 | |
| 4 | 38.17 | 38.20 |
| 3 | 39.11 | 39.16 |
| 2 | 47.38 | 46.04 |

8 位几乎无损。4 位、3 位时 K 的两种方向差别不大，到 2 位时按通道才显出优势。原因和前几节一样：统计这个小模型 K 的各通道最大值，最大的通道也只是中位数的 2 到 3 倍，没有真实大模型中那样突出的离群通道。在 GPU 上，可以用同样的钩子对 Qwen3-0.6B 或 Qwen3-8B 做这个实验，再与 vLLM 的 FP8 KV Cache 对比。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "Qwen3-8B 有 36 层、8 个 KV 头、每头 128 维。BF16 下每个 token 的 KV Cache 约多大？", "options": ["约 36 KiB", "约 72 KiB", "约 144 KiB", "约 288 KiB"], "answer": 2, "explain": "K 和 V 各一份：&#36;2 \\times 36 \\times 8 \\times 128 \\times 2&#36; 字节，约 144 KiB。FP8 时减半为 72 KiB，同样的显存能多存一倍的 token。"}, {"q": "长上下文、高并发的 decode 中，FP8 KV Cache 除了节省显存，还可能带来什么收益？", "options": ["没有其他收益，注意力仍是 BF16 计算", "如果注意力 kernel 直接读取 FP8 的 KV，每一步从显存读取的字节数减半，decode 变快", "让 prefill 的计算量减半", "让模型权重也变成 FP8"], "answer": 1, "explain": "decode 阶段注意力是带宽瓶颈，每一步都要读一遍所有的 KV。只要 kernel 读的是低精度数据、在寄存器里反量化，读取量就减少了，即使计算本身仍用 BF16。"}, {"q": "KIVI 为什么让 K 按通道量化、V 按 token 量化？", "options": ["K 的维度比 V 小", "K 有少数在所有 token 上都很大的固定通道；V 没有，而且注意力输出是各 token 的 V 的加权和，按 token 量化让误差局限在各自的行", "这样实现起来最简单", "V 需要更多的位数"], "answer": 1, "explain": "K 按 token 量化时，每个 token 的 scale 都被离群通道决定；按通道就能把离群通道隔离开。V 则相反，按 token 量化更好。"}, {"q": "按通道量化 K 时，推理过程中会遇到什么工程问题？KIVI 怎样处理？", "options": ["无法使用 GPU，只能在 CPU 上量化", "KV 逐个 token 追加，按通道的 scale 需要多个 token 统计；KIVI 把 token 分组，凑满一组再量化，最近的 token 保持 FP16", "需要在每一步重新量化全部历史 token", "需要重新训练模型"], "answer": 1, "explain": "一个通道的 scale 要在一组 token 上统计。KIVI 每 32 个 token 一组，尚未凑满的和最近一段 token 用全精度保存，这一小部分的显存开销可以忽略。"}]'></div>

## 延伸阅读

- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), ICML 2024。
- Hooper et al., [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079), NeurIPS 2024。
- [vLLM 文档：Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)。
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), ICLR 2024。
