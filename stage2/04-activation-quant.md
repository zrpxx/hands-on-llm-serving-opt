# 2.4 激活量化：SmoothQuant 与 FP8

> 对应原书第 6 章 *Weight-only versus weight-and-activation quantization strategies*。SmoothQuant 等算法是书外内容。

前两节的 GPTQ 和 AWQ 只量化权重。计算时，权重先反量化回 BF16，再和 BF16 的激活相乘，矩阵乘法本身没有变快。在 decode 阶段这没有问题，因为瓶颈是读权重的带宽（0.7 节）。但在 prefill 阶段，或者 batch 很大的 decode，瓶颈变成了算力，只量化权重就帮不上忙了，反量化甚至会带来额外的开销。

要让矩阵乘法本身变快，激活也必须是低精度的，这样才能用上 INT8 或 FP8 的 Tensor Core。这就是 W8A8。这一节讨论激活为什么难量化，以及两种主流的解决办法：SmoothQuant 和 FP8。

## 1. W4A16 与 W8A8

原书表 6-3 比较了两种方案，这里稍作补充：

| | W4A16（GPTQ、AWQ） | W8A8（INT8 或 FP8） |
|---|---|---|
| 权重大小 | 约原来的 1/4 | 原来的 1/2 |
| 矩阵乘法 | 仍是 BF16，先反量化权重 | INT8 或 FP8 Tensor Core，峰值算力是 BF16 的 2 倍 |
| prefill（计算瓶颈） | 基本不变，可能略慢 | 变快 |
| 小 batch 的 decode（带宽瓶颈） | 明显变快 | 变快，但不如 W4A16 |
| 大 batch 的 decode | 收益随 batch 增大而减小 | 收益保持 |
| 适合 | 延迟敏感、并发低、显存紧张 | 吞吐优先、并发高、输入长 |

原书在 Qwen2.5-7B-Instruct 上做了对比实验：并发较低时，GPTQ W4A16 的延迟和吞吐最好；并发升高后，W4A16 的 TTFT 甚至比原始模型还差，而 FP8 W8A8 一直保持着优势。

## 2. 激活为什么难量化

权重的分布比较规整：近似钟形，各个通道的范围相差不大。激活则不同。LLM.int8() 的作者发现，参数量达到约 67 亿以后，模型的隐藏状态中会出现少数几个**离群特征**（outlier features）：

- 它们集中在极少数几个固定的维度上，OPT-6.7B 中只有约 6 个维度；
- 几乎所有层都在同样的维度上出现离群值，大约 75% 的 token 受影响；
- 幅度随模型增大而增大，13B 的模型中可以达到 60 左右，而其他维度的数值通常在 1 以内。

如果激活按张量量化，这几个离群值决定了 scale，其他维度只能用到很少几个等级，这和 2.1 节的组件中看到的问题一样。

那能不能按输入通道来量化激活，给离群通道单独的 scale？不行。矩阵乘法 $Y = XW^\top$ 中，输入通道是求和的那一维：

$$
Y_{ik} = \sum_j X_{ij} W_{kj}
$$

如果激活的 scale 随 $j$ 变化，就无法提到求和号外面，INT8 的乘加结果没法直接换算回浮点数。scale 只能放在求和之外的维度上：激活按行（每个 token 一个 scale），权重按行（每个输出通道一个 scale）。

$$
Y_{ik} = \underbrace{\Delta^X_i \, \Delta^W_k}_{\text{求和之外}} \sum_j \hat{X}_{ij} \hat{W}_{kj}
$$

按 token 量化激活有一定帮助，但离群通道出现在几乎每个 token 里，每个 token 的 scale 仍然被它们决定。

LLM.int8() 的办法是把离群维度挑出来单独用 FP16 计算，其余部分用 INT8。精度很好，但混合精度的计算比 FP16 还慢。SmoothQuant 换了一个思路。

## 3. SmoothQuant

和 AWQ 一样，SmoothQuant 利用了这个等式：

$$
Y = XW^\top = \left(X \,\text{diag}(s)^{-1}\right) \cdot \left(W \,\text{diag}(s)\right)^\top
$$

激活的第 $j$ 个通道除以 $s_j$，权重的第 $j$ 个输入通道乘以 $s_j$。离群通道的 $s_j$ 取得大一些，激活就被压平了，代价是权重对应的那一列被放大。权重本来很好量化，承受一些难度没有问题。$s_j$ 由迁移强度 $\alpha$ 控制：

$$
s_j = \frac{\max |X_j|^{\alpha}}{\max |W_j|^{1-\alpha}}
$$

$\alpha = 0.5$ 时，激活和权重第 $j$ 个通道的最大值被调到相同。$\alpha$ 越大，迁移到权重上的难度越多。下面的组件可以观察这个过程。

<div class="llm-widget" data-widget="smooth-migrate"></div>

可以看到，误差曲线是 U 形的：$\alpha$ 太小，激活仍然难量化；$\alpha$ 太大，权重变得难量化。论文中大多数模型取 0.5，GLM-130B 这类离群值更严重的模型取 0.75。

$\max |X_j|$ 来自校准数据，$s$ 在量化时就确定下来。和 AWQ 一样，$1/s$ 合并到前一个 RMSNorm 或线性层中，推理时没有额外开销。论文在 OPT-175B、BLOOM-176B 等模型上实现了几乎无损的 W8A8，推理最多快 1.56 倍，显存减半。

> [!TIP]
> SmoothQuant 与 AWQ 的公式很像，但目的不同。AWQ 放大显著通道的**权重**，让它们在 4 位量化下更精确，激活不量化；SmoothQuant 压小离群通道的**激活**，让激活能用 8 位量化，权重多承担一点难度。

## 4. FP8

FP8 W8A8 比 INT8 W8A8 简单得多，这也是它这两年成为主流的原因。

原书表 6-4 列出了两种 FP8 格式：E4M3 有 4 位指数、3 位尾数，最大值 448；E5M2 有 5 位指数、2 位尾数，范围更大但精度更低。推理一般用 E4M3，E5M2 主要用于训练中的梯度。

浮点格式的等级是按指数分布的，相对误差大致恒定。一个离群值即使比其他数值大几十倍，也只是占用了更高的几个指数，小的数值仍然有 3 位尾数的相对精度。所以 FP8 对离群值远比 INT8 宽容，很多模型直接按张量量化就能保持精度，不需要 SmoothQuant 这样的预处理。

FP8 的 scale 有几种常见的粒度：

| 方案 | 权重 | 激活 | 例子 |
|---|---|---|---|
| 按张量，静态 | 每个矩阵一个 scale | 校准时确定一个 scale | 早期的 FP8 checkpoint |
| 按张量，动态 | 每个矩阵一个 scale | 每次前向时计算 | vLLM 的 `--quantization fp8` 在线量化 |
| 按通道，动态 | 每个输出通道一个 scale | 每个 token 一个，动态计算 | LLM Compressor 的 `FP8_DYNAMIC` |
| 按块 | 每 128×128 一个 scale | 每个 token 的每 128 个元素一个 | DeepSeek-V3、Qwen3 的官方 FP8 模型 |

动态的 scale 需要在每次前向时先求一遍最大值，多一点开销，但不需要校准数据，也不怕实际输入的范围和校准时不同。精度通常比静态的好。

> [!NOTE]
> 原书说 NVIDIA 的显卡中只有 Hopper 和 Blackwell 支持 FP8。实际上 Ada 架构（L4、L40S、RTX 4090）也有 FP8 Tensor Core。在 A100 等 Ampere 显卡上，vLLM 会把 FP8 模型当作只量化权重的 W8A16 来运行，能省显存和带宽，但矩阵乘法仍然是 BF16。

## 5. 在实际中使用

最简单的做法不需要任何准备，vLLM 加载 BF16 模型时在线量化：

```bash
vllm serve Qwen/Qwen3-8B --quantization fp8
```

要更好的精度，可以离线生成按通道的 FP8 模型，同样不需要校准数据：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

oneshot(
    model="Qwen/Qwen3-8B",
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="Qwen3-8B-FP8-Dynamic",
)
```

INT8 W8A8 要配合 SmoothQuant，并且需要校准数据：

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
# 之后的 oneshot 调用与 2.2 节相同，需要指定 dataset
```

怎样选：

- 有 Hopper、Ada 或 Blackwell 的卡，首选 FP8，几乎无损，流程最简单。
- 只有 Ampere 的卡又需要提高计算吞吐，用 INT8 W8A8 加 SmoothQuant。
- 主要问题是显存或者小 batch 的延迟，用 W4A16。

## 6. 更低的位数

W4A4 能再把算力翻一倍，但 4 位的激活非常难量化，离群值的问题比 8 位严重得多。目前主要有两条路：

- **旋转**。QuaRot、SpinQuant 等方法在激活和权重两边分别乘以一个正交矩阵 $R$ 和 $R^\top$，结果不变。旋转把离群通道的能量分散到所有通道上，分布变得均匀。
- **微缩放格式**。Blackwell 支持的 MXFP4、NVFP4 每 32 个或 16 个元素共用一个 scale，离群值只影响所在的一小块。2.7 节会介绍使用 MXFP4 发布的 gpt-oss，6.2 节会详细讨论这两种格式。

## 7. 实验

`tools/outlier_layer.py` 的后半部分比较了不同的 W8A8 方案：

```bash
cd code/project2
QUANTLAB_IMPL=solution python tools/outlier_layer.py --outlier-scale 1
QUANTLAB_IMPL=solution python tools/outlier_layer.py --outlier-scale 20
```

输出的相对均方误差：

| | INT8 激活按张量 | INT8 激活按 token | SmoothQuant（α = 0.5）+ 按张量 | FP8 激活按 token |
|---|---|---|---|---|
| 没有离群通道 | 2.00e-4 | 1.09e-4 | 1.77e-4 | 1.39e-3 |
| 离群通道放大 20 倍 | 5.62e-3 | 9.89e-4 | 3.87e-4 | 1.12e-3 |

几个值得注意的地方：

- 离群值让按张量的 INT8 误差增大了约 28 倍，按 token 的增大了约 9 倍。
- SmoothQuant 之后，即使激活按张量量化，误差也比按 token 的还小。
- FP8 的误差几乎不受离群值影响，但没有离群值时反而比 INT8 大。原因是 E4M3 只有 3 位尾数，而分布规整时 INT8 的 256 个等级能充分用上。FP8 的优势在于稳健，而不是在所有情况下都更精确。

在小模型上（BF16 困惑度 37.98），各种 W8A8 方案的困惑度都在 38.00 到 38.13 之间，差别很小。同样，这个模型没有离群特征。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "并发很高、输入很长的离线批处理任务中，为什么 W8A8 通常比 W4A16 更合适？", "options": ["W8A8 的模型更小", "这时瓶颈是算力，W8A8 能用上 8 位 Tensor Core，W4A16 的矩阵乘法仍是 BF16", "W4A16 不支持长输入", "W8A8 不需要 KV Cache"], "answer": 1, "explain": "W4A16 只减少了读权重的字节数，适合带宽瓶颈的小 batch decode。prefill 和大 batch 时是计算瓶颈，只有激活也量化才能让矩阵乘法变快。"}, {"q": "激活中有几个离群的输入通道。为什么不能简单地给激活的每个输入通道一个单独的 scale？", "options": ["scale 太多，占用显存", "输入通道是矩阵乘法中求和的那一维，scale 无法提到求和之外，整数乘加的结果没法直接换算", "离群通道的位置每个 token 都不同", "INT8 不支持负数"], "answer": 1, "explain": "低精度 GEMM 要求 scale 只依赖于输出的行和列：激活每个 token 一个、权重每个输出通道一个。这正是 SmoothQuant 要把通道间的差异先通过等价变换挪走的原因。"}, {"q": "SmoothQuant 的迁移强度 &#36;\\alpha&#36; 取 1 时会发生什么？", "options": ["等同于不做平滑", "激活的每个通道被完全压平，离群值的难度全部转移到权重上，权重可能变得难以量化", "权重和激活都变成 FP16", "误差一定最小"], "answer": 1, "explain": "&#36;\\alpha = 1&#36; 时 &#36;s_j = \\max|X_j|&#36;，激活每个通道的最大值都变成 1，但权重对应的列被放大了 &#36;\\max|X_j|&#36; 倍。误差随 &#36;\\alpha&#36; 呈 U 形，通常取 0.5 左右。"}, {"q": "在本节的模拟中，没有离群值时 FP8 的误差反而比 INT8 按 token 量化更大。原因是：", "options": ["FP8 的实现有错误", "E4M3 只有 3 位尾数，分布规整时 INT8 的 256 个均匀等级更精确；FP8 的优势在于对离群值不敏感", "FP8 的最大值只有 448，发生了截断", "FP8 只能按张量量化"], "answer": 1, "explain": "FP8 的相对误差约为 &#36;2^{-4}&#36;，与数值大小无关；INT8 在范围被充分利用时更精确，但一个离群值就会让大多数数值失去精度。所以 FP8 常常不需要 SmoothQuant 这类预处理。"}]'></div>

## 延伸阅读

- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), NeurIPS 2022，以及作者的博客 [LLM.int8() and Emergent Features](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)。
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICML 2023。
- Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), 2022。
- Ashkboos et al., [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456), NeurIPS 2024。
