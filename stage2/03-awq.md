# 2.3 AWQ

> 书外内容为主。原书第 9 章的第 6 步用 Qwen3-14B-AWQ 做了量化前后的性能对比，算法来自 Lin et al., AWQ（MLSys 2024）。

AWQ（Activation-aware Weight Quantization）是另一种常用的 4 位权重量化方法。它和 GPTQ 的出发点相同：权重误差对输出的影响取决于输入。但做法完全不同。GPTQ 通过调整其他权重来补偿误差；AWQ 则在量化之前，把重要的输入通道放大，让这些通道上的权重在量化时获得更高的相对精度。

这一节先看 AWQ 的出发点：少数权重特别重要，而重要与否由激活决定。然后推导为什么放大能减小误差，以及怎样搜索放大的倍数、怎样做到推理时没有额外开销。

## 1. 少数权重特别重要

AWQ 论文做了一个实验：把 OPT 模型量化到 3 位，但保留一小部分权重为 FP16。结果是：

- 如果按**激活的平均幅度**挑选最重要的 1% 输入通道，保留这些通道对应的权重为 FP16，困惑度几乎回到全精度的水平；
- 如果按**权重本身的大小**挑选，效果和随机挑选差不多。

原因和 GPTQ 那一节的分析一致：一个权重的误差在输出中会乘以对应的激活。激活大的通道，权重误差被放大得最多。大模型的激活里普遍存在少数几个幅度特别大的通道（2.4 节会详细讨论），这些通道上的权重就是"显著权重"。

但保留一部分权重为 FP16，意味着同一个矩阵里混合两种精度，kernel 很难高效地实现。AWQ 要找一个不需要混合精度、又能保护显著权重的办法。

## 2. 放大能减小误差

线性层的一个输出可以写成 $y = \sum_j w_j x_j$。把第 $j$ 个输入通道的权重乘以 $s$，激活除以 $s$，结果不变：

$$
w_j x_j = (s \cdot w_j) \cdot (x_j / s)
$$

现在只对放大后的权重做量化。设一组权重的量化间隔是 $\Delta = \max|w| / q_{\max}$，量化误差是 $\Delta \cdot \epsilon$，其中 $\epsilon$ 是舍入带来的误差，平均大小与权重本身无关。

量化 $s \cdot w_j$ 之后，这一项在输出中的误差是：

$$
\Delta' \cdot \epsilon \cdot \frac{x_j}{s}
$$

其中 $\Delta'$ 是放大后这一组的量化间隔。只要 $s \cdot w_j$ 不是这一组里最大的那个权重，组的最大值就不变，$\Delta' = \Delta$，于是这一项的误差缩小为原来的 $1/s$。

但 $s$ 不能无限大。放大到一定程度，$s \cdot w_j$ 会成为组内的最大值，$\Delta'$ 随之变大，组内**其他**权重的误差都会增大。所以需要在保护显著权重和不损害其他权重之间取一个平衡。

## 3. 搜索缩放因子

AWQ 让每个输入通道的缩放因子取决于这个通道激活的平均幅度：

$$
s_j = \overline{|x_j|}^{\,\alpha}, \quad \alpha \in [0, 1]
$$

$\alpha = 0$ 时所有 $s_j = 1$，就是普通的 RTN；$\alpha$ 越大，激活大的通道放得越大。AWQ 在 $[0, 1]$ 上取 20 个点做网格搜索，选使这一层在校准数据上输出误差最小的 $\alpha$：

$$
\alpha^* = \arg\min_\alpha \left\| Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X - W X \right\|^2
$$

因为 $\alpha = 0$ 也在搜索范围内，AWQ 的结果在校准数据上不会比 RTN 差。除了缩放，AWQ 还会再搜索一次每组的截断范围：把最大值适当收缩，用少量截断误差换取更小的舍入误差。

下面的组件用一个模拟的线性层演示搜索过程。输入中有几个通道的幅度比其他通道大，可以调节它们的倍数，观察误差曲线的形状和最优的 $\alpha$。

<div class="llm-widget" data-widget="awq-scale"></div>

可以试试：把离群倍数调到 1（没有显著通道），误差曲线几乎是平的，AWQ 和 RTN 一样；倍数越大，曲线越深，最优的 $\alpha$ 也越明显。和"保留显著通道为 FP16"那条线比较，AWQ 不用混合精度，也能拿到其中相当一部分收益。

## 4. 推理时没有额外开销

缩放之后，激活要除以 $s$。如果在推理时单独做这一步除法，就多了一次读写。AWQ 把它合并到前一个算子里：

| 被缩放的线性层 | 它的输入来自 | $1/s$ 合并到 |
|---|---|---|
| q_proj、k_proj、v_proj | 注意力前的 RMSNorm | RMSNorm 的缩放权重 |
| o_proj | 注意力的输出，即 V 的加权和 | v_proj 的输出通道 |
| gate_proj、up_proj | 前馈网络前的 RMSNorm | RMSNorm 的缩放权重 |
| down_proj | gate 与 up 的逐元素乘积 | up_proj 的输出通道 |

共享同一个输入的几个线性层（比如 q、k、v）必须用同一组 $s$，所以 AWQ 会把它们放在一起搜索。合并之后，模型的结构和普通的 4 位模型完全相同，推理时只需要一个普通的 W4A16 kernel。

## 5. AWQ 与 GPTQ

| | GPTQ | AWQ |
|---|---|---|
| 核心操作 | 逐列量化，把误差补偿到其他列 | 先按激活缩放输入通道，再普通地量化 |
| 用到的校准信息 | 输入的二阶统计量 $XX^\top$ | 每个通道激活的平均幅度 |
| 量化速度 | 较快 | 更快，只做网格搜索 |
| 对校准数据的依赖 | 较强，补偿可能对校准数据过拟合 | 较弱，只用到粗略的统计量 |
| 适合 | 3 位等更低的位数时优势明显 | 指令微调模型、多模态模型等校准数据不易与实际分布一致的场景 |

两者并不互斥。可以先做 AWQ 的缩放，再用 GPTQ 量化缩放后的权重。实际使用时，两者在 4 位、每组 128 的设置下精度通常相当，选哪个更多取决于工具链和现成的模型：很多模型的发布方会同时提供 GPTQ 和 AWQ 版本，比如 Qwen 官方就发布了 Qwen3 各尺寸的 AWQ 版本。

## 6. 在实际中使用

AWQ 原来的实现 AutoAWQ 已经停止维护，现在推荐用 LLM Compressor 的 `AWQModifier`：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

oneshot(
    model="Qwen/Qwen3-8B",
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=AWQModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"]),
    num_calibration_samples=256,
    max_seq_length=2048,
    output_dir="Qwen3-8B-W4A16-AWQ",
)
```

vLLM 能识别 AWQ 格式的 checkpoint，在支持的显卡上自动使用 Marlin kernel（显示为 `awq_marlin`）。原书第 9 章的启动命令写了 `--quantization awq`，在较新的 vLLM 中这会强制使用一个较慢的老 kernel，通常不写这个参数，让 vLLM 自动选择更好。

原书第 9 章在一张 L40S 上对比了 Qwen3-14B 和它的 AWQ 版本：权重从 27.5 GiB 降到 9.4 GiB，KV Cache 能容纳的 token 数从约 7.2 万增加到约 19.1 万；在 ShareGPT 负载下，总吞吐从每秒 474 个 token 提高到 1280 个，约 2.7 倍，平均 TTFT 降低了约 42%。

> [!NOTE]
> 原书在这里把 AWQ 的收益解释为"减少了 CPU 和 GPU 之间的数据传输"。实际上，推理时权重一直在显存里，减少的是每一步从显存读到计算单元的字节数（0.7 节），以及权重占用的显存，后者让 KV Cache 有了更多空间。

## 7. 实验

### 7.1 模拟的离群通道

`tools/outlier_layer.py` 构造了一个线性层，其中 8 个输入通道的幅度被放大，然后比较 RTN、GPTQ 和 AWQ：

```bash
cd code/project2
QUANTLAB_IMPL=solution python tools/outlier_layer.py --outlier-scale 1
QUANTLAB_IMPL=solution python tools/outlier_layer.py --outlier-scale 20
```

一次运行的结果（输出的相对均方误差）：

| | RTN | GPTQ | AWQ |
|---|---|---|---|
| 没有离群通道，W4 | 1.38e-2 | 6.61e-3 | 1.38e-2 |
| 离群通道放大 20 倍，W4 | 1.29e-2 | 2.35e-3 | 7.05e-3 |
| 离群通道放大 20 倍，W3 | 7.08e-2 | 1.53e-2 | 3.85e-2 |

没有离群通道时，AWQ 找不到可以保护的通道，结果和 RTN 一样；有离群通道时，误差降到约一半。这个模拟中 GPTQ 更好，是因为输入通道之间有很强的相关性，正适合 GPTQ 的误差补偿。

### 7.2 小模型

在 2.2 节的小模型上，AWQ 的困惑度和 RTN 几乎相同：3 位、每组 128 时分别是 39.76 和 39.71。这个小模型的激活里没有明显的离群通道，AWQ 没有发挥的空间。离群通道是参数量达到几十亿以后才普遍出现的现象，要观察 AWQ 的效果，需要在 GPU 上用 Qwen3-0.6B 以上的模型做同样的实验。这也提醒我们：量化方法的效果和模型有关，在一个模型上得到的结论，不一定能推广到另一个模型。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "AWQ 论文发现，在 3 位量化时保留 1% 的权重为 FP16 就能恢复大部分精度。这 1% 应该按什么挑选？", "options": ["权重绝对值最大的 1%", "对应输入通道的激活平均幅度最大的 1%", "随机挑选", "梯度最大的 1%"], "answer": 1, "explain": "权重误差在输出中要乘以对应的激活，激活大的通道上，同样的权重误差造成的输出误差最大。论文中按权重大小挑选的效果和随机挑选差不多。"}, {"q": "把第 &#36;j&#36; 个输入通道的权重乘以 &#36;s > 1&#36;、激活除以 &#36;s&#36; 再量化，为什么这个通道的误差会变小？", "options": ["放大后的权重可以用更少的位数表示", "只要组内最大值不变，量化间隔不变，舍入误差在输出中被除以 &#36;s&#36;", "放大后舍入误差变成了零", "激活除以 &#36;s&#36; 后不再需要量化"], "answer": 1, "explain": "舍入误差的大小由量化间隔决定，与权重本身无关。组的最大值不变时间隔不变，而这一项乘的是 &#36;x_j / s&#36;，误差缩小为 &#36;1/s&#36;。&#36;s&#36; 太大时被放大的权重成为组内最大值，间隔变大，其他权重的误差随之增大。"}, {"q": "AWQ 推理时，激活需要除以的 &#36;1/s&#36; 是怎样处理的？", "options": ["在每个线性层前插入一个逐元素除法的算子", "合并到前一个算子里，例如 RMSNorm 的缩放权重或前一个线性层的输出通道", "保存在 checkpoint 中，由 kernel 在计算时读取", "忽略不计，因为 &#36;s&#36; 接近 1"], "answer": 1, "explain": "q、k、v 和 gate、up 的输入来自 RMSNorm，把 &#36;1/s&#36; 乘进 RMSNorm 的权重即可；o_proj 和 down_proj 的缩放分别并入 v_proj 和 up_proj 的输出通道。合并之后模型结构与普通的 W4A16 模型相同。"}, {"q": "在本节的小模型上，AWQ 的困惑度和 RTN 几乎相同。最合理的解释是：", "options": ["AWQ 的实现有错误", "小模型的激活里没有明显的离群通道，缩放没有可以保护的对象", "3 位太低，任何方法都无效", "校准数据太多"], "answer": 1, "explain": "AWQ 的收益来自激活中少数幅度特别大的通道，没有这样的通道时，最优的缩放接近不缩放。离群通道主要出现在参数量较大的模型里，所以要在真实模型上评估量化方法。"}]'></div>

## 延伸阅读

- Lin et al., [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024。
- 原书第 9 章 *Step 6: Benchmark the Quantized Qwen3 Model with vLLM*。
- [LLM Compressor 的 AWQ 示例](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq)。
