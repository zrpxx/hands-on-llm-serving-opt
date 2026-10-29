# 2.2 GPTQ

> 书外内容为主。原书第 6 章 *Quantize a model yourself* 给出了用 GPTQ 量化模型的示例代码，算法本身来自 Frantar et al., GPTQ（ICLR 2023）。

最简单的权重量化是把每个权重四舍五入到最近的等级，叫 RTN（round-to-nearest）。8 位时 RTN 就足够好，到了 4 位，尤其是 3 位，误差就明显了。GPTQ 是目前使用最广的 4 位权重量化方法之一。它的思路是：量化一个权重带来的误差，可以通过调整同一行里还没量化的其他权重来补偿。

这一节先说明 RTN 的问题出在哪里，再推导 GPTQ 的更新公式，然后介绍让它能在大模型上跑得动的几个技巧，最后用项目 2 的代码比较 GPTQ 和 RTN。

## 1. RTN 忽略了输入

RTN 让每个权重的量化误差都尽可能小。但我们真正关心的不是权重本身的误差，而是这一层**输出**的误差。对一个线性层 $y = Wx$，量化后输出的误差是：

$$
\| W X - \hat{W} X \|_F^2
$$

其中 $X$ 是这一层的若干个输入（来自校准数据），$\hat{W}$ 是量化后的权重。

权重的误差对输出的影响并不相同：

- 如果某个输入通道的激活一直很大，这个通道上的权重误差就会被放大；
- 如果两个输入通道高度相关，一个权重的误差可以由另一个权重反向调整来抵消。

RTN 看不到这些信息，GPTQ 则利用了它们。

## 2. 量化一个权重，补偿其他权重

$W$ 的每一行互不影响（每一行对应一个输出），所以可以一行一行地考虑。对一行权重 $w$，输出误差是 $(w - \hat{w}) X X^\top (w - \hat{w})^\top$，其中出现的矩阵

$$
H = 2 X X^\top
$$

就是误差关于 $w$ 的二阶导数（Hessian）。它的大小是 $d_{\text{in}} \times d_{\text{in}}$，只取决于输入，所以同一层的所有行共用一个 $H$。

GPTQ 来自更早的 OBS（Optimal Brain Surgeon）和 OBQ 方法。它们回答了这样一个问题：如果把第 $q$ 个权重固定为量化后的值 $\text{quant}(w_q)$，其他权重应该怎样调整，才能让输出误差最小？答案是：

$$
\delta = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}
$$

也就是把第 $q$ 个权重的量化误差，按 $H^{-1}$ 的第 $q$ 列，分摊到其他权重上。直观地说，和第 $q$ 个输入相关性强的那些权重，会做较大的反向调整，抵消它的误差。

于是可以这样量化一行：从第一个权重开始，量化它，按上式更新剩下的权重；再量化第二个，更新剩下的；直到最后一个。每量化一个权重，还要从 $H^{-1}$ 中去掉对应的行和列，供下一步使用。

下面的组件对一行 16 个权重演示这个过程。输入的前 8 个通道两两高度相关。逐步执行，看每量化一列，后面还没量化的权重怎样被调整，以及输出误差怎样比 RTN 小。

<div class="llm-widget" data-widget="gptq-steps"></div>

## 3. 让它在大模型上可行

OBQ 每一步都选误差最小的那个权重先量化，每一行的顺序都不同，要分别维护 $H^{-1}$，计算量是 $O(d_{\text{row}} \cdot d_{\text{in}}^3)$，对几十亿参数的模型完全不可行。GPTQ 做了三处改动。

**所有行用相同的顺序。** 作者发现，按固定的顺序（比如从左到右）量化，结果并不比贪心地挑选差多少，大模型上尤其如此。所有行用同样的顺序，就只需要维护一份 $H^{-1}$，复杂度降到 $O(d_{\text{in}}^3)$。

**按块延迟更新。** 每量化一列，都要更新后面所有的列，这是一次很小的矩阵运算，GPU 利用率很低。GPTQ 把列分成大小为 128 的块：块内逐列更新，块与块之间累积起来一次更新，变成一次大的矩阵乘。

**用 Cholesky 分解。** 反复从 $H^{-1}$ 中去掉行和列，在大模型上会积累数值误差，甚至让矩阵不再正定。GPTQ 观察到所需的 $H^{-1}$ 的那些行，正好可以从 $H^{-1}$ 的 Cholesky 分解中直接读出，一次分解即可，数值上也更稳定。此外，计算前给 $H$ 的对角线加上一个小量（通常是对角线平均值的 1%，叫 dampening），防止矩阵奇异。

有了这些改动，论文中量化 1750 亿参数的 OPT-175B 只需要单张 A100 约 4 小时。

完整的算法（项目 2 的 `gptq.py` 就是这样实现的）：

```python
H += damp * mean(diag(H)) * I
U = cholesky(inverse(H), upper=True)        # 需要的 H⁻¹ 的行都在 U 里
for 每个大小为 128 的块 [i1, i2):
    for i in 块内的每一列:
        q = quant(W[:, i])                   # 用这一组的 scale 量化第 i 列
        err = (W[:, i] - q) / U[i, i]
        W[:, i+1:i2] -= err ⊗ U[i, i+1:i2]   # 补偿块内剩下的列
        记录 err
    W[:, i2:] -= Err_块 @ U[i1:i2, i2:]      # 一次性补偿之后的所有列
```

## 4. 两个常用选项

**分组。** 4 位量化通常每 128 个权重一组。GPTQ 的块大小和组大小对齐时，每进入一组，就用这一组**当前的值**（已经被前面的误差补偿修改过）计算 scale。

**按激活排序（act-order，也叫 desc_act）。** 先量化 $H$ 对角元大的列，也就是激活能量大、更重要的输入通道。这些列先量化，它们的误差可以由后面更多的列来补偿。对一些模型，尤其是 3 位量化时，效果有明显改善。代价是：和分组同时使用时，列的顺序被打乱，同一组的权重不再连续，推理时需要额外的索引（`g_idx`），部分 kernel 的速度会变慢。后来的实现提供了 static groups 选项，先按原顺序确定分组，再按激活顺序量化，避开了这个问题。

## 5. 校准数据

GPTQ 需要一批校准数据来计算 $H$。论文用的是 128 段、每段 2048 个 token 的 C4 文本。几点经验：

- 数量不需要很多，几百段就够了。
- 内容最好接近实际的使用场景。一个主要用于中文客服的模型，用英文网页做校准，效果可能不如用中文对话。
- 量化是逐层进行的：先用校准数据算第一层的 $H$，量化第一层；再让校准数据经过量化后的第一层，算第二层的 $H$。这样后面的层能看到前面层量化带来的变化。

## 6. 在实际中使用

量化工具：

- [LLM Compressor](https://github.com/vllm-project/llm-compressor)（vLLM 项目维护）：`GPTQModifier`，量化结果可以直接被 vLLM 加载。
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)：AutoGPTQ 的后继项目。
- Hugging Face Transformers 的 `GPTQConfig`：原书给的示例用的就是它，底层调用 GPTQModel。

用 LLM Compressor 把一个模型量化成 W4A16：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

oneshot(
    model="Qwen/Qwen3-8B",
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    num_calibration_samples=512,
    max_seq_length=2048,
    output_dir="Qwen3-8B-W4A16-GPTQ",
)
```

输出头 `lm_head` 通常不量化，它对误差比较敏感，在大模型中所占的参数比例又不大。

量化后的 checkpoint 中，权重存为 `qweight`（每 8 个 4 位数打包成一个 int32），另有每组的 `scales`、`qzeros`（非对称时），以及启用 act-order 时的 `g_idx`。vLLM 加载时会根据显卡自动选择 kernel，比如在 Ampere 和 Hopper 上用 Marlin（2.6 节）。

## 7. 实验：GPTQ 与 RTN

项目 2 在 CPU 上训练了一个 4 层、隐藏维度 256 的小模型，然后用不同的方法量化它的全部线性层，在测试集上计算困惑度。以下是一次运行的结果（全精度为 37.98）：

| 位数与分组 | RTN | GPTQ |
|---|---|---|
| 4 位，每组 128 | 38.20 | 38.13 |
| 3 位，每组 128 | 39.71 | 39.04 |
| 3 位，按通道 | 40.18 | 39.06 |
| 2 位，每组 64 | 78.81 | 52.94 |
| 2 位，每组 32 | 60.46 | 50.05 |

4 位时两者都接近全精度；位数越低，GPTQ 的优势越明显。3 位按通道量化时，GPTQ 的结果甚至和 RTN 每组 128 差不多，相当于用误差补偿换掉了分组的开销。复现：

```bash
cd code/project2
python tools/make_corpus.py --out data
python tools/train_tiny_lm.py --data data --out /tmp/tiny-lm          # 两核 CPU 约 6 分钟
QUANTLAB_IMPL=solution python run.py --model /tmp/tiny-lm --data data --method rtn  --bits 3
QUANTLAB_IMPL=solution python run.py --model /tmp/tiny-lm --data data --method gptq --bits 3
```

每次训练出的模型略有不同，数字会有小的出入，但趋势是稳定的。在 GPU 上可以直接对 Qwen3-0.6B 做同样的实验：`--model` 指向下载好的模型目录，加 `--device cuda`。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "GPTQ 优化的目标是什么？", "options": ["让每个权重的量化误差都尽可能小", "让这一层在校准数据上的输出误差尽可能小", "让整个模型在校准数据上的损失尽可能小", "让权重的分布尽可能均匀"], "answer": 1, "explain": "GPTQ 逐层最小化 &#36;\\|WX - \\hat{W}X\\|^2&#36;。逐个权重的误差最小正是 RTN 做的事；整个模型的损失需要反向传播，属于量化感知训练的范畴。"}, {"q": "GPTQ 用到的矩阵 &#36;H = 2XX^\\top&#36; 由什么决定？", "options": ["只由这一层的权重决定", "只由这一层的输入决定，同一层的所有行共用", "由权重和输入共同决定，每一行各不相同", "由输出头的梯度决定"], "answer": 1, "explain": "输出误差对每一行权重的二阶导数都是 &#36;2XX^\\top&#36;，与权重无关，所以一层只需计算和分解一次。这也是 GPTQ 能让所有行按相同顺序量化、共用一份 &#36;H^{-1}&#36; 的前提。"}, {"q": "相比 OBQ，GPTQ 让所有行按同一个固定顺序量化。这一改动的主要作用是：", "options": ["提高量化精度", "只需维护一份 &#36;H^{-1}&#36;，计算量大幅下降", "让分组量化成为可能", "去掉了对校准数据的依赖"], "answer": 1, "explain": "OBQ 每行按各自的贪心顺序量化，要分别更新各自的 &#36;H^{-1}&#36;。GPTQ 发现固定顺序的精度损失很小，大模型上尤其如此，换来的是计算量降低一个数量级以上。"}, {"q": "开启 act-order（desc_act）并同时使用分组量化，推理时可能有什么代价？", "options": ["模型体积变成原来的两倍", "同一组的权重不再连续，需要额外的 g_idx 索引，部分 kernel 会变慢", "必须在推理时重新计算 Hessian", "只能在 CPU 上运行"], "answer": 1, "explain": "按激活排序后再分组，每组包含的是排序后相邻的列，在原矩阵中是分散的。kernel 要按 g_idx 找到每一列对应的 scale。static groups 先按原顺序确定分组，可以避开这个问题。"}]'></div>

## 延伸阅读

- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), ICLR 2023。
- Frantar & Alistarh, [Optimal Brain Compression](https://arxiv.org/abs/2208.11580), NeurIPS 2022。OBQ 的原始论文。
- Hassibi & Stork, Second Order Derivatives for Network Pruning: Optimal Brain Surgeon, NeurIPS 1992。
- 原书第 6 章 *Quantize a model yourself*。
