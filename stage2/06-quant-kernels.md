# 2.6 低比特算子

> 书外内容。原书第 6 章讨论了 W4A16 和 W8A8 的性能表现，没有展开背后的 kernel。这一节解释那些现象从何而来。

前几节讨论的是量化之后的数值怎样尽可能接近原始值。但量化最终是为了快，而快不快取决于 kernel。一个 4 位的模型如果用一个写得不好的 kernel 运行，可能比 BF16 还慢。这一节看三个问题：低比特的权重在显存里怎样存放，W4A16 的 kernel 怎样一边反量化一边计算，以及 FP8 的 scale 在矩阵乘法中放在哪里。

## 1. 收益随 batch 怎样变化

先用 Roofline 模型（0.7 节）估算一下。线性层的计算量是 $2 \times \text{参数量} \times \text{token 数}$，读权重的字节数与 token 数无关。token 数少时，耗时由读权重决定，权重越小越快；token 数多时，耗时由计算决定，只有更快的 Tensor Core 才有帮助。

<div class="llm-widget" data-widget="lowbit-gemm"></div>

这条曲线解释了原书实验中的现象：并发低时 W4A16 最快；并发高时 W4A16 甚至比原始模型还慢，FP8 W8A8 则一直保持优势。也可以切换到 A100 或 A10，这两张卡不支持 FP8 计算，FP8 模型只能作为 W8A16 运行，计算瓶颈时没有收益。

真实的 kernel 比这个模型复杂：batch 很小时，每个 SM 分到的工作太少，带宽很难跑满；W4A16 在计算瓶颈时的损失取决于反量化的实现。Marlin 论文在 A10 上实测，batch 16 到 32 时加速接近理想的 4 倍，到 batch 128 时降到约 1.5 倍。

## 2. INT4 权重怎样存放

GPU 没有 4 位的数据类型，最小的内存访问单位是字节。所以 4 位的权重要打包：8 个 4 位整数放进一个 32 位整数。GPTQ 格式的 checkpoint 里，`qweight` 就是这样打包的 int32 张量，另外还有每组的 `scales` 和 `qzeros`（2.2 节）。

`code/project2/tools/int4_pack.py` 演示了打包和解包：

```bash
cd code/project2
python tools/int4_pack.py
```

```text
原始权重 (4, 32)，FP16 共 256 字节；打包后 (4, 4)，共 64 字节
第 0 行的前 8 个量化值：[8, 8, 10, 8, 6, 9, 12, 11]  打包成 0xbc968a88
```

checkpoint 中的打包顺序是为了存储方便，不一定适合计算。Marlin、Machete 这样的 kernel 在加载模型时会把权重**重排**一次：按 Tensor Core 指令需要的数据布局重新排列，让每个线程一次读入的 32 位里恰好是它接下来要用的那几个权重。vLLM 启动时把 GPTQ、AWQ 权重转换成 Marlin 格式，就是在做这件事。

## 3. W4A16：在寄存器里反量化

W4A16 kernel 的主循环大致是：

1. 从显存读一块打包的 4 位权重和对应的 scale，读一块 BF16 激活；
2. 在寄存器里把权重解包、反量化成 BF16；
3. 用 BF16 的 Tensor Core 指令做矩阵乘法，累加到 FP32；
4. 重复，直到遍历完整个求和维度。

关键是第 2 步不能把反量化后的权重写回显存，否则读写的字节数反而比直接用 BF16 权重还多。反量化完全在寄存器里完成，显存里自始至终只有 4 位的权重。

第 2 步也必须足够快。整数转浮点的指令吞吐不高，Marlin 等 kernel 用了一个位运算的技巧：FP16 的 1024.0 编码是 `0x6400`，它的尾数部分正好可以表示 0 到 1023 的整数增量。把 4 位整数 $u$ 放进尾数的低 4 位，得到的 FP16 恰好等于 $1024 + u$：

```text
0x6400 | u 作为 FP16 的值： [1024, 1025, 1026, ..., 1039]
```

再减去 $1024 + 8$（8 是对称量化的零点），就得到了 $u - 8$。一条按位或、一条减法，而且两个 FP16 打包在一个 32 位寄存器里可以同时处理。`int4_pack.py` 里的 `magic_dequant` 用 NumPy 实现了这个过程。

为什么 W4A16 在计算瓶颈时会变慢：Tensor Core 的计算量和 BF16 完全一样，又多出了解包、反量化和乘 scale 的指令。这些指令和 Tensor Core 争用同一批线程的发射槽，算力越强的卡，这部分开销占比越明显。

### 常见的 W4A16 kernel

| kernel | 适用的显卡 | 说明 |
|---|---|---|
| Marlin | Ampere 及以后 | IST Austria 开发，vLLM 中 GPTQ、AWQ 模型的默认 kernel（`gptq_marlin`、`awq_marlin`），还有支持 2:4 稀疏的版本 |
| Machete | Hopper | 基于 CUTLASS，使用 Hopper 的 wgmma 指令，batch 较大时比 Marlin 快，vLLM 在 H100 上对支持的格式优先选用 |
| ExLlamaV2、AWQ GEMM 等 | 各种 | 较早的实现，已经很少作为默认选项 |

vLLM 启动时会打印选用的 kernel，例如 `Using MarlinLinearKernel` 或 `Using MacheteLinearKernel`，遇到性能问题时可以先确认这一行。

## 4. W8A8：scale 放在哪里

2.4 节说过，W8A8 的 scale 只能放在求和之外的维度上：

$$
Y = \text{diag}(\Delta^X) \left( \hat{X} \hat{W}^\top \right) \text{diag}(\Delta^W)
$$

括号里是纯粹的 INT8 或 FP8 矩阵乘法，结果累加在 INT32 或 FP32 中；乘 scale 放在 kernel 的**收尾阶段**（epilogue），也就是累加结束、写回显存之前，对每个输出元素乘以所在行的 $\Delta^X_i$ 和所在列的 $\Delta^W_k$。这一步只有逐元素的乘法，几乎没有开销。vLLM 的 `cutlass_scaled_mm` 就是这样实现的，支持按张量、按 token、按通道的各种组合。

按块的 FP8（DeepSeek-V3 的方案，2.4 节）稍复杂一些：每 128 个求和元素就换一组 scale，所以不能只在收尾时乘一次，需要每算完一个 128 的块，就把部分和乘上这一块的 scale，再加到 FP32 的累加器上。DeepSeek 开源的 DeepGEMM 实现了这种 kernel。

DeepSeek-V3 的技术报告还提到一个细节：H800 上 FP8 Tensor Core 内部累加的精度只有约 14 位，求和维度很长时误差会累积。他们的做法是每 128 个元素把部分和提升到 FP32 累加器中，按块的 scale 正好在这一步一起处理。

## 5. 小结

| 方案 | 显存中的权重 | 矩阵乘法 | 额外工作 | 最适合 |
|---|---|---|---|---|
| W4A16 | 4 位，打包并重排 | BF16 Tensor Core | 寄存器里反量化 | 带宽瓶颈：小 batch decode |
| FP8 W8A8 | 8 位 | FP8 Tensor Core | 激活的动态量化，epilogue 里乘 scale | 计算瓶颈：prefill、大 batch |
| INT8 W8A8 | 8 位 | INT8 Tensor Core | 同上，累加在 INT32 | Ampere 上的计算瓶颈 |
| W8A16（FP8 权重） | 8 位 | BF16 Tensor Core | 寄存器里反量化 | 不支持 FP8 计算的卡 |

第 3 部分会用 Triton 写矩阵乘法和融合算子，到时可以回头试着实现一个简单的 W4A16 kernel。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "W4A16 的 kernel 在哪里把 4 位权重反量化成 BF16？", "options": ["加载模型时一次性反量化，存回显存", "每次前向前反量化到一块临时显存", "在寄存器里，读入之后、送进 Tensor Core 之前", "在 CPU 上"], "answer": 2, "explain": "如果把反量化结果写回显存，读写的字节数比直接用 BF16 权重还多，量化就失去了意义。显存中自始至终只有 4 位的权重。"}, {"q": "把 4 位整数 &#36;u&#36; 放进 FP16 数 <code>0x6400</code>（1024.0）尾数的低 4 位，得到什么？", "options": ["&#36;1024 \\times u&#36;", "&#36;1024 + u&#36;", "&#36;u / 1024&#36;", "一个非法的 FP16 值"], "answer": 1, "explain": "1024 到 2047 之间 FP16 的间隔正好是 1，尾数的低位就是整数增量。再减去 &#36;1024 + 8&#36; 就得到对称量化的 &#36;u - 8&#36;，避开了较慢的整数转浮点指令。"}, {"q": "在 H100 上，batch 达到几百个 token 以后，W4A16 为什么可能比 BF16 还慢？", "options": ["4 位权重读取更慢", "此时是计算瓶颈，W4A16 的 Tensor Core 计算量与 BF16 相同，还多了解包和反量化的指令", "batch 大时 4 位权重会溢出", "H100 不支持 4 位权重"], "answer": 1, "explain": "W4A16 只减少了读权重的字节数。进入计算瓶颈后这部分收益消失，反量化的开销却还在。要在计算瓶颈时加速，需要 W8A8 这样真正低精度的矩阵乘法。"}, {"q": "按 token 的激活 scale 和按通道的权重 scale，在 W8A8 的 kernel 中通常在什么时候乘上？", "options": ["在主循环中，每次乘加之前", "在收尾阶段（epilogue），累加完成后对每个输出元素乘以对应的行、列 scale", "在下一层的输入中", "不需要乘，scale 已经并入权重"], "answer": 1, "explain": "这两种 scale 都在求和维度之外，可以从求和中提出来，所以主循环是纯粹的低精度乘加，最后逐元素乘 scale。按块量化时 scale 随求和维度变化，才需要在主循环中分块处理。"}]'></div>

## 延伸阅读

- Frantar et al., [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743), PPoPP 2025，以及 [代码](https://github.com/IST-DASLab/marlin)。
- [Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)，Red Hat 开发者博客，2024。
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 第 3.3 节 FP8 训练，以及 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。
