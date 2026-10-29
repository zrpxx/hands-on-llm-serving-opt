# 3.2 FlashAttention

> 对应原书第 6 章 *Kernel Fusion and Custom Attention Kernels*。在线 softmax 的推导、FlashDecoding 和 FlashInfer 是书外内容。

注意力的计算量并不复杂：两次矩阵乘法加一次 softmax。但用最直接的方式实现，它的大部分时间都花在读写显存上，而且显存占用随序列长度平方增长。FlashAttention 用分块和在线 softmax 解决了这两个问题，现在几乎所有的推理引擎都建立在它或它的变体之上。

这一节先看标准实现的问题，然后推导在线 softmax，再看 FlashAttention 的几代改进，以及 decode 阶段的注意力 kernel 有什么不同。

## 1. 标准实现的问题

一个头的注意力，$Q$、$K$、$V$ 的形状都是 $[N, d]$：

$$
S = \frac{QK^\top}{\sqrt{d}}, \qquad P = \text{softmax}(S), \qquad O = PV
$$

用 PyTorch 逐行写出来，$S$ 和 $P$ 都是 $[N, N]$ 的矩阵，要完整地写进显存。$N = 32\text{K}$ 时，一个头的 $S$ 就有 2 GiB（BF16）。而且每一步操作（矩阵乘、缩放、加掩码、softmax、矩阵乘）都是一个单独的 kernel，每个 kernel 都要把 $N^2$ 的矩阵从显存读进来、再写回去。

与此相比，注意力真正的输入输出只有 $Q$、$K$、$V$、$O$，一共 $4Nd$ 个数。$d = 128$ 时，只要 $N$ 超过几百，中间矩阵的读写量就远远超过了输入输出。注意力是被显存带宽卡住的，而且卡在本来不必要的读写上。

原书把这个问题概括为：FlashAttention 的目标是避免在慢速的显存中生成大矩阵，把计算拆成小块放在片上的 SRAM 和寄存器里完成。难点在于 softmax：它要先知道一整行的最大值和指数和，才能算出任何一个元素的结果。如果按列分块，处理第一块时还不知道后面的块里有没有更大的值。

## 2. 在线 softmax

先看一行。数值稳定的 softmax 是：

$$
m = \max_j s_j, \qquad \ell = \sum_j e^{s_j - m}, \qquad p_j = \frac{e^{s_j - m}}{\ell}
$$

现在把这一行分成若干块，从左到右处理。处理完前几块，我们有当前的最大值 $m$ 和对应的 $\ell$。遇到新的一块，它的最大值是 $\tilde m$：

$$
m' = \max(m, \tilde m), \qquad \ell' = \ell \cdot e^{m - m'} + \sum_{j \in \text{新块}} e^{s_j - m'}
$$

关键是 $e^{m - m'}$ 这个修正系数：之前累加的每一项都是以 $m$ 为基准计算的，换成新的基准 $m'$，只需要整体乘以 $e^{m - m'}$。最大值没有变化时，修正系数是 1。

输出也可以这样累加。维护一个未归一化的输出 $o = \sum_j e^{s_j - m} v_j$，每处理一块：

$$
o' = o \cdot e^{m - m'} + \sum_{j \in \text{新块}} e^{s_j - m'} v_j
$$

全部处理完后，$O = o / \ell$。整个过程只扫描了一遍 $K$ 和 $V$，从来不需要完整的一行 $S$。

## 3. FlashAttention 的分块

把在线 softmax 推广到矩阵：$Q$ 按行分成大小为 $B_r$ 的块，$K$、$V$ 按行分成大小为 $B_c$ 的块。每个线程块（Triton 中的一个 program）负责一个 $Q$ 块，在片上维护这 $B_r$ 个查询的 $m$、$\ell$ 和 $o$，依次读入每个 $K$、$V$ 块：

```text
对每个 Q 块（并行）：
    读入 Q_i；m = -∞，ℓ = 0，o = 0
    对每个 K、V 块 j（顺序）：
        读入 K_j、V_j
        S_ij = Q_i K_jᵀ / √d                 # [B_r, B_c]，只在片上
        按在线 softmax 更新 m、ℓ，修正并累加 o
    写出 O_i = o / ℓ
```

下面的组件演示了这个过程。可以跟踪某一行的 $m$ 和 $\ell$，看它们怎样逐块更新，最终和一次性计算的结果相同：

<div class="llm-widget" data-widget="flash-tiling"></div>

加上因果掩码后，矩阵右上角的块完全被掩码，可以整块跳过，计算量减少约一半。只有对角线上的块需要逐元素地判断掩码。

FlashAttention 的计算量并没有减少，反而略有增加（修正系数的乘法）。它的收益来自两点：

- **显存读写**。$S$ 和 $P$ 从不写入显存。$K$、$V$ 每个 $Q$ 块读一遍，其中很大一部分会命中 L2 缓存。论文证明，在 SRAM 大小为 $M$ 时，它的显存访问量是 $O(N^2 d^2 / M)$，而标准实现是 $O(Nd + N^2)$。
- **显存占用**。额外的显存只有每行的 $m$ 和 $\ell$，是 $O(N)$ 的。这使得很长的上下文成为可能。

训练时的反向传播需要 $P$。FlashAttention 不保存 $P$，而是保存每行的 $m$ 和 $\ell$，反向时重新计算，用一些计算换取显存。推理只需要前向，没有这个问题。

## 4. 几代改进

| 版本 | 主要改进 |
|---|---|
| FlashAttention（2022） | 分块与在线 softmax，融合整个注意力 |
| FlashAttention-2（2023） | 减少矩阵乘以外的运算；除了按头和 batch 并行，也按查询块并行，长序列时也能用满 GPU；改进线程束之间的分工。在 A100 上达到峰值算力的 50% 到 73% |
| FlashAttention-3（2024） | 针对 Hopper：用 wgmma 和 TMA 指令；生产者、消费者线程束分工，让 softmax 与矩阵乘重叠执行；支持 FP8。在 H100 上 FP16 达到约 740 TFLOPS，峰值的 75% |
| FlashAttention-4（2025） | 针对 Blackwell，用 CuTe DSL 编写 |

> [!NOTE]
> 原书说 FlashAttention-2 和 3 都加入了"矩阵乘与 softmax 的重叠计算"。这是 FlashAttention-3 的特点，它依赖 Hopper 的异步指令；FlashAttention-2 的改进主要是并行方式和工作划分。

每一代都和硬件紧密相关。FlashAttention-3 只能在 Hopper 上运行，在 A100 上推理引擎仍然使用 FlashAttention-2 或 FlashInfer。这是同一个算法在不同硬件上需要不同 kernel 的典型例子。

## 5. decode 阶段的注意力

前面讨论的都是 prefill：很多查询对很多 key。decode 时每个请求只有一个查询，情况完全不同：

- **算术强度很低**。一个查询对 $N$ 个 key，读 $2Nd$ 个数，只做约 $4Nd$ 次运算。这完全是带宽瓶颈，目标是让读取速度接近峰值带宽。
- **并行度不够**。按"每个请求的每个头一个线程块"来分，batch 小、上下文长时，线程块的数量远少于 SM 的数量。比如 batch 为 1、8 个 KV 头时只有 8 个线程块，H100 有 132 个 SM，大部分闲着。

**FlashDecoding**（2023）的办法是沿 KV 的长度再切分：把一个请求的 KV 分成若干段，每段由一个线程块独立计算部分结果 $(m_s, \ell_s, o_s)$，最后再合并。合并用的还是在线 softmax 的公式：

$$
m = \max_s m_s, \qquad \ell = \sum_s \ell_s\, e^{m_s - m}, \qquad O = \frac{\sum_s o_s\, e^{m_s - m}}{\ell}
$$

论文报告长序列 decode 时最多快 8 倍。vLLM、SGLang 的 decode kernel 都采用了这种切分。

decode 还要处理分页的 KV Cache（1.3 节）：一个请求的 KV 分散在不连续的物理块里，kernel 按块表找到每一块的位置。项目 3 的第 5 步就是写这样一个 kernel，它和项目 1 的 KV 布局相同。

## 6. 推理引擎中的注意力后端

推理引擎通常同时集成了几种注意力 kernel，按硬件和模型自动选择：

| 后端 | 说明 |
|---|---|
| FlashAttention | vLLM 在 NVIDIA GPU 上的默认选择，包括分页和 decode 的变体 |
| FlashInfer | 专为推理设计的注意力库，用块稀疏的格式统一描述分页 KV 和各种掩码，支持共享前缀的级联注意力 |
| FlashMLA | DeepSeek 为 MLA 开源的 decode kernel |
| Triton | 用 Triton 写的实现，可移植性好，常用于 AMD 等平台 |

原书给出了在 vLLM 中切换后端的方法：

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
vllm serve Qwen/Qwen3-8B
```

SGLang 使用 `--attention-backend` 参数。不同版本的可选值和默认值会变化，以启动日志中打印的实际后端为准。原书的建议是先使用默认的后端，把其他优化做完之后，再试验不同的 kernel。

## 7. 实验

项目 3 的第 3 步和第 4 步分别实现在线 softmax 和 FlashAttention 的前向，第 5 步实现分页的 decode 注意力。三者都可以在 CPU 上用 Triton 的解释器运行和测试：

```bash
cd code/project3
FORGE_IMPL=solution python -m pytest -q tests -k "step03 or step04 or step05"
```

有 GPU 时，`bench.py` 会把它们与 PyTorch 的 `scaled_dot_product_attention` 比较。对于这个只有几十行的教学实现，能达到 PyTorch 速度的一半就很不错了，差距主要来自前面表格里 FlashAttention-2、3 的那些优化。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "在线 softmax 处理新的一块时，最大值从 &#36;m&#36; 变成了 &#36;m_{\\text{new}} > m&#36;。已经累加的 &#36;\\ell&#36; 要怎样处理？", "options": ["清零重新累加", "乘以 &#36;e^{m - m_{\\text{new}}}&#36;", "乘以 &#36;e^{m_{\\text{new}} - m}&#36;", "不需要处理"], "answer": 1, "explain": "之前的每一项都是 &#36;e^{s_j - m}&#36;，换成新基准后应为 &#36;e^{s_j - m_{\\text{new}}} = e^{s_j - m} \\cdot e^{m - m_{\\text{new}}}&#36;。未归一化的输出 &#36;o&#36; 也乘以同一个系数。"}, {"q": "FlashAttention 相比标准实现，主要减少了什么？", "options": ["浮点运算次数", "显存读写量和显存占用：S、P 从不写入显存，额外显存只有 O(N)", "KV Cache 的大小", "模型参数量"], "answer": 1, "explain": "计算量没有减少，甚至略有增加。注意力在标准实现下受限于中间矩阵的显存读写，FlashAttention 让这部分读写消失了。"}, {"q": "batch 为 1、上下文 64K 的 decode，如果每个请求的每个 KV 头只分配一个线程块，会有什么问题？FlashDecoding 怎样解决？", "options": ["显存不够；把 KV 卸载到 CPU", "线程块太少，大部分 SM 闲置；沿 KV 长度切成多段并行计算，最后用 log-sum-exp 合并", "精度不够；改用 FP32", "没有问题"], "answer": 1, "explain": "每段算出自己的 &#36;(m_s, \\ell_s, o_s)&#36;，合并时按 &#36;e^{m_s - m}&#36; 加权，与在线 softmax 的修正完全相同。"}, {"q": "因果掩码下，FlashAttention 为什么能把计算量减少约一半？", "options": ["因为 softmax 只需要算一半", "完全位于对角线右上方的块被整块跳过，不需要读入对应的 K、V，也不做计算", "因为 V 不需要读取", "因为可以用 FP8"], "answer": 1, "explain": "查询块只需要遍历到它能看到的最远位置。只有跨过对角线的块需要逐元素加掩码。"}]'></div>

## 延伸阅读

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022。
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), ICLR 2024。
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608), NeurIPS 2024。
- Dao et al., [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/), 2023。
- Ye et al., [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005), MLSys 2025。
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018。
