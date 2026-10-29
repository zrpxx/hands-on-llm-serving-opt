# 3.3 Kernel Fusion 与 Triton 语言

> 对应原书第 6 章 *Kernel fusion*。Triton 编程是书外内容，参考 Triton 官方教程和项目 3 的代码。

上一节的 FlashAttention 本质上是一个融合算子：把矩阵乘、缩放、掩码、softmax、矩阵乘合并成一个 kernel，中间结果不离开芯片。这种思路不只适用于注意力。LLM 的每一层里还有很多小算子，比如残差相加、RMSNorm、RoPE、激活函数，单独执行时都受限于显存带宽。这一节讨论融合能省下什么，然后介绍写融合算子最常用的工具 Triton。

## 1. 融合省下了什么

GPU 执行一个算子的过程是：从显存读入输入，在 SM 上计算，把结果写回显存。对逐元素的运算（加法、激活函数）和简单的归约（求均值），计算本身几乎不花时间，耗时几乎全在读写上。原书用一个图说明了这一点：不融合时，每一步的中间结果都要写回显存再读出来；融合之后，数据读进来一次，所有计算都在寄存器或共享内存里完成，只写出最终结果。

以每层都有的"残差相加 + RMSNorm"为例，用 PyTorch 逐行写：

```python
h = x + residual
var = h.pow(2).mean(-1, keepdim=True)
y = h * torch.rsqrt(var + eps) * weight
```

这几行会启动五六个 kernel，大部分都要把整个 $[N, D]$ 的张量读写一遍。融合成一个 kernel 后，只需要读 $x$ 和 residual、写 $y$ 和新的 residual。下面的组件估算了两者的差别：

<div class="llm-widget" data-widget="fusion-io"></div>

组件揭示了两种情况：

- **token 多的时候（prefill）**，收益来自读写量的减少。残差加 RMSNorm 的读写量减少到约 40%。
- **token 少的时候（decode）**，数据量很小，每个 kernel 的固定开销反而是大头：CPU 发起调用、GPU 调度线程块、kernel 结束时的同步，加起来每个 kernel 要几微秒。一个 8B 模型每一步有几百个 kernel，光是这部分就有一两毫秒。融合减少了 kernel 的数量，3.6 节的 CUDA Graph 则把 CPU 端的启动开销一并去掉。

## 2. 谁来做融合

推理引擎中的融合算子有三个来源：

- **手写的 CUDA kernel**。vLLM 的 `csrc/` 目录下有 RMSNorm、RoPE、激活函数、量化等一系列手写 kernel，性能最好，但编写和维护成本高。
- **编译器自动生成**。`torch.compile` 的后端 Inductor 会分析计算图，把相邻的逐元素和归约操作合并，生成 Triton 代码。vLLM 默认用 `torch.compile` 编译模型，还加了一些自定义的融合规则，比如把 RMSNorm 和其后的 FP8 量化合并、把多卡的 all-reduce 和 RMSNorm 合并。
- **用 Triton 等 DSL 手写**。介于两者之间：比 CUDA 简单得多，又能精确控制融合的方式。vLLM 中 MoE 层的核心 kernel（`fused_moe`）、Triton 注意力后端、部分量化 kernel 都是用 Triton 写的。

## 3. Triton 的编程模型

CUDA 以**线程**为单位编程：你写的是一个线程做什么，要自己考虑线程怎样协作、数据怎样在共享内存里排布。Triton 以**块**为单位编程：你写的是一个 program 处理哪一块数据，块内的并行、向量化、共享内存的使用由编译器负责。

一个最简单的例子，两个向量相加：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # 这是第几个 program
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 这个 program 负责的 BLOCK 个下标
    mask = offs < n                              # 最后一块可能不满
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

grid = (triton.cdiv(n, 1024),)                   # 启动多少个 program
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

需要掌握的几个概念：

| 概念 | 含义 |
|---|---|
| `@triton.jit` | 这个函数会被编译成 GPU kernel，第一次调用时按参数的类型和 `constexpr` 的值编译 |
| `tl.program_id(axis)` | 当前 program 在网格中的位置，相当于 CUDA 的 `blockIdx` |
| `tl.arange(0, BLOCK)` | 一个长度为 BLOCK 的向量，BLOCK 必须是 2 的幂 |
| 指针运算 | 传入的张量变成指向首元素的指针，`ptr + offs` 是一组地址，多维张量要自己按步长计算地址 |
| `mask` | 越界的位置不读不写，`other` 指定读不到时的填充值 |
| `tl.constexpr` | 编译期常量，用于块大小、开关等；每个不同的值会编译出一个不同的 kernel |
| `tl.dot` | 块与块的矩阵乘法，编译为 Tensor Core 指令 |

再看一个归约的例子，项目 3 第 1 步的 RMSNorm。每个 program 处理一行，整行一次读入：

```python
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)                   # BLOCK 是不小于 D 的 2 的幂
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D              # 注意除以 D，不是 BLOCK
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * D + cols, x * tl.rsqrt(var + eps) * w, mask=mask)
```

`other=0.0` 让越界位置读成 0，不影响平方和。计算在 float32 中进行，最后写回时自动转换成输出张量的类型。

## 4. 矩阵乘法与自动调优

Triton 也能写出接近 cuBLAS 性能的矩阵乘法。思路和 FlashAttention 一样：每个 program 负责输出矩阵的一个 $[B_M, B_N]$ 块，沿求和维度每次读入 $A$ 的 $[B_M, B_K]$ 块和 $B$ 的 $[B_K, B_N]$ 块，用 `tl.dot` 累加：

```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, K, BK):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BK * stride_ak
    b_ptrs += BK * stride_bk
```

块大小对性能影响很大，最优值取决于矩阵形状和显卡。`@triton.autotune` 可以列出若干候选配置，第一次运行时逐个测试，选最快的一个：

```python
@triton.autotune(
    configs=[triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
             triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4)],
    key=["M", "N", "K"],                         # 形状变化时重新调优
)
@triton.jit
def matmul_kernel(...): ...
```

`num_stages` 控制软件流水线的级数：在计算当前块的同时预取后面几块，隐藏显存的延迟。Triton 官方教程中的矩阵乘法还用了一个按组重排 program 顺序的技巧，让相邻的 program 读到相同的 $A$ 块，提高 L2 缓存的命中率。

vLLM 的 `fused_moe` 就是这样一个矩阵乘法 kernel：它把属于同一个专家的 token 收集起来，一次 kernel 调用完成所有专家的计算。它为常见的显卡和专家形状预先调好了配置，存在 JSON 文件中。

## 5. 在 CPU 上调试

设置环境变量 `TRITON_INTERPRET=1` 后，Triton kernel 会在 CPU 上用 NumPy 逐个 program 地解释执行。这对学习和调试很有用：

- 没有 GPU 也能验证 kernel 的正确性。项目 3 的全部测试都在解释器模式下运行；
- 可以在 kernel 里用 `print` 打印中间结果，用 Python 调试器单步执行。

解释器模式很慢，而且不会暴露一些只有在 GPU 上才出现的问题，比如超出共享内存、`tl.dot` 对块大小的要求（每一维至少 16）。正确性在 CPU 上验证，性能一定要在 GPU 上测。

## 6. 其他选择

Triton 之外，近两年还出现了一批写高性能 kernel 的工具：NVIDIA 的 CUTLASS 和 CuTe DSL（FlashAttention-4 就是用后者写的）、TileLang、ThunderKittens 等。它们给了比 Triton 更细的控制，适合追求极致性能的场景。6.9 节会做比较。

## 7. 实验

项目 3 的前两步就是本节的内容：

```bash
cd code/project3
FORGE_IMPL=solution python -m pytest -q tests -k "step01 or step02"
```

有 GPU 时，运行 `bench.py`，比较融合的 RMSNorm、SwiGLU 与 PyTorch 逐行实现的耗时，并计算有效带宽（读写的字节数除以耗时），和显卡的峰值带宽（0.5 节）比较。一个写得好的逐元素 kernel 应该能达到峰值带宽的 80% 以上。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "decode 阶段只有 16 个 token 时，把残差加 RMSNorm 的六个 kernel 融合成一个，收益主要来自哪里？", "options": ["计算量减少", "每个 kernel 的固定开销（启动、调度、同步）少了五份；此时数据量很小，固定开销是大头", "显存占用减少", "精度提高"], "answer": 1, "explain": "16 个 token 的张量只有几百 KB，在 H100 上读写不到 1 微秒，而每个 kernel 的固定开销有几微秒。token 多时，收益才主要来自读写量的减少。"}, {"q": "Triton 的 RMSNorm kernel 中，D = 96，BLOCK 取 128。求均方时应该除以多少？越界的位置应该怎样处理？", "options": ["除以 128；越界位置不用管", "除以 96；读取时用 mask 让越界位置填 0，不影响平方和", "除以 96；越界位置填 1", "BLOCK 必须等于 D，不能是 128"], "answer": 1, "explain": "tl.arange 的长度必须是 2 的幂，所以 BLOCK 往往大于真实长度。mask 加 other=0.0 保证越界位置对求和没有贡献，除数要用真实的 D。"}, {"q": "关于 TRITON_INTERPRET=1，下面哪个说法正确？", "options": ["它让 kernel 在 CPU 上以接近 GPU 的速度运行", "它在 CPU 上逐个 program 解释执行 kernel，适合验证正确性和调试，但不能用来测性能，也发现不了共享内存不足等 GPU 特有的问题", "它会把 kernel 编译成 CUDA 代码", "它只支持逐元素的算子"], "answer": 1, "explain": "解释器用 NumPy 模拟每个 program 的执行，可以打印中间结果、单步调试。性能和资源相关的问题只有在 GPU 上才能看到。"}, {"q": "@triton.autotune 中的 num_stages 控制什么？", "options": ["kernel 被调用的次数", "软件流水线的级数：计算当前块的同时预取后面几块，隐藏显存延迟", "program 的数量", "精度"], "answer": 1, "explain": "级数越多，能隐藏的延迟越长，但占用的共享内存也越多。最优值取决于块大小和显卡，所以交给自动调优去选。"}]'></div>

## 延伸阅读

- Tillet et al., [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), MAPL 2019。
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)，尤其是 Fused Softmax、Matrix Multiplication 和 Fused Attention 三篇。
- [vLLM 的 fused_moe kernel](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe)。
- He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html), 2022。原书图 6-11 的出处。
