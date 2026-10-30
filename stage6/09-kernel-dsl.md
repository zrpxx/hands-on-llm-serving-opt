# 6.9 算子 DSL：TileLang 与 CuTe DSL

> 书外内容。3.3 节用 Triton 写了几个融合算子，项目 3 实现了 RMSNorm、SwiGLU、FlashAttention 等 kernel。本节比较近两年流行的几种 kernel 编程方式。TileLang 和 CuTe DSL 都需要 NVIDIA GPU 才能编译运行，本节的代码只用于阅读。

推理引擎的性能最终落在 kernel 上。2024 年以前，写高性能 kernel 基本只有两条路：直接写 CUDA C++（配合 CUTLASS 模板库），或者用 Triton。近两年出现了一批新的选择，DeepSeek 的 DeepInfer 和 V3.2 的稀疏注意力用 TileLang，FlashAttention-4 用 CuTe DSL。它们的出现与硬件的变化直接相关。

## 1. 为什么又需要新的 DSL

Hopper 和 Blackwell 引入了一系列新的硬件机制：

- **TMA**（Tensor Memory Accelerator）：用一条指令在全局内存和共享内存之间搬运一整块数据，异步完成；
- **WGMMA / tcgen05**：以 warpgroup 为单位的异步矩阵乘法，Blackwell 还加入了专门的 Tensor Memory；
- **warp 专门化**：一部分 warp 只负责搬数据，一部分只负责计算，通过屏障同步；
- **线程块簇**（cluster）：相邻的线程块可以访问彼此的共享内存。

要发挥这些硬件的性能，kernel 需要显式地组织数据搬运、计算和同步的流水线。Triton 的抽象层级较高，编译器替你做了很多决定，在新硬件上往往要等编译器支持，而且难以精细控制。CUDA C++ 能控制一切，但代码冗长，模板错误难以调试，编译很慢。新的 DSL 试图在二者之间找到平衡。

## 2. 几种方式的比较

| | Triton | TileLang | CuTe DSL | CUDA C++ / CUTLASS |
|---|---|---|---|---|
| 语言 | Python | Python | Python | C++ |
| 编程的单位 | 线程块处理的 tile | 线程块处理的 tile，可以下探到线程 | 布局（layout）与线程、数据的映射 | 线程 |
| 共享内存与流水线 | 编译器决定 | 显式分配，`T.Pipelined` 自动生成多级流水线 | 显式 | 显式 |
| 新硬件特性 | 依赖编译器 | 编译器支持，也可以手写 | 直接暴露 | 直接暴露 |
| 编译 | JIT，较快 | JIT，基于 TVM | JIT，比 C++ 模板快得多 | 慢 |
| 典型用户 | vLLM、SGLang 的大量自定义算子 | DeepSeek（V3.2 的 DSA、DeepInfer） | FlashAttention-4、CUTLASS 4 | cuBLAS、FlashAttention-3、DeepGEMM |

Triton 还推出了更底层的 Gluon，让用户直接控制布局和流水线；PyTorch 团队推出了 Helion，在 Triton 之上提供更高层的抽象并自动调优。这个领域仍在快速变化。

## 3. 读一个 TileLang 的矩阵乘法

下面是 TileLang 中一个标准的矩阵乘法，与 3.3 节 Triton 的写法对照着读：

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        # 网格：每个线程块负责 C 的一个 block_M × block_N 的块，128 个线程
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)      # 显式分配共享内存
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_r = T.alloc_fragment((block_M, block_N), accum_dtype)  # 寄存器中的累加器
            T.clear(C_r)
            # 沿 K 循环；num_stages=3 让编译器生成三级流水线：搬运下一块的同时计算这一块
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_s)
                T.copy(B[k * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_r)                            # 映射到 Tensor Core
            T.copy(C_r, C[by * block_M, bx * block_N])
    return gemm
```

和 Triton 相比，几个不同：

- **内存层次是显式的**：共享内存（`alloc_shared`）和寄存器（`alloc_fragment`）分开声明，数据在它们之间的搬运也是显式的 `T.copy`。在 Triton 中，`tl.load` 加载的数据放在哪里由编译器决定。
- **流水线是声明式的**：`T.Pipelined(..., num_stages=3)` 一行就生成了多级的异步搬运与计算的重叠，在 Hopper 上会用到 TMA。
- **需要时可以下探**：同一个 kernel 里可以写到线程级别的代码，处理 tile 抽象表达不了的部分。

这种"默认在 tile 层面写，必要时下探"的设计，是 TileLang 适合写注意力这类复杂 kernel 的原因。DeepSeek-V3.2 发布时，同时开源了 TileLang 版本的 DSA kernel，作为便于阅读的参考实现。

## 4. CuTe DSL

CuTe 是 CUTLASS 3 中描述张量布局的核心抽象：用"形状 + 步长"表示数据在内存中的排布，以及线程与数据之间的映射。CUTLASS 4（2025 年）推出了 CuTe 的 Python 版本，即 CuTe DSL。它保留了 C++ 版本的全部控制力，直接暴露 TMA、WGMMA、tcgen05 等硬件特性，但编译时间从几分钟缩短到几秒，也能在 Python 中调试。

FlashAttention-4 针对 Blackwell，全部用 CuTe DSL 编写。它的抽象层级比 TileLang 更低，写起来更接近 CUDA，适合追求极致性能、需要精确控制每一个硬件细节的 kernel。

## 5. 怎样选择

- **大多数自定义算子**（融合的逐元素操作、归一化、简单的量化）：Triton。生态成熟，项目 3 已经用过，性能足够。
- **注意力、复杂的 MoE、稀疏计算**等需要精细流水线的算子：TileLang 或 CuTe DSL。前者更容易上手，后者控制更细。
- **矩阵乘法**：优先用现成的库（cuBLAS、CUTLASS、DeepGEMM、FlashInfer）。自己写的 GEMM 很难超过它们。

无论选哪种，3.6 节的方法都适用：先用 Roofline 估算这个算子的上限，再用 Nsight Compute 看实际达到了多少，差距在哪。

## 6. 实验

- **CPU**：用项目 3 的 Triton 解释器（`TRITON_INTERPRET=1`）运行 `forge` 中的 RMSNorm，然后对照本节的 TileLang 写法，写出同一个 RMSNorm 的 TileLang 版本（不需要运行）。哪些在 Triton 中隐式的东西，在 TileLang 中需要显式写出来？
- **读代码**：读 DeepInfer 的 `deepinfer/model_executor/layers/tilelang_ops` 中的一个算子，说明它的网格怎样划分、用了几级流水线。
- **GPU（Hopper 及以上）**：安装 TileLang，运行本节的矩阵乘法，与 `torch.matmul` 比较 4096 × 4096 × 4096 的耗时；再把 `num_stages` 改为 1、2、4，观察性能的变化。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "与 Triton 相比，TileLang 在内存层次上的主要不同是：", "options": ["TileLang 不使用共享内存", "共享内存和寄存器显式声明，数据搬运用 T.copy 显式写出；Triton 中数据放在哪里由编译器决定", "TileLang 只能写逐元素算子", "两者没有区别"], "answer": 1, "explain": "TileLang 默认在 tile 层面写，需要时可以下探到线程级别，适合注意力这类复杂的 kernel。"}, {"q": "T.Pipelined(..., num_stages=3) 的作用是：", "options": ["把循环展开 3 次", "生成三级流水线：在计算当前块的同时，异步搬运后面的块", "用 3 个线程块计算", "把 K 切成 3 段"], "answer": 1, "explain": "在 Hopper 上会用到 TMA 异步搬运。级数越多越能掩盖延迟，但占用更多共享内存。"}, {"q": "推理引擎需要一个大小常见的 BF16 矩阵乘法。最合理的做法是：", "options": ["用 Triton 自己写一个", "优先使用 cuBLAS、CUTLASS、DeepGEMM 等现成的库", "用 CuTe DSL 从头写", "在 CPU 上计算"], "answer": 1, "explain": "自己写的 GEMM 很难超过这些高度调优的库。DSL 的价值在于融合、稀疏、注意力这类库覆盖不到的算子。"}]'></div>

## 延伸阅读

- [TileLang](https://github.com/tile-ai/tilelang)，以及 Wang et al., [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577), 2025。
- [CUTLASS 文档：CuTe DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)。
- Tri Dao et al., [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention)，其中 `flash_attn/cute` 是 FlashAttention-4 的 CuTe DSL 实现。
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。
