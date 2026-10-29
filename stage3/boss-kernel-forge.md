# 项目 3　Kernel Forge：用 Triton 写推理算子，实现投机解码

> 前置：3.2～3.4 节。代码在 [`code/project3`](https://github.com/zrpxx/hands-on-llm-serving-opt/tree/main/code/project3)。全部开发和正确性测试都可以在 CPU 上完成：没有 GPU 时，测试会自动用 Triton 的解释器运行 kernel。性能测试需要 NVIDIA GPU。

这个项目有两部分。前五步用 Triton 写推理中最关键的几个算子：融合的 RMSNorm、SwiGLU、在线 softmax、FlashAttention 的前向，以及分页 KV Cache 上的 decode 注意力。最后一步用 PyTorch 实现投机解码的验证逻辑，在两个小模型上测量接受率。

做完之后，你应该能读懂 vLLM、SGLang 中大部分 Triton kernel 的代码，也能独立判断一个投机解码的配置是否合理。

## 1. 代码结构

```text
code/project3/
├── starter/forge/          要完成的代码，6 步
├── solution/forge/         参考实现
├── tests/test_steps.py     分步测试，没有 GPU 时自动设置 TRITON_INTERPRET=1
├── bench.py                在 GPU 上与 PyTorch 比较速度和有效带宽
└── spec_demo.py            在 CPU 上做投机解码实验
```

| 文件 | 要完成的部分 | 步骤 | 对应章节 |
|---|---|---|---|
| `rmsnorm.py` | `_rmsnorm_kernel` 的主体 | 第 1 步 | 3.3 |
| `swiglu.py` | `_swiglu_kernel` 的主体 | 第 2 步 | 3.3 |
| `softmax.py` | `_online_softmax_kernel` 的两遍循环 | 第 3 步 | 3.2 |
| `flash_attn.py` | `_flash_fwd_kernel` 的内层循环 | 第 4 步 | 3.2 |
| `paged_decode.py` | `_paged_decode_kernel` 的内层循环 | 第 5 步 | 3.2、1.3 |
| `spec.py` | `rejection_sample`、`greedy_verify`、`ngram_propose` | 第 6 步 | 3.4、3.5 |

每个文件里，Python 端的包装函数（计算网格、分配输出、传参）和 PyTorch 的参考实现都已经写好。要写的是 kernel 内部的计算。

## 2. 准备环境

```bash
cd code/project3
pip install -r requirements.txt        # torch、triton、transformers、pytest

FORGE_IMPL=solution python -m pytest -q tests    # 参考实现，应全部通过（CPU 上几秒钟）
python -m pytest -q tests                        # 你的实现，现在全部失败
```

在 Linux 上，即使没有 GPU，`pip install triton` 也能直接安装，解释器模式不需要 CUDA。Triton 没有提供 macOS 的版本，Mac 用户可以在 Linux 虚拟机、Docker 或云端的 CPU 实例上完成这个项目。

## 3. 分步完成

| 步骤 | 内容 | 验证 |
|---|---|---|
| 第 1 步 | 融合的残差加 RMSNorm | `python -m pytest tests/test_steps.py -k step01` |
| 第 2 步 | 融合的 SwiGLU | `-k step02` |
| 第 3 步 | 在线 softmax | `-k step03` |
| 第 4 步 | FlashAttention 前向 | `-k step04` |
| 第 5 步 | 分页 decode 注意力 | `-k step05` |
| 第 6 步 | 投机解码 | `-k step06`，之后运行 `spec_demo.py` |
| 第 7 步 | GPU 上的性能测试 | `bench.py` |

### 第 1 步：RMSNorm

每个 program 处理一行。3.3 节第 3 节给出了不带残差的版本，这里还要处理残差：`HAS_RES` 为真时，先把残差加到输入上，把相加的结果写到 `res_out_ptr`（下一层要用它作为残差），再对它做 RMSNorm。

<details>
<summary>提示：HAS_RES 是 constexpr</summary>

`if HAS_RES:` 在编译时就决定了走哪个分支，没有残差的版本里不会有任何读残差的代码。所以没有残差时，包装函数随便传一个指针进去也没有关系。
</details>

测试用 D = 96（不是 2 的幂）检查 mask 和除数是否正确。

### 第 2 步：SwiGLU

输入 `[N, 2I]` 的每一行，前 I 个是 gate，后 I 个是 up。网格是二维的：第一维是行，第二维把一行切成若干个 BLOCK。输出 `silu(gate) * up`，其中 $\text{silu}(x) = x \cdot \sigma(x)$。

测试中 I = 200、BLOCK = 128，第二块只有 72 个有效元素。

### 第 3 步：在线 softmax

一行的长度可能远大于 BLOCK，所以要循环。第一遍循环维护 `m` 和 `l`，第二遍写出结果。3.2 节第 2 节给出了更新公式。

<details>
<summary>提示 1：越界位置填什么</summary>

求最大值时，越界位置应该不影响结果，所以 `other=float("-inf")`。它的 `exp` 是 0，对求和也没有影响。
</details>

<details>
<summary>提示 2：第一块时 m 是 -inf</summary>

`m` 的初值是 $-\infty$，第一块时修正系数 `exp(m - m_new)` 是 `exp(-inf)`，等于 0，乘上初值为 0 的 `l` 仍然是 0，结果正确。不需要特殊处理第一块。
</details>

测试中有一行包含 1000 左右的大数，直接计算 `exp` 会溢出，这一行检查你是否减去了最大值。

### 第 4 步：FlashAttention

这是整个项目最难的一步。kernel 的框架已经写好：每个 program 负责一个头的 BLOCK_M 个查询，已经读入了 Q，初始化了 `m_i`、`l_i`、`acc`，并计算了因果掩码下要遍历到哪里。你要写的是遍历 K、V 块的循环体，以及最后的归一化。

接口有两个要注意的地方：

- **查询可以只是序列的最后 M 个位置**（M ≤ N）。chunked prefill（1.2 节）时，一个块的查询要看到之前所有的 KV。第 $i$ 个查询的绝对位置是 $N - M + i$，已经算好放在 `q_pos` 里。因果掩码的条件是 key 的位置 ≤ 查询的位置。
- **GQA**：第 $h$ 个查询头读第 `h // GROUP` 个 KV 头，框架中的 `kvh` 已经算好。

<details>
<summary>提示 1：形状</summary>

`q` 是 `[BLOCK_M, D]`，`k`、`v` 是 `[BLOCK_N, D]`。`tl.dot(q, tl.trans(k))` 得到 `[BLOCK_M, BLOCK_N]` 的分数。`m_i`、`l_i` 是 `[BLOCK_M]`，和分数做运算时要加 `[:, None]` 扩展成列。
</details>

<details>
<summary>提示 2：掩码</summary>

两种位置要设为 $-\infty$：超出 N 的 key（最后一块可能不满），以及因果时位置在查询之后的 key。用 `tl.where(valid, s, float("-inf"))`。
</details>

<details>
<summary>提示 3：会不会出现整行都是 -inf</summary>

因果时，每个查询至少能看到第 0 个 key，而第 0 个 key 在第一块里，所以处理完第一块后 `m_i` 就是有限值了，之后的修正系数不会出现 $-\infty - (-\infty)$。超出 M 的查询行（最后一个查询块不满）的位置大于 N - 1，能看到所有 key，也不会出问题，它们的结果在写回时被 mask 掉。
</details>

<details>
<summary>提示 4：tl.dot 的类型</summary>

`acc += tl.dot(p.to(v.dtype), v)`：`p` 是 float32，要转成和 `v` 一样的类型才能用 Tensor Core。在 GPU 上，BF16 的输入用 BF16 的 Tensor Core，累加仍然在 float32 的 `acc` 中进行。
</details>

三个测试分别覆盖：不加掩码、因果加 GQA 加 M < N、序列长度不是块大小的整数倍。

### 第 5 步：分页 decode 注意力

每个 program 负责一个序列的一个查询头。KV 缓存的布局和项目 1 相同：`[num_blocks, block_size, H_kv, D]`，`block_tables[b, j]` 是序列 b 的第 j 个逻辑块的物理块号。循环遍历这个序列的每一个逻辑块，读出物理块中第 `kvh` 个头的 K、V，做在线 softmax。

<details>
<summary>提示 1：地址怎样算</summary>

物理块 `phys` 中第 `t` 个槽位、第 `kvh` 个头、第 `d` 维的元素，偏移是 `((phys * BLOCK_SIZE + t) * HKV + kvh) * D + d`。用 `offs_t[:, None]` 和 `offs_d[None, :]` 一次算出 `[BLOCK_SIZE, D]` 个地址。
</details>

<details>
<summary>提示 2：为什么这里不用 tl.dot</summary>

只有一个查询，分数是 `[BLOCK_SIZE]` 的向量，用 `tl.sum(k * q[None, :], axis=1)` 计算即可。`tl.dot` 要求每一维至少 16。生产级的 kernel 会把同一组 GQA 的几个查询头放在一起，凑成矩阵再用 Tensor Core，这是扩展练习之一。
</details>

测试中物理块的顺序是打乱的，三个序列的长度分别是 1、33、64，覆盖了只有一个 token、最后一块不满、恰好填满几种情况。

### 第 6 步：投机解码

`spec.py` 中的 `speculative_generate` 已经写好了外层的循环：起草、目标模型的一次前向、验证、统计。你要实现三个函数：

- `rejection_sample`：3.4 节第 2.2 节的拒绝采样。
- `greedy_verify`：贪心解码时的验证。
- `ngram_propose`：在已有 token 中查找与结尾相同的 n-gram。

<details>
<summary>提示 1：返回值的长度</summary>

两个验证函数都返回这一轮最终输出的 token，长度是"接受的草稿数 + 1"。最后那一个，要么是被拒绝位置重新采样的结果，要么是全部接受后额外采样的结果。
</details>

<details>
<summary>提示 2：n-gram 的查找</summary>

对每个 n（从 `n_max` 到 `n_min`），取结尾的 n 个 token，从倒数第 n+1 个位置往前找相同的片段，找到最近的一个就返回它后面最多 k 个 token。注意不要匹配到结尾本身。
</details>

测试检查四件事：

- 用一个很差的草稿分布做 6000 次拒绝采样，输出的分布与目标分布的偏差不超过 0.025；
- 贪心验证在全部接受和中途拒绝时的输出；
- n-gram 的查找结果；
- 用两个随机的小模型做贪心的投机解码，输出与目标模型自己的贪心解码逐 token 一致，草稿模型和 n-gram 两种方式都要检查。

通过之后，运行 3.4 节的 `spec_demo.py`。草稿模型要用项目 2 的训练脚本，复用目标模型的分词器：

```bash
python ../project2/tools/train_tiny_lm.py --data ../project2/data --out /tmp/draft-lm \
    --tokenizer /tmp/tiny-lm --layers 1 --hidden 128
python spec_demo.py --target /tmp/tiny-lm --draft /tmp/draft-lm --data ../project2/data
```

### 第 7 步：GPU 上的性能

有 NVIDIA GPU 时，运行：

```bash
python bench.py
```

它会用 Qwen3-8B 量级的形状测试每个 kernel，与 PyTorch 的实现比较耗时：

- RMSNorm、SwiGLU、softmax、decode 注意力是带宽瓶颈，报告**有效带宽**，即"最少需要读写的字节数 ÷ 耗时"。和显卡的峰值带宽（0.5 节）比较，好的实现能达到 80% 以上。
- FlashAttention 是计算瓶颈，报告 TFLOPS，和 PyTorch 的 `scaled_dot_product_attention` 比较。

对每个 kernel，分析它距离硬件的上限还差多少，原因可能是什么。可以用 3.6 节的 Nsight Compute 验证你的猜测。

> [!NOTE]
> 参考实现的正确性在 CPU 的解释器中验证过，`bench.py` 在发布前没有在 GPU 上运行过。如果在你的 GPU 上遇到编译错误或数值问题，那正好是第 7 步的一部分：记录下来，修好它。

## 4. 完成标准

- `python -m pytest -q tests` 全部通过。
- `spec_demo.py` 的结果表，并根据各位置的接受率，用 3.4 节的公式算出最优的 K，与实测对比。
- 有 GPU 时：`bench.py` 的结果，以及对每个 kernel 与硬件上限之间差距的分析。

## 5. 扩展练习

- **FlashDecoding。** 给第 5 步的 kernel 加一个维度，把一个序列的 KV 切成若干段并行处理，再写一个合并的 kernel（3.2 节第 5 节的公式）。在 batch 小、上下文长时比较加速效果。
- **GQA 打包。** 让 decode kernel 的一个 program 同时处理同一组的所有查询头，把分数计算变成矩阵乘法，用上 `tl.dot`。
- **自动调优。** 给 FlashAttention 加上 `@triton.autotune`，尝试不同的 BLOCK_M、BLOCK_N、`num_warps`、`num_stages`，看最优配置在不同序列长度下是否变化。
- **融合量化。** 在 RMSNorm 的输出端直接做按 token 的 FP8 量化（2.4 节），输出 FP8 张量和每行的 scale。这是 vLLM 中真实存在的融合。
- **树形验证。** 为 `speculative_generate` 实现一个最简单的树：第一个位置取草稿模型的前两个候选，各自往后起草，用树形注意力掩码一次验证两条路径。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "chunked prefill 时，FlashAttention 的查询是长度为 N 的序列中最后 M 个位置。第 i 个查询能看到哪些 key？", "options": ["第 0 到第 i 个", "第 0 到第 N - M + i 个", "全部 N 个", "第 i 到第 N - 1 个"], "answer": 1, "explain": "第 i 个查询的绝对位置是 N - M + i，因果掩码按绝对位置比较。直接用 i 作为位置，是实现切块时最常见的错误。"}, {"q": "分页 decode kernel 中，物理块 phys、槽位 t、KV 头 kvh、维度 d 的元素在 [num_blocks, block_size, H_kv, D] 缓存中的偏移是：", "options": ["phys * D + t", "((phys * block_size + t) * H_kv + kvh) * D + d", "(t * H_kv + kvh) * D + d", "phys * block_size * D + d"], "answer": 1, "explain": "按行优先的布局逐维展开即可。phys 来自块表，逻辑上相邻的块在物理上可以相隔很远，这正是分页的意义。"}, {"q": "为什么第 6 步的测试要用一个很差的草稿分布来检验拒绝采样？", "options": ["为了让测试跑得更快", "草稿与目标分布差别越大，错误的实现（例如重新采样时直接用 p）造成的偏差越明显，越容易被发现", "因为好的草稿分布无法采样", "为了提高接受率"], "answer": 1, "explain": "如果草稿分布和目标分布几乎相同，几乎所有草稿都被接受，重新采样的分支很少执行，写错了也看不出来。"}]'></div>
