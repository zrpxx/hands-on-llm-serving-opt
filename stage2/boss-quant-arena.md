# 项目 2　Quant Arena：亲手实现主流量化方法

> 前置：2.1～2.5 节。代码在 [`code/project2`](https://github.com/zrpxx/hands-on-llm-serving-opt/tree/main/code/project2)。全部开发、测试和小模型上的对比实验都可以在 CPU 上完成，真实模型的精度与性能实验需要 GPU。

读懂 GPTQ 的推导和把它写对是两回事。这个项目从最基本的量化、反量化函数开始，依次实现 FP8、GPTQ、AWQ、SmoothQuant 和 KV Cache 量化，然后把它们用到一个真实结构（Qwen3）的小模型上，在同一个测试集上比较困惑度。所有实现都是**伪量化**（fake quantization）：量化之后立刻反量化回浮点数，用来测量精度。性能则交给 vLLM 在 GPU 上测。

## 1. 代码结构

```text
code/project2/
├── starter/quantlab/        要完成的代码，6 步共 8 处 TODO
├── solution/quantlab/       参考实现
├── run.py                   量化一个模型并计算困惑度
├── tools/make_corpus.py     生成训练和测试语料（本站的 Markdown 加上 Python 标准库的文档）
├── tools/train_tiny_lm.py   在 CPU 上训练一个 4 层的 Qwen3 结构小模型
├── tools/outlier_layer.py   构造带离群通道的线性层，比较各种方法
├── tools/int4_pack.py       2.6 节：INT4 打包与位运算反量化
├── tools/prune_24.py        2.8 节：2:4 稀疏剪枝
└── tests/                   分步测试和端到端测试
```

| 文件 | 要完成的函数 | 步骤 | 对应章节 |
|---|---|---|---|
| `formats.py` | `int_qparams`、`int_quant_dequant` | 第 1 步 | 2.1 |
| `formats.py` | `fake_quant_fp8` | 第 2 步 | 2.1、2.4 |
| `gptq.py` | `gptq_quantize` 的逐列循环 | 第 3 步 | 2.2 |
| `awq.py` | `awq_search` | 第 4 步 | 2.3 |
| `smoothquant.py` | `smooth_scales`、`w8a8_matmul` | 第 5 步 | 2.4 |
| `kv.py` | `fake_quant_kv` | 第 6 步 | 2.5 |

逐层量化整个模型的框架（`model_quant.py`：用钩子收集每层的输入、按层顺序量化、替换线性层）和困惑度评测（`eval.py`）已经写好。

## 2. 准备环境

```bash
cd code/project2
pip install -r requirements.txt

# 参考实现的测试应该全部通过（约 10 秒）
QUANTLAB_IMPL=solution python -m pytest -q tests

# 不设环境变量时测试 starter，现在会全部失败
python -m pytest -q tests
```

测试用的小模型由 `tests/conftest.py` 现场生成，不需要下载。第 7 步的模型对比实验需要一个训练过的小模型，可以现在就开始训练，和前几步并行：

```bash
python tools/make_corpus.py --out data
python tools/train_tiny_lm.py --data data --out /tmp/tiny-lm    # 两核 CPU 约 6 分钟
```

随机权重的模型没有意义：所有方法的困惑度都接近词表大小，看不出差别。所以这里用本站的文档和 Python 标准库的 docstring 训练一个 4 层、隐藏维度 256、词表 2048 的模型，测试集困惑度约 38。

## 3. 分步完成

每一步完成后运行对应的测试，通过了再进入下一步。

| 步骤 | 内容 | 验证 |
|---|---|---|
| 第 1 步 | 整数量化与反量化 | `python -m pytest tests/test_steps.py -k step01` |
| 第 2 步 | FP8 伪量化 | `-k step02` |
| 第 3 步 | GPTQ | `-k step03` |
| 第 4 步 | AWQ 的 α 搜索 | `-k step04` |
| 第 5 步 | SmoothQuant 与 W8A8 | `-k step05` |
| 第 6 步 | KV Cache 量化 | `-k step06`，之后 `tests/test_model.py` 全部通过 |
| 第 7 步 | 在小模型上比较所有方法 | `run.py` |
| 第 8 步 | GPU 上的真实模型实验 | vLLM、lm-eval |

### 第 1 步：整数量化

`int_qparams(x, bits, symmetric)` 沿最后一维计算 scale 和零点，`int_quant_dequant` 用它们量化再反量化。docstring 里给出了公式。`fake_quant_int` 已经写好，它先用 `_group` 把最后一维切成组，再调用这两个函数。

手算一个例子。`x = [0.2, -0.6, 1.0, 0.1]`，4 位：

- 对称：$q_{\max} = 7$，scale $= 1.0 / 7 \approx 0.1429$。$x / \text{scale} = [1.4, -4.2, 7, 0.7]$，四舍五入为 $[1, -4, 7, 1]$，反量化得 $[0.1429, -0.5714, 1.0, 0.1429]$。
- 非对称：scale $= (1.0 - (-0.6)) / 15 \approx 0.1067$，零点 $= \text{round}(0.6 / 0.1067) = \text{round}(5.625) = 6$。量化值为 $[8, 0, 15, 7]$，反量化得 $[0.2133, -0.64, 0.96, 0.1067]$。

注意非对称的结果里，最小值和最大值都没有精确还原。零点必须是整数，四舍五入之后整个网格平移了一点。

<details>
<summary>提示 1：形状</summary>

`x` 的形状是 `[..., n_groups, group_size]`。`amax`、`amin` 都沿 `dim=-1` 取，加 `keepdim=True`，这样 scale 的形状是 `[..., n_groups, 1]`，可以直接和 `x` 做广播运算。
</details>

<details>
<summary>提示 2：截断的范围</summary>

对称量化截断到 $[-2^{b-1}, 2^{b-1} - 1]$，非对称截断到 $[0, 2^b - 1]$。反量化是 $(q - \text{zero}) \times \text{scale}$，对称时 zero 为 0。
</details>

测试检查三件事：8 位对称量化的往返误差足够小；非对称量化用满了 0 到 $2^b - 1$ 的整个范围；分组越小误差越小。

### 第 2 步：FP8

`fake_quant_fp8(x, granularity, fmt)` 先求 amax（`"tensor"` 整个张量一个，`"row"` 每行一个），scale $= \text{amax} / 448$，然后把 $x / \text{scale}$ 转换成 PyTorch 的 `torch.float8_e4m3fn`，再转回 float32 乘回 scale。PyTorch 的类型转换会按 E4M3 的规则舍入，我们不需要自己实现浮点舍入。

一个例子：按张量量化 `[0.3, 1.7, 100, 448]`，amax 恰好是 448，scale 为 1，结果是 `[0.3125, 1.75, 96, 448]`。在 1 附近间隔是 0.125，在 64 到 128 之间间隔是 8，相对误差大致相同（2.1 节）。

<details>
<summary>提示：amax 为 0 的行</summary>

全零的行 scale 会是 0，除法得到 NaN。像第 1 步一样给 amax 加一个很小的下限。
</details>

### 第 3 步：GPTQ

`gptq_quantize` 的框架已经写好：处理从未激活的通道、act-order 的重排、阻尼、Cholesky 分解得到 $H^{-1}$ 的上三角因子、按块循环，以及块与块之间的批量补偿。要完成的是**块内**的逐列循环，对应 2.2 节第 4 节伪代码的内层：

```text
for i in range(块的宽度):
    w = W1[:, i]
    q = 用当前组的 scale、zero 量化再反量化 w
    Q[:, i1 + i] = q
    err = (w - q) / Hinv1[i, i]
    W1[:, i:] -= err ⊗ Hinv1[i, i:]
    Err1[:, i] = err
```

<details>
<summary>提示 1：量化一列</summary>

`int_quant_dequant` 沿最后一维广播 scale。一列 `w` 的形状是 `[out]`，scale 的形状是 `[out, 1]`。可以把 `w` 变成 `[out, 1]` 再调用，或者把 scale 压成 `[out]`。
</details>

<details>
<summary>提示 2：外积</summary>

`err` 的形状是 `[out]`，`Hinv1[i, i:]` 的形状是 `[块宽 - i]`。外积可以写成 `err[:, None] * Hinv1[i, i:][None, :]`，也可以用 `torch.outer`。`W1[:, i:]` 包含第 $i$ 列本身，更新后它会变成 `q`，这没有关系，下一轮不会再用到它。
</details>

<details>
<summary>提示 3：为什么用上三角的 Cholesky 因子</summary>

GPTQ 需要的是"量化了前 $i$ 列之后，剩余列的 $H^{-1}$"。2.2 节说明了，这些逐步更新的矩阵的第一行，恰好就是 $H^{-1}$ 的 Cholesky 上三角因子的对应行。所以循环里直接读 `Hinv1[i, i:]`，不需要在循环中更新任何矩阵。
</details>

测试检查 GPTQ 的输出误差比 RTN 小，并且每个量化后的值都落在它所在组的量化网格上。后者能发现一类常见的错误：把误差补偿加到了已经量化过的列上。

### 第 4 步：AWQ

`awq_search(weight, x, bits, group_size, n_grid)` 对 $\alpha = 0, 1/n, 2/n, \dots$ 逐个尝试：

1. $s = \overline{|x|}^{\alpha}$，按输入通道计算，再除以 $\sqrt{\max(s) \cdot \min(s)}$ 归一化；
2. 等效权重 $= Q(W \cdot s) / s$，$Q$ 是 `fake_quant_int`；
3. 用校准输入 `x` 计算输出误差 $\|x W_{\text{eq}}^\top - x W^\top\|^2$。

返回误差最小的等效权重、对应的 $\alpha$，以及每个 $\alpha$ 的误差列表。

<details>
<summary>提示：为什么要归一化</summary>

数学上，$s$ 整体乘以一个常数 $c$，结果完全不变：对称量化时，每组的 scale 也随之乘以 $c$，舍入的结果一模一样，$Q(cWs) / (cs) = Q(Ws)/s$。归一化是为了数值稳定。真实的实现用 FP16 计算，$\overline{|x|}^{\alpha}$ 可能非常大或非常小，归一化让 $s$ 分布在 1 的两侧，避免溢出。官方实现也是这样做的。
</details>

测试检查 AWQ 的误差不大于 RTN。$\alpha = 0$ 时 $s$ 全为 1，就是 RTN，所以只要 $\alpha = 0$ 在搜索范围内，这一点就一定成立。

### 第 5 步：SmoothQuant 与 W8A8

`smooth_scales(act_absmax, weight, alpha)` 实现 $s_j = \max|X_j|^{\alpha} / \max|W_j|^{1-\alpha}$。权重的 $\max|W_j|$ 是第 $j$ 个**输入**通道在所有输出通道上的最大值，也就是 `weight` 沿 `dim=0` 取最大值。

例子（测试中的第一个用例）：激活每个通道的最大值是 $[16, 1]$，权重是 $[[1, 4]]$，$\alpha = 0.5$，得到 $s = [\sqrt{16}/\sqrt{1}, \sqrt{1}/\sqrt{4}] = [4, 0.5]$。

`w8a8_matmul(x, W, act, fmt)` 是伪量化的 W8A8 矩阵乘：权重每个输出通道一个 scale，激活按张量或按 token 一个 scale，然后相乘。

<details>
<summary>提示：按张量量化激活</summary>

`fake_quant_int` 总是沿最后一维分组。要让整个张量共用一个 scale，可以先 `reshape(1, -1)`，量化后再变回原来的形状。
</details>

测试用一个带离群通道的层检查：平滑之后，按张量量化的误差降到原来的 30% 以下；按 token 量化的误差小于按张量量化。

### 第 6 步：KV Cache 量化

`fake_quant_kv(t, bits, axis, fmt)` 的输入形状是 `[tokens, heads, head_dim]`。按 token 量化时，每个 token 的每个头的 `head_dim` 个数共用一个 scale，直接调用 `fake_quant_int` 即可。按通道量化时，每个头的每个通道在所有 token 上共用一个 scale，要先把 tokens 换到最后一维。整数用非对称量化（2.5 节）。

测试构造了一个第 5 个通道放大 20 倍的 K，检查按通道量化的误差小于按 token。

完成这一步之后，运行全部测试：

```bash
python -m pytest -q tests
```

`test_model.py` 在随机权重的小模型上把每种方法都跑一遍，检查 8 位量化后的输出与原模型接近，以及 KV Cache 量化的钩子能正常工作。

### 第 7 步：在小模型上比较

用训练好的小模型，对每种方法运行 `run.py`：

```bash
python run.py --model /tmp/tiny-lm --data data --method rtn  --bits 4
python run.py --model /tmp/tiny-lm --data data --method gptq --bits 3 --act-order
python run.py --model /tmp/tiny-lm --data data --method smoothquant --act tensor
python run.py --model /tmp/tiny-lm --data data --method none --kv-bits 2 --k-axis channel
```

把结果整理成一张表。下面是参考实现的一次运行（全精度 37.98），自己的结果应该在它附近：

| 方法 | 设置 | 困惑度 |
|---|---|---|
| RTN | W4 g128 | 38.20 |
| GPTQ | W4 g128 | 38.13 |
| AWQ | W4 g128 | 38.19 |
| RTN | W3 g128 | 39.71 |
| GPTQ | W3 g128 | 39.04 |
| GPTQ + act-order | W3 g128 | 38.69 |
| AWQ | W3 g128 | 39.76 |
| RTN | W2 g32 | 60.46 |
| GPTQ | W2 g32 | 50.05 |
| AWQ | W2 g32 | 60.37 |
| INT8 W8A8 | 激活按张量 | 38.13 |
| INT8 W8A8 | 激活按 token | 38.00 |
| SmoothQuant | α = 0.5，激活按张量 | 38.12 |
| FP8 W8A8 | 激活按 token | 38.02 |
| KV Cache | 4 位，K 按通道 | 38.20 |
| KV Cache | 2 位，K 按通道 | 46.04 |

然后用 `tools/outlier_layer.py` 在带离群通道的层上再比较一次（2.3、2.4 节有它的结果）。

最后写一段分析，回答三个问题：

1. 小模型上哪些方法拉开了差距，哪些没有？
2. AWQ 和 SmoothQuant 在小模型上几乎没有收益，在 `outlier_layer.py` 中却效果明显。原因是什么？可以用钩子统计小模型每层输入的各通道最大值来佐证。
3. 根据这些结果，如果要把 Qwen3-8B 部署到一张 L40S 上，你会先尝试哪两种方案？

### 第 8 步：GPU 上的真实模型

这一步需要一张 24 GB 以上的显卡。Qwen 官方发布了 Qwen3-8B 的 AWQ 和 FP8 版本，可以直接使用。

**显存与 KV Cache 容量。** 分别启动三个模型，记录启动日志中的权重显存和 KV Cache 容量：

```bash
vllm serve Qwen/Qwen3-8B     --max-model-len 8192
vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 8192
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 8192
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 8192 --kv-cache-dtype fp8
```

**性能。** 对每个配置，用 2.7 节第 5 节的脚本在并发 1、4、16、64、128 下压测，画出输出吞吐和 TPOT 随并发变化的曲线，找出 AWQ 和 FP8 的交叉点。用 2.6 节的组件解释交叉点出现的位置。确认日志中的 kernel：AWQ 应该是 Marlin 或 Machete，FP8 应该是 CUTLASS 的 FP8 GEMM（Ampere 上是 Marlin 的 W8A16）。

**精度。** 用 lm-eval 在 GSM8K 上比较三个模型：

```bash
lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-8B-AWQ,max_model_len=4096 \
  --tasks gsm8k --batch_size auto
```

计算恢复率。有余力的话，统计两个模型答案不同的题目比例（2.7 节的 flips）。

**自己量化。** 用 2.2 节的 LLM Compressor 脚本对 Qwen3-8B 做 W4A16 GPTQ，和官方的 AWQ 版本比较精度和性能。也可以直接用本项目的 `run.py` 对 Qwen3-0.6B 做伪量化（`--model` 指向模型目录，加 `--device cuda`），观察真实模型上 AWQ 与 RTN 的差距是否比小模型大。

## 4. 完成标准

- `python -m pytest -q tests` 全部通过。
- 第 7 步的结果表和三个问题的分析。
- 有 GPU 时：第 8 步的显存、性能曲线和精度对比，整理成一份简短的报告，最后给出你对"Qwen3-8B 在这张卡上该用哪种方案"的结论。

## 5. 扩展练习

- **AWQ 的截断搜索。** 2.3 节提到 AWQ 还会搜索每组的截断比例。在 `awq.py` 中加上这一步，看 3 位时能改善多少。
- **SparseGPT。** 把 `gptq.py` 改成剪枝：把"量化到最近的等级"换成"按 2:4 的模式置零"，其余的误差补偿不变。与 `tools/prune_24.py` 中不做补偿的结果比较。
- **KL 散度评测。** 在 `eval.py` 中增加一个函数，计算量化模型与原模型在每个位置的输出分布的 KL 散度。比较它和困惑度的差别，哪个对小的误差更敏感？
- **旋转。** 在一个线性层上实现 2.4 节提到的旋转：给激活右乘一个随机正交矩阵（如归一化的 Hadamard 矩阵），权重同时左乘它的转置。比较旋转前后 W4A4 的误差。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "4 位非对称量化 &#36;[0.2, -0.6, 1.0, 0.1]&#36;，反量化后最大值 1.0 变成了 0.96。原因是：", "options": ["实现有错误", "零点必须是整数，round(5.625) = 6 让整个量化网格平移了一点", "4 位只能表示 0 到 0.96", "非对称量化不能表示正数"], "answer": 1, "explain": "scale 是按 min 和 max 精确计算的，但零点要四舍五入成整数，网格整体移动了不到半个间隔，端点也就无法精确还原。对称量化没有这个问题，但浪费了一侧的范围。"}, {"q": "GPTQ 的测试检查每个输出值都落在所在组的量化网格上。这能发现哪类错误？", "options": ["Hessian 计算错误", "误差补偿被加到了已经量化过的列上，使它们离开了网格", "分组大小设置错误", "校准数据太少"], "answer": 1, "explain": "补偿只应该作用于还没有量化的列。如果更新范围写错，已经量化的列会被改动，不再是某个等级上的值，这种模型无法真正以 4 位存储。"}, {"q": "第 7 步中，AWQ 和 SmoothQuant 在小模型上几乎没有收益，在 outlier_layer.py 中效果明显。最合理的解释是：", "options": ["小模型的层数太少", "两者都依赖激活中的离群通道；小模型的激活各通道幅度相近，没有可以利用的结构", "小模型用的是 FP32", "校准数据和测试数据来自不同的分布"], "answer": 1, "explain": "AWQ 保护大激活通道上的权重，SmoothQuant 把大激活通道的难度转给权重，两者都以离群通道的存在为前提。用钩子统计各通道的最大值，小模型多数线性层的最大通道只是中位数的两三倍，down_proj 稍高，也不到 10 倍。"}]'></div>
