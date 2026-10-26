# 0.3 显存估算：权重与 KV Cache

> 对应书中第 5 章 Bottlenecks in LLM Model Loading

## 学习目标

- 根据参数量和精度估算权重显存，根据 `config.json` 估算每个 token 的 KV Cache 大小。
- 算出一张卡在给定上下文长度下能同时服务多少请求，并和 vLLM 启动日志对照验证。
- 解释 GQA、FP8 KV、张量并行分别如何改变这个数字。

## 1. 为什么权重必须放在显存里

推理时每生成一个 token 都要把全部权重读一遍（见 0.4）。几种存储介质的带宽差距（书中表 5-5）：

| 介质 | 带宽 |
|---|---|
| SSD | 0.5–14 GB/s |
| CPU 内存（DRAM） | 50–200 GB/s |
| GPU 显存（HBM / GDDR） | 0.3–3+ TB/s（H200 为 4.8 TB/s） |

如果权重放在 CPU 内存里按需搬运，每个 token 的延迟会增加一到两个数量级。所以加载流程是"磁盘 → CPU 内存 → 显存"，之后权重常驻显存。

## 2. 显存的三个组成部分

$$
M_{\text{GPU}} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{other}}
$$

- **权重**：加载后固定不变。
- **KV Cache**：随着生成的进行线性增长，是决定并发和上下文长度的部分。
- **其他**：前向计算的中间激活、CUDA Graph、通信缓冲区、框架自身开销等，一般为几 GB。

## 3. 权重

$$
M_{\text{weights}} = P \times b_w
$$

$P$ 是参数量，$b_w$ 是每个参数的字节数：

| 精度 | 字节 / 参数 |
|---|---|
| FP32 | 4 |
| FP16 / BF16 | 2 |
| FP8 / INT8 | 1 |
| INT4 | 0.5（另加少量缩放因子） |

精度可以在 `config.json` 的 `torch_dtype` 字段里看到。书中的例子：Llama-2-7B，FP16，约 $7 \times 10^9 \times 2 = 14$ GB，与 Hugging Face 上权重文件合计约 13 GB 一致（实际参数量 6.74B）。

> [!NOTE]
> 模型名里的参数量是取整后的值。Qwen3-8B 实际约 8.19B，BF16 权重 16.4 GB。精确计算时以权重文件大小或逐层累加为准。

## 4. KV Cache

### 4.1 每个 token 的大小

$$
\text{KV}_{\text{token}} = 2 \times L \times n_{kv} \times d_h \times b_{kv}
$$

| 符号 | 含义 | `config.json` 字段 |
|---|---|---|
| 2 | K 和 V 各一份 | |
| $L$ | 层数 | `num_hidden_layers` |
| $n_{kv}$ | KV 头数 | `num_key_value_heads` |
| $d_h$ | 每个头的维度 | `head_dim`，没有时为 `hidden_size / num_attention_heads` |
| $b_{kv}$ | KV 的字节数 | 默认与模型精度相同；FP8 KV 为 1 |

> [!WARNING]
> 书中的公式写的是"number of attention heads"，这只对多头注意力（MHA）成立。对使用 GQA 的模型（Llama-3、Qwen2/3 等），应该用 KV 头数 $n_{kv}$，它通常只有 Q 头数的 1/4 到 1/8。用 Q 头数会把 KV Cache 高估数倍。

**两个例子：**

| 模型 | $L$ | Q 头 / KV 头 | $d_h$ | KV / token（BF16） |
|---|---|---|---|---|
| Llama-2-7B（MHA） | 32 | 32 / 32 | 128 | $2 \times 32 \times 32 \times 128 \times 2 = 512$ KiB |
| Qwen3-8B（GQA） | 36 | 32 / 8 | 128 | $2 \times 36 \times 8 \times 128 \times 2 = 144$ KiB |

两个模型参数量接近，但 Qwen3-8B 每个 token 的 KV 只有 Llama-2-7B 的 28%。这就是 GQA 的作用，3.1 节会详细讨论。

### 4.2 总量

$$
M_{\text{KV}} = \text{KV}_{\text{token}} \times \sum_{i \in \text{batch}} (\text{输入长度}_i + \text{已生成长度}_i)
$$

最坏情况下每个请求都用满最大长度，总量就是 $\text{KV}_{\text{token}} \times B \times S_{\max}$。

书中的例子：Llama-2-7B，最大长度 4096，批大小 16，KV Cache 需要 $0.5\,\text{MB} \times 4096 \times 16 = 32$ GB，超过了权重本身的 14 GB。

### 4.3 反过来算：一张卡能放多少请求

$$
T_{\max} = \frac{M_{\text{GPU}} \times u - M_{\text{weights}} - M_{\text{other}}}{\text{KV}_{\text{token}}}, \qquad
B_{\max} = \frac{T_{\max}}{S_{\max}}
$$

$u$ 对应 vLLM 的 `--gpu-memory-utilization`，默认 0.9。

书中表 5-7 的对比（Llama-2-7B，4096 token / 请求，不计其他开销）：

| GPU | 扣除权重后剩余 | 最大批大小 |
|---|---|---|
| A10 24 GB | 10 GB | 4–5 |
| L40S 48 GB | 34 GB | 16–17 |

L40S 单价更高，但能服务约 4 倍的并发，单位成本反而更低。这是"显存大小直接影响单位成本"的一个直观例子。

**本笔记的例子：Qwen3-8B，BF16，单张 80 GB 卡（A100 / A800 / H100）。**

$$
T_{\max} \approx \frac{80 \times 0.9 - 16.4 - 2}{144\,\text{KiB}} \approx 3.6 \times 10^5 \text{ token}
$$

如果每个请求最多 4096 token，可以同时放约 88 个；如果按 32K 上下文配置，只能放约 11 个。

## 5. 改变容量的几种手段

| 手段 | 对 $T_{\max}$ 的影响 | 章节 |
|---|---|---|
| 权重量化（BF16 → INT4） | 权重减到约 1/4，剩余空间增加 | 第 2 部分 |
| KV 量化（BF16 → FP8） | 每 token KV 减半，容量翻倍 | 2.5 |
| GQA / MQA / MLA | 模型结构层面减少 KV | 3.1 |
| 张量并行 TP=2 | 权重和 KV 都切到两张卡，每卡 KV 减半 | 4.1 |
| 分页管理（PagedAttention） | 不改变每 token 大小，但消除预分配和碎片造成的浪费 | 1.6 |
| 前缀缓存 | 共享前缀只存一份 | 1.7 |

书中给出的经验值：**显存至少按权重大小的两倍来准备**，才能留出足够的 KV Cache 空间来发挥并行能力。

## 6. 显存随时间的变化

服务空闲时显存里只有权重（以及框架预分配但尚未使用的 KV 池）。请求进来后，prefill 一次性写入输入部分的 KV，decode 每生成一个 token 再追加一份。峰值出现在生成结束前，所以容量规划要按"输入 + 最大输出"来算，否则长输出时会触发显存不足或抢占（1.6 节）。

## 动手实验

**实验 1：用脚本估算。** 仓库里的 [`code/stage0/estimate_memory.py`](https://github.com/zrpxx/hands-on-llm-serving-opt/blob/main/code/stage0/estimate_memory.py) 读取 `config.json` 并输出上面的各项：

```bash
python code/stage0/estimate_memory.py code/stage0/configs/qwen3-8b.json --gpu-mem 80 --max-len 4096
```

```text
参数量            8.19 B
权重（每卡）      16.4 GB   (2 B/参数, tp=1)
KV / token（每卡）144 KiB  (2 B/元素)
KV 预算（每卡）   53.6 GB = 80.0×0.9 - 权重 - 2.0 预留
可缓存 token 数   363,624
并发上限          88.8 个请求（每个 4096 token 用满时）
```

**实验 2：与 vLLM 对照。** 用同一个模型启动 vLLM，从日志中读出 `GPU KV cache size: N tokens`，和估算值比较：

```bash
vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.9 2>&1 | grep -E "Model loading|KV cache size|Maximum concurrency"
```

| 项 | 估算 | vLLM 日志 | 误差 |
|---|---|---|---|
| 权重 | 16.4 GB | | |
| KV 容量（token） | 363,624 | | |
| 4096 token 并发 | 88.8 | | |

误差主要来自"其他"开销的估计。把 `--overhead` 调到让两者对齐，记下这台机器上的实际值，后面的估算就更准了。

**实验 3：改变变量。** 依次加上 `--kv-cache-dtype fp8`、`--tensor-parallel-size 2`，观察日志中的 KV 容量变化是否符合第 5 节的预期。

## 自测题

1. Llama-3-8B 有 32 层、32 个 Q 头、8 个 KV 头、$d_h = 128$。BF16 下每个 token 的 KV 是多少？如果误用 Q 头数计算，会得到多少？
2. 一张 24 GB 的卡跑 Qwen3-8B BF16，最大上下文 8K，能同时服务几个请求？（其他开销按 2 GB 算，$u = 0.9$）
3. 为什么 vLLM 启动时报告的 KV 容量，常常比"总显存减权重"算出来的少几 GB？
4. 把 KV Cache 从 BF16 改为 FP8，对 decode 速度有什么影响？

<details>
<summary>参考答案</summary>

1. $2 \times 32 \times 8 \times 128 \times 2 = 131072$ 字节 = 128 KiB。误用 32 个头会得到 512 KiB，高估 4 倍。
2. 预算 $24 \times 0.9 - 16.4 - 2 = 3.2$ GB，可缓存约 $3.2 \times 10^9 / 147456 \approx 21{,}700$ token，8K 上下文下约 2 个请求。这张卡几乎只够跑单请求，需要量化或换卡。
3. $u = 0.9$ 预留了 10%；vLLM 在启动时会做一次 profiling 前向，测出激活峰值并扣除；CUDA Graph 捕获和 NCCL 缓冲区也会占用显存。
4. 容量翻倍，可以跑更大的批；同时 decode 时读取 KV 的字节数减半，在长上下文下（KV 读取占比大）会直接降低每步延迟。代价是可能的精度损失，2.5 节讨论。

</details>

## 延伸阅读

- 本笔记附录 [常用估算公式](appendix/cheatsheet.md)
- vLLM 文档中关于 `gpu_memory_utilization` 与 KV Cache 分配的说明
