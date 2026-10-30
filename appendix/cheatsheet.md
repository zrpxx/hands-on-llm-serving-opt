# 常用估算公式

> 记号：层数 $L$，隐藏维度 $d$，KV 头数 $n_{kv}$，头维度 $d_h$，参数量 $P$，每个元素的字节数 $B$（BF16 为 2，FP8 为 1，INT4 约 0.5 加上缩放因子），序列长度 $s$，batch $b$。所有估算都只到数量级，用之前对照模型的 `config.json` 核实层数和头数。

## 显存

**权重**（0.6 节）

$$M_{\text{weights}} \approx P \times B$$

**每个 token 的 KV**（0.4、0.6 节）

$$m_{\text{KV}} = 2 \times L \times n_{kv} \times d_h \times B$$

系数 2 是 K 和 V。MLA 的每个 token 每层只存一个压缩向量，$m_{\text{KV}} = L \times (d_c + d_r) \times B$，DeepSeek-V3 中 $d_c + d_r = 512 + 64$（3.1 节）。

**能放下多少 token 的 KV**

$$T_{\max} \approx \frac{M_{\text{GPU}} \times u - M_{\text{weights}} - M_{\text{其他}}}{m_{\text{KV}}}$$

$u$ 是 vLLM 的 `gpu_memory_utilization`，默认 0.9；$M_{\text{其他}}$ 包括激活、CUDA Graph 等，通常留 1～3 GB。

## 耗时

**算术强度与平衡点**（0.7 节）

$$I = \frac{\text{浮点运算次数}}{\text{读写的字节数}}, \qquad I^* = \frac{\text{峰值算力}}{\text{峰值带宽}}$$

$I < I^*$ 时受带宽限制，否则受算力限制。H100 的 BF16 平衡点约为 300 FLOP/字节。

**decode 一步的下限**：受带宽限制时

$$t_{\text{decode}} \gtrsim \frac{M_{\text{weights}} + b \cdot s \cdot m_{\text{KV}}}{\text{带宽}}$$

batch 增大到 $2P \cdot b / \text{算力} > M_{\text{weights}} / \text{带宽}$ 时，线性层转为受算力限制。

**prefill 的下限**：受算力限制时

$$t_{\text{prefill}} \gtrsim \frac{2P \times s}{\text{峰值算力} \times \text{MFU}}$$

长输入时还要加上注意力的计算，它与 $s^2$ 成正比。

**端侧 batch 为 1 的 decode 速度上限**（5.5 节）

$$\text{token/s} \lesssim \frac{\text{内存带宽}}{M_{\text{weights}}}$$

## 吞吐与并发

**Little 定律**（5.1 节）

$$\text{在途请求数} = \text{到达速率} \times \text{平均每个请求的耗时}$$

并发上限固定时，吞吐 = 并发 ÷ 每个请求的耗时，只能靠降低延迟提高。

**每个请求的耗时**

$$t_{\text{req}} \approx \text{TTFT} + (n_{\text{out}} - 1) \times \text{TPOT}$$

## 量化

**每个元素的位数**（2.1、6.2 节）

$$\text{位数} = \text{元素位数} + \frac{\text{缩放因子位数}}{\text{组大小}}$$

INT4 每 128 个一组、FP16 缩放：4.125 位；MXFP4：4.25 位；NVFP4：4.5 位。

**信噪比**：SQNR 每提高 6 dB，误差的幅度约减半。

## 投机解码

**每轮的期望产出**（3.4 节）：各位置的接受率都为 $\alpha$、草稿长度为 $K$ 时

$$E[\text{token}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

**加速比**：草稿每个 token 的开销是目标模型一次前向的 $c$ 倍时

$$S = \frac{E[\text{token}]}{1 + cK}$$

batch 大到受算力限制时，验证 $K+1$ 个 token 的耗时不再与验证 1 个相同，加速比会降到 1 以下（6.5 节）。

## 分布式

**张量并行的通信**（4.1 节）：每层两次 all-reduce，每次每张卡收发约

$$2 \times \frac{N-1}{N} \times b \times d \times B \text{ 字节}$$

**MoE 每个专家分到的 token**（4.2、6.3 节）：全局 batch 为 $b$，每个 token 选 $k$ 个，共 $E$ 个专家时为 $bk/E$。这也约等于 MoE 层的算术强度。

**PD 分离的 KV 传输**（4.3 节）

$$t_{\text{transfer}} \approx \frac{s \times m_{\text{KV}}}{\text{链路带宽}}$$

逐层传输时，大部分可以与 prefill 重叠，暴露在外的只有最后一层。

**上下文并行的负载**（6.6 节）：因果注意力中位置 $i$ 的计算量正比于 $i$。按顺序切成 $N$ 段时，最忙的卡约为平均的 $\frac{2N-1}{N}$ 倍；首尾配对切分后接近 1。

## 例：Qwen3-8B，BF16，一张 80 GB 的 H100

| 量 | 估算 |
|---|---|
| 权重 | 8.2B × 2 字节 ≈ 16 GB |
| 每个 token 的 KV（36 层，8 个 KV 头，头维度 128） | 2 × 36 × 8 × 128 × 2 字节 = 144 KB |
| 能放下的 KV | (80 × 0.9 − 16 − 2) GB ÷ 144 KB ≈ 37 万个 token |
| batch 为 1 时 decode 的下限 | 16 GB ÷ 3.35 TB/s ≈ 5 ms，即每秒约 200 个 token |
| 线性层转为受算力限制的 batch | 约 300 个 token 的 batch（按峰值算力；按实际效率约为 200） |
| 4K 输入的 prefill | 2 × 8.2B × 4096 ÷ (989 TFLOPS × 0.6) ≈ 113 ms |
