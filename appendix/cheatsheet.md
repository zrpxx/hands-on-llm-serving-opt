# 常用估算公式

> 设：层数 $L$，隐藏维度 $d$，KV 头数 $n_{kv}$，头维度 $d_h$，序列长度 $s$，batch $b$，参数量 $P$，每元素字节数 $B$。

## 显存

**权重**
$$M_{\text{weights}} \approx P \times B$$
（BF16：$B=2$；INT8/FP8：$B=1$；INT4：$B\approx0.5$ + scale 开销）

**KV Cache**
$$M_{\text{KV}} = 2 \times L \times n_{kv} \times d_h \times s \times b \times B$$
（系数 2 = K 和 V；GQA 下 $n_{kv}$ 远小于 Q 头数，这就是它省显存的原因）

**能塞下的最大 token 数**
$$T_{\max} \approx \frac{M_{\text{GPU}} \times u - M_{\text{weights}} - M_{\text{act}}}{2 L n_{kv} d_h B}$$
（$u$ 为 `gpu_memory_utilization`，vLLM 默认 0.9）

## 算力与带宽

**前向 FLOPs（每 token）**
$$\text{FLOPs} \approx 2P$$

**算术强度**
$$I = \frac{\text{FLOPs}}{\text{Bytes moved}}$$
硬件拐点 $I^* = \dfrac{\text{峰值 FLOPS}}{\text{峰值带宽}}$。$I < I^*$ → 带宽瓶颈；$I > I^*$ → 算力瓶颈。

**Decode 每步延迟下限（带宽瓶颈）**
$$t_{\text{decode}} \gtrsim \frac{M_{\text{weights}} + M_{\text{KV}}}{\text{HBM 带宽}}$$

**Prefill 延迟下限（算力瓶颈）**
$$t_{\text{prefill}} \gtrsim \frac{2P \times s}{\text{峰值 FLOPS} \times \text{MFU}}$$

## 投机采样

接受率 $\alpha$（假设各位置独立同分布），投机长度 $k$，每轮期望产出 token 数：
$$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{k+1}}{1-\alpha}$$

## 例：Qwen3-8B @ BF16 单卡 80GB

| 量 | 估算 |
|---|---|
| 权重 | 8B × 2 B ≈ 16 GB |
| 每 token KV（36 层，8 KV 头，d_h=128） | 2×36×8×128×2 B ≈ 144 KB |
| 剩余 ~55 GB 可存 token 数 | ≈ 37 万 token |

> [!WARNING]
> 以上为数量级估算，用前请对照模型 `config.json` 核实层数与头数。
