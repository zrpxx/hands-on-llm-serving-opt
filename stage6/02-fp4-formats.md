# 6.2 NVFP4 与 MXFP4

> 书外内容。原书第 5 章讨论了 INT8、INT4、FP8，提到 Blackwell 开始支持 FP4，但没有展开。本节的误差实验和格式对比是书外内容。

2.4 节讲过，Hopper 的 FP8 Tensor Core 让 W8A8 成为数据中心推理的默认选择。Blackwell 又往下走了一步：Tensor Core 原生支持 **4 位浮点**，FP4 矩阵乘法的峰值算力是 FP8 的两倍。与 INT4 的 weight-only 量化（2.2、2.3 节）不同，这里权重和激活都是 4 位，乘法本身就在 4 位上完成，不需要先反量化。

问题是 4 位太少了。一个 4 位数只有 16 种取值，直接把整个张量缩放到这 16 个值上，误差无法接受。NVFP4 和 MXFP4 的答案相同：**很小的块，每块一个缩放因子**。区别在块有多大、缩放因子用什么格式。

## 1. 元素：FP4 E2M1

两种格式的元素都是 E2M1：1 位符号，2 位指数，1 位尾数。它能表示的非负值只有 8 个：

$$
0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6
$$

和 INT4 的等间距网格（$-8 \sim 7$）相比，E2M1 在 0 附近更密、在远处更疏，与神经网络中权重和激活的分布（大部分值集中在 0 附近）更匹配。但它的动态范围只有 $6 / 0.5 = 12$ 倍，必须靠缩放因子把每一块的数值搬到这个范围里。

## 2. 两种缩放方式

| | MXFP4 | NVFP4 |
|---|---|---|
| 提出者 | OCP Microscaling（MX）规范，2023 | NVIDIA，随 Blackwell 推出 |
| 块大小 | 32 | 16 |
| 块缩放因子 | E8M0：只有指数，只能是 2 的幂 | FP8 E4M3：有 3 位尾数 |
| 第二层缩放 | 无 | 每个张量一个 FP32 |
| 每元素位数 | $4 + 8/32 = 4.25$ | $4 + 8/16 = 4.5$ |
| 硬件 | Blackwell、AMD MI355X 等 | Blackwell |

MXFP4 的缩放因子只能是 2 的幂。块内绝对值最大的元素是 $a$ 时，缩放因子取 $2^{\lfloor \log_2 a \rfloor - 2}$（E2M1 的最大值 $6 = 1.5 \times 2^2$）。这意味着缩放后的最大值落在 $[4, 8)$ 之间：比 6 大的会被截断，比 6 小很多的又浪费了一部分范围。平均下来，相当于损失了将近一位的精度。

NVFP4 用 E4M3 表示缩放因子，可以精确到 $a / 6$ 附近，几乎不浪费范围。但 E4M3 本身的范围有限（最大 448），所以再加一个张量级的 FP32 缩放，先把整个张量缩放到一个合适的区间。块更小（16 个），一个离群值只会影响 15 个邻居。

## 3. 误差有多大

`code/stage6/fp4_formats.py` 用 NumPy 模拟了这几种格式，在三种数据上比较信噪比（SQNR，越大越好）：

```bash
python code/stage6/fp4_formats.py
```

```text
每个元素占用的位数（含缩放因子）：
  INT4 g128  4.125
  INT4 g16   5.000
  MXFP4      4.250
  NVFP4      4.500

数据                         INT4 g128    INT4 g16       MXFP4       NVFP4
权重（正态）                    18.6 dB      21.4 dB      18.9 dB      20.4 dB
激活（1% 通道放大 30 倍）        12.8 dB      20.5 dB      15.8 dB      21.8 dB
重尾（t 分布，自由度 3）          13.0 dB      19.0 dB      17.0 dB      20.9 dB
```

几个观察：

- **块大小是主要因素**。同样是 INT4，每组 128 个和每组 16 个，在有离群值的激活上差了近 8 dB。这和 2.1 节分组量化的结论一致。
- **在正态分布上，E2M1 并不比 INT4 好**：同样每组 16 个，INT4 g16 的 SQNR 反而略高。E2M1 的优势在重尾和离群值的情况下才显现出来。
- **NVFP4 的缩放因子更省**。INT4 g16 用 FP16 作缩放因子，每元素 5 位；NVFP4 用 FP8，每元素 4.5 位，误差相近甚至更好。
- **MXFP4 因为 2 的幂的缩放，比同等条件下损失明显**。

下面的组件可以调节离群通道的强度和比例：

<div class="llm-widget" data-widget="fp4-block"></div>

## 4. 在推理中怎样使用

FP4 的收益来自三个方面，对应推理的不同瓶颈：

- **算力**：FP4 矩阵乘法的峰值是 FP8 的两倍，对 prefill 和大 batch 的 decode 有效。
- **显存和带宽**：权重是 FP8 的一半，对小 batch 的 decode 有效，也让更大的模型能放进更少的卡。
- **通信**：MoE 的 dispatch 前把激活量化成 FP4 再发送，all-to-all 的数据量是 BF16 的四分之一左右（6.3 节）。

vLLM 在 GB200 上部署 DeepSeek-R1 时的做法很有代表性：MoE 专家的矩阵乘法和注意力的输出投影用 NVFP4；对量化更敏感的 MLA 的 Q 投影仍然用 FP8；MoE dispatch 前把激活量化到 FP4；把 RoPE、量化和写入 Q 融合成一个 kernel，省掉中间的读写。**哪些层用 FP4、哪些保留更高的精度**，是需要逐层评估的（2.7 节）。

开源模型也开始直接以 4 位格式发布。OpenAI 的 gpt-oss 把 MoE 的权重以 MXFP4 发布；NVIDIA 为 DeepSeek-R1、Llama 等模型发布了 NVFP4 的版本。在没有 FP4 Tensor Core 的 Hopper 上，这些权重只能先反量化再计算，相当于 weight-only 量化，省显存但不省算力。

> [!NOTE]
> 量化后的误差最终要在任务上评估，SQNR 只是一个代理。NVFP4 的实际精度还取决于量化方法：只做简单的就近舍入（本节的模拟就是这样），还是配合 2.2、2.3 节的 GPTQ、AWQ 等方法，或者训练时就使用 FP4（量化感知训练、FP4 预训练）。

## 5. 实验

- **CPU**：修改 `fp4_formats.py`，给 MXFP4 加一个变体：缩放因子取 $2^{\lceil \log_2 (a/6) \rceil}$，保证不截断。误差变好还是变差？为什么？
- **CPU**：把 NVFP4 的块缩放因子从 E4M3 换成 E8M0（只保留指数），误差损失多少？这能说明 NVFP4 的改进中有多少来自缩放因子的格式。
- **组件**：在什么条件下 MXFP4 优于 INT4 g128？什么时候差别不大？
- **Blackwell GPU**：用 vLLM 分别加载一个模型的 FP8 和 NVFP4 版本（如 `nvidia/` 下发布的 FP4 权重），比较 prefill 吞吐、decode 的 TPOT 和 GSM8K 的准确率。

## 自测

<!--QUIZ-->

## 延伸阅读

- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)，2023。
- Rouhani et al., [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537), 2023。
- NVIDIA, [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), 2025。
- vLLM 团队，[Driving vLLM WideEP and Large-Scale Serving Toward Maturity on Blackwell (Part I)](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)，2026。
