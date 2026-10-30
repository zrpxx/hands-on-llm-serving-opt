# 6.4 Attention-FFN 分离

> 书外内容。原书第 7 章介绍了 Prefill-Decode 分离（4.3 节）。本节讨论另一种分离：把一层中的 Attention 和 FFN（MoE）放到不同的卡上。算术强度的估算和流水线组件是书外内容。

PD 分离按**阶段**拆分：prefill 和 decode 的计算特征不同，就放到不同的实例上。Attention-FFN 分离（AFD）按**算子**拆分：即使只看 decode，一层中的 Attention 和 FFN 的计算特征也截然不同。StepFun 的 Step-3 和字节跳动的 MegaScale-Infer 在 2025 年提出了这种做法，vLLM 在 2026 年 7 月发布了 AFD 插件。

## 1. 两种算子，两种瓶颈

0.7 节的 Roofline 告诉我们，算子是受带宽还是算力限制，取决于它的算术强度（每读一个字节做多少次计算）。decode 时：

- **Attention**：每个请求读的是**自己的** KV Cache，batch 再大也不能复用。算术强度与 batch 无关，只取决于模型结构：GQA 约等于每组 KV 对应的 Q 头数，MLA 因为 128 个头共享同一份压缩的 KV，可以达到两百多。
- **FFN / MoE**：所有请求共用同一份权重。每个专家分到的 token 越多，权重读一次被用得越充分，算术强度随 batch 线性增长。

`code/stage6/afd_balance.py` 按 H800 的规格估算了两个模型的一层：

```bash
python code/stage6/afd_balance.py
```

```text
上下文 4096 token，EP = 32；H800 的平衡点约 591 FLOP/字节

Qwen3-235B-A22B
  每卡 batch    Attention 强度    MoE 强度   Attention       MoE
        16              16        64        40 us      23 us
        64              16       256       160 us      23 us
       256              16      1024       641 us      39 us
      1024              16      4096      2564 us     156 us

DeepSeek-V3
  每卡 batch    Attention 强度    MoE 强度   Attention       MoE
        16             242        32        23 us     105 us
        64             242       128        90 us     105 us
       256             242       512       361 us     105 us
      1024             242      2048      1442 us     365 us
```

两个模型的情况不同。Qwen3-235B 的 Attention 始终受限于带宽，耗时随 batch 线性增长，很快就成了一层中的大头；而 32 张卡分摊之后，MoE 的耗时很小。DeepSeek-V3 的 MLA 算术强度高得多，Attention 与 MoE 的耗时在 batch 一两百时相当。

## 2. 拆开的理由

在同一张卡上，两者的需求互相牵制：

- 卡上要同时放专家的权重和 KV Cache。KV 占得越多，能放的专家越少，EP 就要铺得更宽；反过来，专家占的显存挤压了 KV，限制了 batch。
- 两种算子轮流执行，GPU 的带宽和算力总有一项没用满。
- 两部分只能一起扩缩，不能根据负载单独调整。

拆开之后，一组卡只做 Attention，存全部的 KV Cache；另一组只做 FFN，存全部的专家权重。每张 FFN 卡汇集了多张 Attention 卡的 token，每个专家分到的 token 更多。两边的数量之比可以单独调整，甚至可以用不同的硬件：Attention 需要大显存、高带宽，FFN 需要高算力。

代价是每一层都要通信两次：Attention 的输出发给 FFN 卡，FFN 的输出发回来。这与 EP 的 all-to-all 量级相当，但通信模式变成了 M 对 N。

## 3. 用流水线掩盖等待和通信

拆开以后，如果只有一个 batch，Attention 卡算完要等 FFN 卡，FFN 卡算完又要等 Attention 卡，双方各有一半时间闲着。MegaScale-Infer 的做法是 **ping-pong 流水线**：把 batch 切成几个微批，微批 1 在 FFN 卡上计算时，微批 2 在 Attention 卡上计算，再交换。只要两边每层的耗时接近、通信时间短于计算时间，两边就都能保持忙碌。

<div class="llm-widget" data-widget="afd-pingpong"></div>

这里有两个约束：微批数越多，每个微批越小，每次计算的效率越低；微批太小时，单次计算已经藏不住通信。所以通常用两到三个微批。

## 4. 实际效果

公开的结果显示，AFD 的收益与模型结构和硬件密切相关：

- **Step-3** 从模型设计阶段就考虑了 AFD，它的注意力（MFA）刻意降低了 KV 的大小和算术强度，让 Attention 适合放在带宽高、算力一般的卡上。论文报告，在 Hopper 上 50 ms TPOT 的约束下，decode 吞吐达到每卡每秒 4039 个 token，高于 DeepSeek-V3 的公开数字。
- **vLLM AFD 插件**在昇腾 910C 上测试 DeepSeek-V3.2：48 张 Attention 卡加 16 张 FFN 卡，比 EP64 慢约 5%；64 张 Attention 卡加 16 张 FFN 卡，比 EP64 快约 11%（16K 输入）。作者说明这是用较小规模模拟较大规模的实验，结果不代表精度和服务延迟，插件仍处于实验阶段。

一个合理的总结：对于 MLA 这类算术强度已经不低的注意力，在同构的卡上，AFD 相对于成熟的 Wide EP 收益有限；它更大的价值在于**异构硬件**和**独立扩缩**。这与 PD 分离很像：分离本身不保证更快，它提供的是按需配置的自由度。

## 5. 与其他分离的关系

| 分离方式 | 拆开的是 | 解决的问题 | 本教程 |
|---|---|---|---|
| PD 分离 | prefill 与 decode 两个阶段 | prefill 打断 decode，两者的最优并行方式不同 | 4.3 节 |
| EPD 分离 | 多模态编码器、prefill、decode | 编码器的开销 | 5.4 节 |
| AFD | 一层中的 Attention 与 FFN | 两种算子的瓶颈不同，显存互相挤占 | 本节 |

这几种可以组合：先做 PD 分离，再在 decode 实例内部做 AFD。

## 6. 实验

- **CPU**：在 `afd_balance.py` 中把上下文改为 32768，Attention 和 MoE 的耗时之比怎样变化？这说明长上下文的负载更适合怎样的 Attention 卡与 FFN 卡之比？
- **CPU**：给 `afd_balance.py` 加一种 GQA 模型，每组 KV 对应的 Q 头数为 4。它的 Attention 算术强度是多少？
- **组件**：Attention 每层 3、FFN 每层 2、通信 1 时，微批数从 1 加到 4，总耗时怎样变化？什么时候再加微批就没有用了？

## 自测

<!--QUIZ-->

## 延伸阅读

- StepFun, [Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding](https://arxiv.org/abs/2507.19427), 2025。
- Zhu et al., [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263), 2025。
- vLLM 团队，[Announcing vLLM AFD Plugin: Disaggregating Attention and FFN for Flexible MoE Serving](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin)，2026。
