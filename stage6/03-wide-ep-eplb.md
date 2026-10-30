# 6.3 Wide EP 与 EPLB

> 书外内容。原书第 7 章介绍了专家并行，4.2 节已经讨论了它的基本做法。本节讨论大规模专家并行（Wide EP）、专家负载均衡（EPLB）和 DeepEP 的两种通信模式。负载模拟是书外内容。

4.2 节说过，DeepSeek-V3 的 decode 部署用了 320 张卡，MoE 部分 EP = 320，每张卡只放一个路由专家。这种把专家铺到几十上百张卡上的做法，被称为 **Wide EP**。2025 年以来，vLLM、SGLang 都把它做成了标准的部署方式，在 Blackwell 的 NVL72 机柜上更是如此。

## 1. 为什么要把 EP 铺宽

decode 时，MoE 层的算术强度取决于**每个专家分到多少个 token**。每个 token 选 $k$ 个专家，共 $E$ 个专家，全局 batch 为 $B$ 时，每个专家平均分到

$$
\frac{B \cdot k}{E}
$$

个 token。DeepSeek-V3 是 $k = 8, E = 256$，全局 batch 要到几千，每个专家才能分到几十上百个 token。

单机 8 卡时，全局 batch 受限于 8 张卡上的 KV Cache；铺到 64 张卡上，KV Cache 的总容量是 8 倍，全局 batch 也能大约放大 8 倍，而每张卡上的专家权重反而变少了。**卡越多，每张卡的专家越少、能留给 KV 的显存越多、每个专家分到的 token 越多**。这是 Wide EP 的核心收益。

代价是通信：每一层都要做两次 all-to-all，把 token 发给它选中的专家所在的卡（dispatch），再把结果收回来（combine）。卡越多，通信越分散，对网络的要求越高。这也是 GB200 NVL72 这类大 NVLink 域受欢迎的原因：72 张卡之间都走 NVLink，不用经过较慢的 RDMA 网络。

## 2. DeepEP：两种通信模式

DeepSeek 开源的 DeepEP 是专为 MoE 的 dispatch 和 combine 设计的通信库，提供两套 kernel：

| | 普通模式（高吞吐） | 低延迟模式 |
|---|---|---|
| 用于 | prefill、训练 | decode |
| 数据量 | 每次几千个 token | 每次几十到几百个 token |
| 网络 | 机内 NVLink + 机间 RDMA，先经 RDMA 发到目标机器，再经 NVLink 转发 | 纯 RDMA |
| 重叠计算 | 占用部分 SM 做通信 | 基于 hook 的方式，不占用 SM，通信在后台完成 |

decode 时每次通信的数据很少，瓶颈在延迟而不在带宽，所以低延迟模式直接走 RDMA、避免转发，并且让通信与计算重叠：这一个微批在通信时，另一个微批在计算（两个微批交替，与 6.4 节的流水线思路相同）。

dispatch 前还可以先把激活量化成 FP8 或 FP4 再发送（6.2 节），通信量减半甚至更少。

## 3. 负载不均衡

EP 有一个与 TP 不同的问题：每张卡的工作量**取决于路由**。如果某个专家很热门，它所在的卡就要处理更多的 token，而 MoE 层要等最忙的那张卡算完。用

$$
\text{均衡度} = \frac{\text{各卡的平均负载}}{\text{最忙的卡的负载}}
$$

衡量，均衡度就是算力的有效利用率。专家的热度天然不均，而且随请求的内容（代码、数学、中文、英文）变化。

## 4. EPLB：冗余专家

DeepSeek 开源的 EPLB（Expert Parallelism Load Balancer）用两步缓解不均衡：

1. **复制热门的专家**。每张卡多留几个槽位，把副本分给负载最高的专家。一个专家有 $r$ 个副本时，它的 token 由 $r$ 张卡分担。贪心地做：每次把一个副本给"当前每副本负载"最大的专家。
2. **把副本装进各卡**。按副本的负载从大到小，依次放进当前负载最小、还有空位的卡，使各卡的总负载接近。

EPLB 还提供分层的版本：先把专家组分配到机器上，再在机器内复制和放置，尽量让通信留在机器内部。

`code/stage6/eplb_sim.py` 实现了简化版的算法，在 256 个专家、32 张卡上模拟专家热度随时间缓慢漂移的情况：

```bash
python code/stage6/eplb_sim.py
```

```text
256 个专家，top-8，32 张卡，每卡 8 个专家 + 1 个冗余槽位；Zipf 指数 0.7

放置方式            均衡度    最忙的卡 / 平均
静态                 0.36        2.77 倍
EPLB（统计是新的）    0.83        1.20 倍
EPLB（统计已过期）    0.67        1.50 倍
```

只多存 1 个专家（显存多 11%），均衡度就从 0.36 提高到 0.83。但统计过期后，效果明显下降：**热度会漂移，放置要定期更新**。vLLM 在 DeepSeek-V4.1-Flash 上的配置是每 3000 步重新放置一次，依据前 1000 步的统计；目标模型的均衡度从 0.35 提高到 0.94。重新放置要在卡之间搬动专家权重，vLLM 用对称内存实现后，32 张卡的重新放置在一秒内完成，并且不中断 decode。

<div class="llm-widget" data-widget="eplb-balance"></div>

## 5. 在实际中使用

```bash
# vLLM：注意力数据并行 + 专家并行，开启 EPLB，每张卡 1 个冗余专家
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 16 --enable-expert-parallel \
  --enable-eplb --eplb-config '{"num_redundant_experts": 16}'
```

参数名随版本变化，以所用版本的文档为准。几个经验：

- 冗余专家占显存，在 KV Cache 紧张的 decode 上要权衡；prefill 的 batch 大、负载更平滑，冗余可以少一些。
- 带投机解码时，草稿模型如果也是 MoE（如 DSpark），它的专家要单独统计和放置。
- 负载统计本身也有开销，生产中通常只在部分步骤上采样。

## 6. 实验

- **CPU**：在 `eplb_sim.py` 中把 `--tokens` 改为 256（小 batch 的 decode），均衡度有什么变化？为什么小 batch 时随机波动的影响更大？
- **CPU**：实现"每 N 步用最近 M 步的统计重新放置"，扫描 N，观察均衡度与重新放置的频率之间的关系。
- **组件**：冗余槽位从 0 加到 8，均衡度的提升在哪里开始变缓？
- **多卡 GPU**：用 vLLM 部署一个较小的 MoE 模型（如 Qwen3-30B-A3B），开启专家并行，分别开关 EPLB，比较高并发下的 TPOT。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "把 EP 从 8 张卡铺到 64 张卡，最主要的收益是什么？", "options": ["每张卡上的专家变少，留给 KV 的显存变多，全局 batch 更大，每个专家分到更多 token", "通信量减少", "不再需要负载均衡", "每个 token 选的专家变少"], "answer": 0, "explain": "每个专家平均分到 B × k / E 个 token，全局 batch B 越大，MoE 的算术强度越高。代价是通信更分散。"}, {"q": "DeepEP 的低延迟模式为什么适合 decode？", "options": ["它的带宽更高", "decode 每次通信的数据很少，瓶颈在延迟；低延迟模式直接走 RDMA、不经转发，并且不占用 SM 就能与计算重叠", "它支持更多的专家", "它会压缩 KV Cache"], "answer": 1, "explain": "prefill 的数据量大，用普通模式：先经 RDMA 发到目标机器，再经 NVLink 转发，追求带宽。"}, {"q": "EPLB 用统计过的专家热度做了放置，过了很久没有更新。会发生什么？", "options": ["没有影响，热度是固定的", "热度随请求内容漂移，均衡度逐渐下降，需要定期重新统计、重新放置", "模型的输出会出错", "冗余专家会被自动删除"], "answer": 1, "explain": "本节的模拟中，统计过期后均衡度从 0.83 降到 0.67。vLLM 每 3000 步依据前 1000 步重新放置。"}]'></div>

## 延伸阅读

- [DeepEP](https://github.com/deepseek-ai/DeepEP) 与 [EPLB](https://github.com/deepseek-ai/EPLB)。
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024。第 3.4 节的推理部署。
- vLLM 团队，[vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP](https://vllm.ai/blog/2025-12-17-large-scale-serving)，2025。
- vLLM 团队，[Serving DeepSeek-V4.1-Flash at Scale](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)，2026。
