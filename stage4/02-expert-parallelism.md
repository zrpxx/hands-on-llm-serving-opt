# 4.2 MoE 与专家并行

> 对应原书第 7 章 *Expert Parallelism*。MoE 推理的通信、负载与 DeepSeek 的部署方式是书外内容。

现在最大的一批开源模型几乎都是 MoE（Mixture of Experts）结构：DeepSeek-V3、Qwen3-235B-A22B、gpt-oss、Kimi K2 等。MoE 让模型的参数量很大、每个 token 的计算量却很小，这对推理系统提出了和稠密模型完全不同的要求。这一节先看 MoE 的结构对推理意味着什么，再讨论专家并行。

## 1. MoE 的结构

MoE 模型把 Transformer 层中的 FFN 换成了一组**专家**，每个专家本身就是一个 FFN。每个 token 经过一个很小的**路由器**（一个线性层），得到它对每个专家的打分，选出分数最高的 $k$ 个专家，只用这 $k$ 个专家计算，再按分数加权求和。注意力层与稠密模型相同。

| 模型 | 总参数 | 每个 token 激活的参数 | 专家数 | 每个 token 选几个 |
|---|---|---|---|---|
| Mixtral 8x7B | 约 470 亿 | 约 130 亿 | 8 | 2 |
| gpt-oss-120b | 约 1170 亿 | 约 51 亿 | 128 | 4 |
| Qwen3-235B-A22B | 约 2350 亿 | 约 220 亿 | 128 | 8 |
| DeepSeek-V3 | 约 6710 亿 | 约 370 亿 | 256 个路由专家，另有 1 个共享专家 | 8 |

DeepSeek-V3 的每个 token 只用 370 亿参数计算，计算量与一个 37B 的稠密模型相当，但 6710 亿参数都要放在显存里。

## 2. MoE 对推理意味着什么

**显存按总参数算，计算按激活参数算。** DeepSeek-V3 的 FP8 权重约 700 GB，一台 8 卡 H200（共 1128 GB）才放得下，但每个 token 的计算量只相当于一个 37B 的模型。

**decode 需要很大的 batch。** decode 是带宽瓶颈。batch 很小时，每一步只有少数几个专家被选中，只需要读这些专家的权重，这是 MoE 的优势。但 batch 一大，几乎所有专家都会被某个 token 选中，每一步要读的权重接近全部参数，而每个专家只处理了一小部分 token。以 DeepSeek-V3 为例，每个 token 选 8 个专家，batch 为 256 时，平均每个专家只分到 8 个 token，矩阵乘法非常"瘦"，算术强度很低。要让每个专家分到足够多的 token，整个系统要同时处理几千个请求。这就是 MoE 模型倾向于大规模部署的原因。

**负载是动态的。** 哪个专家被选中，取决于输入的内容。某些专家可能一时特别热门，处理它们的卡就成了瓶颈。

## 3. 专家并行

原书的描述是：专家并行（EP）把专家分到多张卡上，每个 token 只发往它所选专家所在的卡，避免在不需要的专家上浪费计算。它通常与 TP、PP 配合使用。

一个 MoE 层在 EP 下的执行分为四步：

1. **路由**：每张卡为自己的 token 计算路由结果。
2. **dispatch**：把每个 (token, 专家) 对发往该专家所在的卡。每张卡都要向所有其他卡发送数据，这是一次 **all-to-all** 通信。
3. **专家计算**：每张卡把发给同一个专家的 token 收集起来，做一次矩阵乘法。多个专家的矩阵乘法通常合并成一个 grouped GEMM kernel。
4. **combine**：把结果送回 token 原来所在的卡，按路由权重加权求和，又是一次 all-to-all。

`code/stage4/moe_demo.py` 在 CPU 上模拟了这四步，验证按专家分组计算与逐个计算的结果相同：

```bash
python code/stage4/moe_demo.py --experts 64 --topk 4 --ep 8
```

```text
64 个专家、每个 token 选 4 个、EP = 8（每张卡 8 个专家）
分组计算与逐个计算的最大差：4.44e-15
每张卡收到的 (token, 专家) 对： [1063, 1012, 906, 1061, 1017, 1031, 1090, 1012] ；最忙的是平均的 1.06 倍
dispatch + combine 的跨卡数据量（BF16）：7.0 MiB，是这一层输入激活（1.0 MiB）的 7.0 倍
```

最后一行说明了 EP 的主要代价：每个 token 要发往 $k$ 个专家，dispatch 和 combine 加起来，跨卡的数据量是输入激活的约 $2k$ 倍。这比 TP 的 all-reduce 大得多，所以 EP 对网络的要求很高。

下面的组件可以观察真实模型在不同 EP 大小、不同专家热度下的负载：

<div class="llm-widget" data-widget="moe-ep"></div>

## 4. 注意力用数据并行，专家用专家并行

MoE 模型中，注意力和专家适合不同的并行方式：

- **注意力**的参数不多，如果用 TP 切开，每层多两次 all-reduce。MLA（3.1 节）只有一份隐向量，TP 时每张卡都要存完整的 KV Cache，切了反而浪费。所以注意力常用**数据并行**：每张卡（或每组卡）处理不同的请求，各自保存自己请求的 KV。
- **专家**的参数占了绝大部分，用 **EP** 分散到所有卡上。

于是每个 MoE 层的前后，token 在"按请求分布"和"按专家分布"之间来回转换，这正是 dispatch 和 combine 的作用。

DeepSeek-V3 的技术报告给出了它的部署方式：prefill 阶段的最小部署单元是 4 台机器、32 张 H800，MoE 部分 EP = 32；decode 阶段是 40 台机器、320 张卡，MoE 部分 EP = 320，每张卡只放一个路由专家，另外用一部分卡放冗余的热门专家。prefill 和 decode 分开部署（4.3 节），各自选择最适合的并行方式。

## 5. 负载均衡

EP 下，一个 MoE 层的耗时由最忙的那张卡决定。负载不均的原因有两个：专家本身的热度不同，以及同一批 token 碰巧集中选了某几个专家。处理的办法有：

- **训练时的均衡**。训练时加入辅助损失，或者像 DeepSeek-V3 那样给每个专家一个可调的偏置，让各专家被选中的频率接近。
- **冗余专家与重新分配**。统计各专家的负载，把热门专家复制几份放到不同的卡上，并重新安排专家的位置，让每张卡的总负载接近。DeepSeek 开源了这个算法（EPLB），6.3 节会详细讨论。
- **更快的通信库**。DeepSeek 开源的 DeepEP 为 all-to-all 提供了两套 kernel：prefill 用的高吞吐版本，以及 decode 用的低延迟版本，后者直接通过 RDMA 通信，并且能与计算重叠。

## 6. 在实际中使用

```bash
# vLLM：8 卡，注意力用数据并行，MoE 层用专家并行（EP = 8）
vllm serve Qwen/Qwen3-30B-A3B --data-parallel-size 8 --enable-expert-parallel

# SGLang：同样的思路
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --tp 8 --dp 8 --enable-dp-attention
```

vLLM 中，MoE 层的 EP 大小等于 TP 与 DP 的乘积。不加 `--enable-expert-parallel` 时，MoE 层的专家也按 TP 的方式切开。具体的参数名和可选的 all-to-all 后端在各版本之间有变化，以所用版本的文档为准。

## 7. 实验

- **CPU**：运行 `moe_demo.py`，把 `--experts` 改成 8、`--topk` 改成 2（Mixtral 的配置），对比跨卡数据量与输入激活之比的变化。
- **CPU**：在组件中选择 DeepSeek-V3、EP = 64，把 token 数从 4096 降到 256，观察最忙与平均之比。为什么 decode 阶段的负载均衡更难？
- **GPU（2 张或以上）**：用 Qwen3-30B-A3B 比较 `--tensor-parallel-size 2` 与 `--data-parallel-size 2 --enable-expert-parallel` 在高并发下的吞吐。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "DeepSeek-V3 总参数约 6710 亿，每个 token 激活约 370 亿。部署时显存和计算量分别按什么估算？", "options": ["都按 370 亿", "显存按 6710 亿（所有专家都要放下），每个 token 的计算量按 370 亿", "都按 6710 亿", "显存按 370 亿，计算量按 6710 亿"], "answer": 1, "explain": "哪个专家会被选中事先不知道，所有专家都要常驻显存；但每个 token 只经过它选中的少数专家，计算量与激活参数相当。"}, {"q": "为什么 MoE 模型的 decode 需要很大的 batch 才能高效？", "options": ["因为路由器很慢", "batch 稍大，几乎所有专家都会被选中，每一步要读接近全部的权重，而每个专家只分到很少的 token，算术强度很低", "因为注意力的计算量大", "因为 KV Cache 很小"], "answer": 1, "explain": "要让每个专家的矩阵乘法有足够多的 token，整个系统要同时处理很多请求。这也是 DeepSeek 用几百张卡部署 decode 的原因之一。"}, {"q": "专家并行中，dispatch 和 combine 指的是什么？", "options": ["加载和卸载专家的权重", "把每个 (token, 专家) 对发往专家所在的卡，以及把结果送回 token 原来的卡，两次 all-to-all 通信", "路由器的两个线性层", "prefill 和 decode"], "answer": 1, "explain": "每个 token 要发往 k 个专家，两次通信的数据量合计约为输入激活的 2k 倍，所以 EP 对网络的要求很高。"}, {"q": "MoE 模型常用「注意力数据并行、专家用专家并行」。注意力为什么不用张量并行？", "options": ["注意力不能切分", "注意力参数不多，TP 每层多两次 all-reduce；MLA 只有一份隐向量，TP 时每张卡都要存完整的 KV，切了反而浪费", "因为注意力在 CPU 上计算", "因为 EP 不支持注意力"], "answer": 1, "explain": "数据并行时每张卡处理不同的请求，各自保存自己请求的 KV。token 在 MoE 层前后通过 dispatch 和 combine 在两种分布之间转换。"}]'></div>

## 延伸阅读

- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 第 2.1.2 节（DeepSeekMoE 与无辅助损失的负载均衡）和第 3.4 节（推理部署）。
- [DeepEP](https://github.com/deepseek-ai/DeepEP) 与 [EPLB](https://github.com/deepseek-ai/EPLB)。
- Grootendorst, [A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts), 2024。原书推荐的 MoE 入门读物。
- [vLLM 文档：Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)。
