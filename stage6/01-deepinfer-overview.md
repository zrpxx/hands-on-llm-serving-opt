# 6.1 DeepInfer 导读

> 书外内容。仓库：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)，2026 年 10 月开源，Apache-2.0 协议。本节的事实以仓库 README 为准，模型结构参考了 vLLM 博客和第三方解读，写作时（2026 年 10 月底）DeepSeek 尚未发布完整的技术报告。

原书写作于 2025 年底。此后一年，大规模推理的做法又有了不少变化：FP4 进入了生产，专家并行的规模越来越大，Attention 和 FFN 开始被拆到不同的卡上，确定性推理成了 RL 训练的刚需。第 6 部分讨论这些变化。我们从一个具体的系统入手：DeepSeek 开源的推理框架 DeepInfer。它几乎用到了第 6 部分的每一个主题，可以作为一张地图。

## 1. 它是什么

README 的定位是 DeepSeek 大规模线上服务所用的推理框架。这次开源的是**离线推理**版本，只支持一个模型：**DeepSeek-V4.1-Flash**（默认权重 `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。致谢中写明它最初主要基于 vLLM 构建，也参考了 SGLang 和 TensorRT-LLM，所以读它的时候可以一直拿 vLLM 做参照。

| 项 | 内容 |
|---|---|
| 硬件 | NVIDIA SM100（Blackwell，CUDA 13）；华为昇腾 950（CANN 9.2） |
| 环境 | Linux、GCC 13 以上、Python 3.12 |
| 算子 | FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect，以及自己用 TileLang 写的算子（`deepinfer/model_executor/layers/tilelang_ops`） |
| 并行 | Attention 用数据并行，配合均衡的上下文并行；MoE 用专家并行，配合 EPLB |
| 基准 | Prefill 在 4 张 SM100 上测，Decode 在 64 张上测，都在一个 scale-up 域（NVLink 域）内；复现脚本是 `examples/sweep/run_sweep.sh` |

手上只有 A100、H100、4090 这类卡的话，它跑不起来。这一节的重点是读懂设计，第 6 部分的其他小节再把这些思路放到能跑的环境里实验。

## 2. 只支持一个模型

一个只支持一个模型的框架，看起来像是退步。原书第 8 章比较推理框架时，通用性是一个重要的维度：vLLM、SGLang 支持几百个模型。但从另一个角度看，**通用性是有代价的**：每加一个优化，都要考虑它对其他模型、其他硬件、其他功能组合是否正确。专门为一个模型写的框架可以：

- 把模型的结构直接写进调度和内存管理，比如 V4.1-Flash 的跨层 KV 共享、FP4 的 KV Cache；
- 为每一个形状挑选、调优专门的算子；
- 去掉用不到的抽象和分支，CPU 开销更小。

代价也很明显：换一个模型几乎要重写，社区难以参与。DeepInfer 开源几天后，vLLM 团队发布博客：在 GB200 上，131K 上下文的 decode 性能与 DeepInfer 公布的结果相差不到 6%。专用框架给通用框架指明了上限，通用框架再追上来，这是近两年反复出现的模式。

## 3. 模型的几个特点

DeepInfer 的很多设计是为模型服务的，所以先了解一下 V4.1-Flash。以下来自 vLLM 的博客和第三方的解读，具体以 DeepSeek 后续的技术报告为准：

- **编码器与解码器分开**：40 层被分为前 20 层和后 20 层。前一半处理输入并生成全局的 KV；后一半复用这份全局 KV，不再为整个输入建立自己的全局状态。这与 YOCO（You Only Cache Once）的思路相近。
- **滑动窗口 + 稀疏索引**：每一层都有一个最近 128 个 token 的滑动窗口；窗口之外的上下文，通过一个稀疏索引选出相关的位置。
- **极小的 KV**：全局 KV 以 FP4 存储，加上跨层共享，每个 token 只有约 890 字节。作为对比，DeepSeek-V3 的 MLA 每个 token 约 69 KB（6.6 节）。
- **MoE**：384 个路由专家，每个 token 选 6 个。
- **DSpark 投机解码**：取代了之前的 MTP 模块，一次前向预测之后 5 个位置，并给出置信度（6.5 节）。

## 4. 值得重点看的五个机制

| 机制 | README 中的描述 | 本教程的相关章节 |
|---|---|---|
| **DSpark 投机解码** | 用草稿 token 加速 decode，由置信度估计决定什么时候做验证。`DEEPINFER_ENABLE_MTP_VARLEN=1` 开启，`DEEPINFER_USE_NEXTN_DECODE` 设置最大草稿长度（默认 5） | 3.4、3.5 → [6.5](stage6/05-mtp-speculative.md) |
| **SWA bounded replay** | prefill 时，解码器部分只重放输入的一段有界后缀；保留的长度要覆盖滑动窗口加上最大回溯长度，V4.1 的默认配置保留 140 个 token | 3.1 → [6.6](stage6/06-long-context.md) |
| **异步调度** | CPU 上的批调度与设备上的执行重叠，默认开启 | 1.2、3.6 → [6.8](stage6/08-async-scheduling.md) |
| **批不变与确定性采样** | `DEEPINFER_BATCH_INVARIANT=1` 保证不同的批组合下结果相同；再关闭投机解码、固定 `seq_seed`，可以得到确定性的采样（仅 NVIDIA） | 5.6 → [6.7](stage6/07-batch-invariance.md) |
| **EP + EPLB** | MoE 专家并行，配合专家负载均衡 | 4.2 → [6.3](stage6/03-wide-ep-eplb.md) |

另外它支持 regex、JSON Schema、EBNF 三种结构化输出的约束（5.7 节）。

**SWA bounded replay** 值得多说一句，它是模型结构和推理引擎配合的一个好例子。解码器部分不建立全局 KV，只需要滑动窗口内的状态。所以 prefill 时，解码器的 20 层根本不必处理整个输入，只要把最后一段送进去即可，长度要覆盖 128 个 token 的窗口，再加上最大回溯长度。输入有几万个 token 时，模型一半的层只算了 140 个。vLLM 实现同样的优化后，32K 输入的 prefill 吞吐提高了约 20%。

## 5. 这次没有开源的部分

README 写明以下内容不在本次发布的范围内：

- KV Cache 系统（包括前缀缓存和分层存储）；
- Prefill-Decode 分离（README 说引擎一侧需要的功能大部分已经具备）；
- 数据并行副本之间的负载均衡。

也就是说，4.3 节的 PD 分离、4.4 节的 KV 分层存储、4.6 节的路由，在 DeepInfer 里都看不到实现。这几部分要回到 vLLM、SGLang、Mooncake、LMCache 去看。

## 6. 建议的阅读顺序

顶层目录：`deepinfer/`、`csrc/`、`deep_cluster/`、`examples/`、`eval/`、`tests/`、`vendor/`、`doc/`。

1. 从 `examples/` 中的离线推理入口开始，找到一个请求从进入到返回的主路径。
2. 顺着主路径读调度器，重点看异步调度怎样把 CPU 的工作和 GPU 的执行错开。与 vLLM 的 `Scheduler` 和 `GPUModelRunner` 对照着读。
3. 读投机解码的起草与验证循环，看置信度怎样决定验证多少个草稿。
4. 读 MoE 层的前向，看 DeepEP 的 dispatch、combine 和 EPLB 的冗余专家怎样接进来。
5. 最后看 `csrc/` 和 TileLang 算子，挑一个和 FlashMLA 或 DeepGEMM 对照（6.9 节）。

每读一部分，记下一句话：**这里和 vLLM 的做法有什么不同，为什么**。

## 7. 实验

| 实验 | 需要 |
|---|---|
| 读调度器，画出 DeepInfer 与 vLLM 在一个 decode 步中 CPU 和 GPU 的时序图，标出两者的不同 | 读代码 |
| 在 vLLM 上用小模型做高并发测试，比较开关异步调度（`--no-async-scheduling`）时的 TPOT | 自己的 GPU |
| 参照 `run_sweep.sh` 的参数组合，在 vLLM 上对一个 MoE 模型做 prefill 和 decode 的帕累托扫描：横轴每个用户的速度，纵轴每卡的吞吐 | 自己的 GPU |

## 自测

<!--QUIZ-->

## 延伸阅读

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 的 README。
- [DeepSeek-V4.1-Flash-1017 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。
- vLLM 团队，[Serving DeepSeek-V4.1-Flash at Scale: vLLM Matches DeepInfer and Scales to GB300 NVL72](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)，2026。vLLM 怎样在通用框架中实现同样的优化，以及与 DeepInfer 的对比。
- Sun et al., [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254), 2024。
