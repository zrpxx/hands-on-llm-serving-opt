# 6.1 DeepInfer 导读

> 书外内容 ｜ 仓库：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) ｜ 开源时间：2026-10-20 ｜ 协议：Apache-2.0

> [!NOTE]
> 状态：未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 它是什么

README 的定位是 DeepSeek 大规模线上服务所用的推理框架。这次开源的是**离线推理**版本，只支持 **DeepSeek-V4.1-Flash** 一个模型（默认权重 `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。致谢里写明它最初主要基于 vLLM 构建，也参考了 SGLang 和 TensorRT-LLM，所以读它的时候可以一直拿 vLLM 做参照。

| 项 | 内容 |
|---|---|
| 硬件 | NVIDIA SM100（CUDA 13）；华为昇腾 950（CANN 9.2） |
| 环境 | Linux、GCC 13+、Python 3.12 |
| 算子 | FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect，以及自写的 TileLang 算子 |
| 并行 | Attention 数据并行、balanced context parallelism、MoE 专家并行 + EPLB |
| 基准 | Prefill 在 4 张 SM100 上测，Decode 在 64 张上测，各自都在一个 scale-up 域内；复现脚本 `examples/sweep/run_sweep.sh` |

> [!WARNING]
> 手上如果只有 A100/A800/H100/4090 这类卡，跑不起来。这一节的重点是读懂设计，第 6 部分 的其他小节再把思路移植到能跑的环境里。

## 值得重点看的五个机制

| 机制 | README 里的描述 | 对应书中/本库章节 |
|---|---|---|
| **DSpark 投机解码** | 用 draft token 加速 decode，由置信度估计来安排什么时候做验证。`DEEPINFER_ENABLE_MTP_VARLEN=1` 开启，`DEEPINFER_USE_NEXTN_DECODE` 设最大草稿长度（默认 5） | 3.4 / 3.5 → [6.5](stage6/05-mtp-speculative.md) |
| **SWA bounded replay** | prefill 时只重放输入的一段有界后缀；保留的长度要覆盖滑动窗口加上最大回溯长度（默认配置保留 140 个 token） | 3.1 → [6.6](stage6/06-long-context.md) |
| **异步调度** | CPU 上的批调度与设备执行重叠，默认开启 | 1.5 / 1.6 → [6.8](stage6/08-async-scheduling.md) |
| **批不变 + 确定性采样** | `DEEPINFER_BATCH_INVARIANT=1` 保证不同批组合下结果一致；再关掉 MTP、固定 `seq_seed` 可得到确定性采样（仅 NVIDIA） | 5.6 → [6.7](stage6/07-batch-invariance.md) |
| **EP + EPLB** | MoE 专家并行，配合专家负载均衡 | 4.2 → [6.3](stage6/03-wide-ep-eplb.md) |

另外支持 regex、JSON Schema、EBNF 三种结构化输出约束。

## 这次没有开源的部分

README 写明以下内容不在本次发布范围内：

- KV Cache 系统
- Prefill / Decode 分离（README 说引擎侧需要的功能大部分已经具备）
- 数据并行副本之间的 prefill / decode 负载均衡

也就是说，书里第 7 章讲的 PD 分离、阶段四的 KV 分层存储和路由，在 DeepInfer 里是看不到实现的。这几块要回到 vLLM、SGLang、LMCache 去看。

## 建议的阅读顺序

顶层目录：`deepinfer/`、`csrc/`、`deep_cluster/`、`examples/`、`eval/`、`tests/`、`vendor/`、`doc/`。

1. 先跑通 `examples/` 里的离线推理入口（读代码即可），找到请求从进入到返回的主路径
2. 顺着主路径读调度器，重点看异步调度怎么把 CPU 工作和 GPU 执行错开
3. 读投机解码的 draft / verify 循环，看置信度是怎么决定验证时机的
4. 读 MoE 层的前向，看 DeepEP 的 dispatch / combine 和 EPLB 的冗余专家怎么接进来
5. 最后看 `csrc/` 和 TileLang 算子，挑一个和 FlashMLA / DeepGEMM 对照

每一步都记一句：**这里和 vLLM 的做法有什么不同，为什么。**

## 动手实验

| 实验 | 在哪做 |
|---|---|
| 统计 DeepInfer 与 vLLM 在调度主循环上的结构差异，画两张时序图 | 读代码 |
| 在 vLLM 上开启 async scheduling，对比小模型高并发下的 TPOT | 自己的 GPU |
| 用 `run_sweep.sh` 的参数组合，在 vLLM 上做同样的 prefill / decode 帕累托扫描 | 自己的 GPU |

## 自测题

1. 为什么只支持一个模型的框架反而更容易做到极致性能？代价是什么？
2. SWA bounded replay 为什么要求保留长度覆盖"窗口 + 最大回溯长度"？
3. 批不变推理为什么和投机解码不能同时开？

## 踩坑记录

> [!ATTENTION]
> 

## 资料

- [DeepInfer GitHub](https://github.com/deepseek-ai/DeepInfer)
- [DeepSeek-V4.1-Flash-1017 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
