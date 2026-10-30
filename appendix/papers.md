# 论文与资料

> 按部分整理，每部分只列最值得读的几篇。各节末尾的「延伸阅读」有更完整的列表。读法：先读书中或本教程对应的一节，再读论文的引言和方法部分的图，最后对照 vLLM 或 SGLang 中的实现。

## 第 1 部分　推理服务与调度

| 论文或资料 | 一句话 | 章节 |
|---|---|---|
| Yu et al., [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022 | 迭代级调度，连续批处理的源头 | 1.2 |
| Agrawal et al., [Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024 | chunked prefill，平衡 TTFT 与 ITL | 1.2 |
| Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023 | vLLM，把操作系统的分页用于 KV Cache | 1.3 |
| Zheng et al., [SGLang](https://arxiv.org/abs/2312.07104), NeurIPS 2024 | RadixAttention 与结构化生成 | 1.4、5.7 |

## 第 2 部分　量化与模型压缩

| 论文或资料 | 一句话 | 章节 |
|---|---|---|
| Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323), ICLR 2023 | 用二阶信息逐列量化并补偿误差 | 2.2 |
| Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), MLSys 2024 | 根据激活保护重要通道 | 2.3 |
| Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438), ICML 2023 | 把激活的离群值迁移到权重上 | 2.4 |
| Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), 2022 | E4M3 与 E5M2 | 2.4 |
| Frantar et al., [Marlin](https://arxiv.org/abs/2408.11743), 2024 | 批量推理下接近理想加速比的 INT4 kernel | 2.6 |
| Sun et al., [Wanda](https://arxiv.org/abs/2306.11695), ICLR 2024 | 按权重与激活的乘积剪枝 | 2.8 |

## 第 3 部分　加速算法与算子

| 论文或资料 | 一句话 | 章节 |
|---|---|---|
| Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245), EMNLP 2023 | 分组共享 KV | 3.1 |
| DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434), 2024 | MLA | 3.1 |
| Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) 1、2、3, 2022～2024 | 分块计算注意力，不写回注意力矩阵 | 3.2 |
| Tillet et al., [Triton](https://dl.acm.org/doi/10.1145/3315508.3329973), MAPL 2019 | 以线程块为单位的 kernel 语言 | 3.3 |
| Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023 | 投机解码与无损验证 | 3.4 |
| Li et al., [EAGLE-3](https://arxiv.org/abs/2503.01840), 2025 | 融合多层特征的草稿头 | 3.5、6.5 |

## 第 4 部分　分布式与大规模 Serving

| 论文或资料 | 一句话 | 章节 |
|---|---|---|
| Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053), 2019 | 张量并行 | 4.1 |
| DeepSeek-AI, [DeepSeek-V3](https://arxiv.org/abs/2412.19437), 2024 | 大规模 EP 部署、MTP、冗余专家 | 4.2、6.3、6.5 |
| Zhong et al., [DistServe](https://arxiv.org/abs/2401.09670), OSDI 2024 | PD 分离与 goodput | 4.3 |
| Qin et al., [Mooncake](https://arxiv.org/abs/2407.00079), FAST 2025 | 以 KV Cache 为中心的分离式架构 | 4.3、4.4 |
| Sheng et al., [S-LoRA](https://arxiv.org/abs/2311.03285), MLSys 2024 | 同时服务数千个 LoRA 适配器 | 4.7 |

## 第 5 部分　实践与前沿

| 论文或资料 | 一句话 | 章节 |
|---|---|---|
| [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | 语义缓存与路由的开源实现 | 5.3 |
| Bai et al., [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), 2025 | 视觉 token 的计算方式 | 5.4 |
| Hu et al., [OpenRLHF](https://arxiv.org/abs/2405.11143), 2024 | RL 训练中推理引擎的角色 | 5.6 |
| Fu et al., [AReaL](https://arxiv.org/abs/2505.24298), 2025 | 完全异步的 RL | 5.6 |
| Manus, [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus), 2025 | Agent 应用怎样保持前缀缓存命中 | 5.7 |
| Dong et al., [XGrammar](https://arxiv.org/abs/2411.15100), MLSys 2025 | 高效的约束解码 | 5.7 |

## 第 6 部分　2026 年生产实践

| 论文或资料 | 一句话 | 章节 |
|---|---|---|
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer)，DeepSeek，2026 | 专为 V4.1-Flash 写的生产推理框架 | 6.1 |
| vLLM 团队，[Serving DeepSeek-V4.1-Flash at Scale](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)，2026 | 通用框架怎样追上专用框架 | 6.1、6.3 |
| NVIDIA, [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), 2025 | NVFP4 的两层缩放 | 6.2 |
| vLLM 团队，[WideEP on GB200, Part I](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)，2026 | NVFP4、FP4 dispatch 与算子融合的完整案例 | 6.2、6.3 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) 与 [EPLB](https://github.com/deepseek-ai/EPLB)，2025 | MoE 通信与专家负载均衡 | 6.3 |
| StepFun, [Step-3](https://arxiv.org/abs/2507.19427), 2025 | 为 AFD 设计的模型与系统 | 6.4 |
| Zhu et al., [MegaScale-Infer](https://arxiv.org/abs/2504.02263), 2025 | 分离式专家并行与 ping-pong 流水线 | 6.4 |
| Yuan et al., [Native Sparse Attention](https://arxiv.org/abs/2502.11089), 2025 | 可训练的稀疏注意力 | 6.6 |
| DeepSeek-AI, [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556), 2025 | DSA 与闪电索引器 | 6.6 |
| He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), 2025 | 批不变的算子 | 6.7 |
| vLLM 团队，[Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2)，2026 | 把输入的准备搬到 GPU 上 | 6.8 |
| Wang et al., [TileLang](https://arxiv.org/abs/2504.17577), 2025 | 可组合的 tile 编程模型 | 6.9 |
