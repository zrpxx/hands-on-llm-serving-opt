# 论文与资料

| 阶段 | 论文 | 一句话 |
|---|---|---|
| 1 | *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI'22) | iteration-level scheduling，Continuous Batching 的源头 |
| 1 | *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP'23) | vLLM，把 OS 分页搬进 KV Cache |
| 1 | *SGLang: Efficient Execution of Structured Language Model Programs* (2023) | RadixAttention，前缀复用 |
| 1 | *SARATHI / Sarathi-Serve* (2023/2024) | Chunked Prefill，平衡 TTFT 与 TPOT |
| 2 | *GPTQ: Accurate Post-Training Quantization for GPTs* (ICLR'23) | 基于 Hessian 的逐层量化 |
| 2 | *AWQ: Activation-aware Weight Quantization* (MLSys'24) | 保护 1% 显著通道 |
| 2 | *SmoothQuant* (ICML'23) | 激活离群值迁移，W8A8 |
| 2 | *FP8 Formats for Deep Learning* (2022) | E4M3 / E5M2 |
| 3 | *FlashAttention* 1 / 2 / 3 (2022–2024) | IO-aware 分块注意力 |
| 3 | *Fast Inference from Transformers via Speculative Decoding* (ICML'23) | 投机采样无损性证明 |
| 3 | *Medusa* (2024) / *EAGLE* 1–3 (2024–2025) | 自带 draft head 的投机方案 |
| 3 | *GQA* (2023) / *DeepSeek-V2 MLA* (2024) | KV 头压缩 |
| 3 | *Triton: An Intermediate Language and Compiler for Tiled NN Computations* (2019) | Triton 语言 |
| 4 | *Megatron-LM* (2019) | 张量并行 |
| 4 | *DistServe* (OSDI'24) / *Splitwise* (ISCA'24) | Prefill-Decode 分离 |
| 4 | *Mooncake* (2024) | 以 KV Cache 为中心的分离式架构 |
| 4 | *S-LoRA* / *Punica* (2023) | Multi-LoRA Serving |

## 第 6 部分：2025–2026 生产实践

| 主题 | 资料 | 一句话 |
|---|---|---|
| 生产引擎 | [DeepInfer](https://github.com/deepseek-ai/DeepInfer)（DeepSeek, 2026-10） | 头部厂商的生产推理代码，V4.1-Flash 专用 |
| FP4 / Wide EP | [vLLM WideEP on GB200, Part I](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)（2026-02） | NVFP4 GEMM、FP4 dispatch、算子融合的完整案例 |
| EP 通信 | [DeepEP](https://github.com/deepseek-ai/DeepEP) / [EPLB](https://github.com/deepseek-ai/EPLB)（2025） | MoE all-to-all 与专家负载均衡 |
| AFD | [Step-3](https://arxiv.org/abs/2507.19427)（2025）/ [MegaScale-Infer](https://arxiv.org/abs/2504.02263)（2025） | Attention 与 FFN 分开部署 |
| AFD | [vLLM AFD Plugin](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin)（2026-07） | AFD 进入开源框架 |
| AFD | [How Far Can Disaggregation Go?](https://www.alphaxiv.org/abs/2605.28302)（2026） | AFD 设计空间探索 |
| 稀疏注意力 | NSA（2025）/ DeepSeek-V3.2 DSA（2025） | 长上下文下的可训练稀疏注意力 |
| 确定性 | Thinking Machines：*Defeating Nondeterminism in LLM Inference*（2025-09） | 批不变算子 |
| 算子 DSL | [TileLang](https://github.com/tile-ai/tilelang) / CuTe DSL | Triton 之后的新选择 |
| 综述 | [From Attention to Disaggregation](https://arxiv.org/abs/2511.07422)（2025） | LLM 推理系统演进脉络 |

> [!TIP]
> 读法：先读书中对应章节 → 再读论文的 Introduction + Method 图 → 最后对照 vLLM / SGLang 源码中的实现。
