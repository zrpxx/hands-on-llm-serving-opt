<!-- 图例：📖 ChX = 书中对应章节 · 📎 = 书外拓展（书里没讲或只点到为止，需要读论文/源码） · 🏁 = PBL 项目 -->

* [🏠 首页](/)
* [🗺️ 路线图 & 使用说明](guide/roadmap.md)
* [🧰 实验环境搭建](guide/lab-setup.md)

---

* **⓪ 基础：先学会估算**
  * [0.1 模型服务全景：从训练到 Serving 📖Ch1](stage0/01-serving-landscape.md)
  * [0.2 读懂 GPU 规格：FLOPS / HBM / 带宽 📖Ch5](stage0/02-gpu-specs.md)
  * [0.3 显存账本：权重与 KV Cache 估算 📖Ch5](stage0/03-memory-budget.md)
  * [0.4 Roofline 与算术强度：Prefill 算力墙 vs Decode 带宽墙 📖Ch5](stage0/04-roofline.md)
  * [0.5 指标体系：TTFT / TPOT / ITL / 吞吐 / Goodput 📖Ch4](stage0/05-metrics.md)

* **① 推理引擎与高效 Serving**
  * [1.1 Transformer 解剖：Decoder-only 与 Attention 📖Ch2](stage1/01-transformer-anatomy.md)
  * [1.2 逐行跑通 Qwen：自回归生成全过程 📖Ch2](stage1/02-autoregressive-walkthrough.md)
  * [1.3 KV Cache 与 Prefill / Decode 两阶段 📖Ch2](stage1/03-kv-cache-prefill-decode.md)
  * [1.4 从零手搓 Serving：单请求 → Batching → Streaming 📖Ch3](stage1/04-serving-from-scratch.md)
  * [1.5 Dynamic → Continuous Batching → Chunked Prefill 📖Ch6](stage1/05-continuous-batching.md)
  * [1.6 PagedAttention 与 vLLM 架构 / Scheduler 深潜 📖Ch8](stage1/06-vllm-internals.md)
  * [1.7 Prefix Caching 与 RadixAttention 📖Ch6](stage1/07-prefix-caching.md)
  * [1.8 框架横评：vLLM / SGLang / TensorRT-LLM / llama.cpp 📖Ch8](stage1/08-framework-comparison.md)
  * [🏁 项目 1 · MiniServe：手写迷你推理引擎](stage1/boss-miniserve.md)

* **② 压缩与量化**
  * [2.1 量化基础：对称/非对称、粒度、W/A/KV 量化 📖Ch6](stage2/01-quant-fundamentals.md)
  * [2.2 GPTQ：基于二阶信息的逐层重建 📖Ch6](stage2/02-gptq.md)
  * [2.3 AWQ：激活感知的显著权重保护 📖Ch6 Ch9](stage2/03-awq.md)
  * [2.4 SmoothQuant：把激活离群值"迁移"给权重 📎](stage2/04-smoothquant.md)
  * [2.5 FP8 / INT4 / KV Cache 量化：硬件格式与选型 📖Ch6](stage2/05-fp8-int4-kv-quant.md)
  * [2.6 量化之外：蒸馏与剪枝 📖Ch6](stage2/06-distillation-pruning.md)
  * [2.7 量化 Kernel 是怎么跑快的：Marlin / W4A16 / W8A8 📎](stage2/07-quant-kernels.md)
  * [🏁 项目 2 · Quant Arena：量化方案对比](stage2/boss-quant-arena.md)

* **③ 加速算法与算子**
  * [3.1 Attention 变体：MQA / GQA / MLA / 滑动窗口 📖Ch6](stage3/01-attention-variants.md)
  * [3.2 FlashAttention：IO-aware 的分块与在线 Softmax 📖Ch6](stage3/02-flashattention.md)
  * [3.3 Kernel Fusion 与 Triton 语言入门 📎](stage3/03-triton-kernels.md)
  * [3.4 投机采样原理：Draft-Verify 与无损性证明 📖Ch7](stage3/04-speculative-decoding.md)
  * [3.5 投机采样进阶：EAGLE / Medusa / N-gram 与调参 📖Ch7](stage3/05-spec-decoding-advanced.md)
  * [3.6 性能剖析：torch.profiler / Nsight Systems 📖Ch10](stage3/06-profiling.md)
  * [🏁 项目 3 · Kernel Forge：手写算子与投机采样](stage3/boss-kernel-forge.md)

* **④ 分布式与大规模 Serving**
  * [4.1 并行三板斧：DP / TP / PP 📖Ch7](stage4/01-dp-tp-pp.md)
  * [4.2 MoE 与 Expert Parallelism 📖Ch7](stage4/02-expert-parallelism.md)
  * [4.3 Prefill-Decode 分离与 KV 传输 📖Ch7](stage4/03-pd-disaggregation.md)
  * [4.4 进阶 KV Cache：长上下文、分层存储、LMCache 📖Ch7](stage4/04-advanced-kv-cache.md)
  * [4.5 多模型服务与 Triton Inference Server 📖Ch3](stage4/05-multi-model-serving.md)
  * [4.6 集群负载均衡与 KV/Prefix-aware 路由 📖Ch7 📎](stage4/06-load-balancing-routing.md)
  * [4.7 Multi-LoRA Serving 📖Ch10](stage4/07-multi-lora.md)
  * [4.8 企业级架构：开源栈 vs 云厂商、Build or Buy 📖Ch4](stage4/08-enterprise-architecture.md)
  * [🏁 项目 4 · Cluster Commander：SLO 下的集群调度](stage4/boss-cluster-commander.md)

* **⑤ 实战与前沿（书的收尾）**
  * [5.1 端到端优化 8 步法：Qwen3-14B × vLLM 📖Ch9](final/01-optimization-playbook.md)
  * [5.2 常见优化权衡速查 📖Ch9](final/02-tradeoffs.md)
  * [5.3 Semantic Caching 📖Ch10](final/03-semantic-caching.md)
  * [5.4 多模态 Serving 📖Ch10](final/04-multimodal-serving.md)
  * [5.5 Edge AI：端侧推理 📖Ch10](final/05-edge-ai.md)
  * [5.6 RL 中的 Serving：Rollout 与确定性 📖Ch10](final/06-rl-serving.md)
  * [5.7 Agent 时代的 Serving 📖Ch4](final/07-agentic-serving.md)

* **⑥ 2026 生产实践 📎**
  * [6.1 DeepInfer 导读：DeepSeek 的生产推理引擎](stage6/01-deepinfer-overview.md)
  * [6.2 FP4 时代：NVFP4 / MXFP4 与低精度通信](stage6/02-fp4-formats.md)
  * [6.3 Wide EP 与 EPLB：大规模 MoE 专家并行](stage6/03-wide-ep-eplb.md)
  * [6.4 Attention-FFN 分离（AFD）](stage6/04-attention-ffn-disaggregation.md)
  * [6.5 投机解码新形态：MTP 与置信度调度验证](stage6/05-mtp-speculative.md)
  * [6.6 长上下文：滑动窗口、稀疏注意力与 Context Parallelism](stage6/06-long-context.md)
  * [6.7 批不变推理与确定性采样](stage6/07-batch-invariance.md)
  * [6.8 异步调度：把 CPU 开销藏起来](stage6/08-async-scheduling.md)
  * [6.9 新一代算子 DSL：TileLang / CuTe DSL](stage6/09-kernel-dsl.md)
  * [🎓 毕业项目](final/boss-graduation.md)

---

* **📚 附录**
  * [术语表（中英对照）](appendix/glossary.md)
  * [论文与资料](appendix/papers.md)
  * [速算公式卡片](appendix/cheatsheet.md)
  * [学习日志](appendix/journal.md)
