<!-- 各节对应的书中章节写在页面开头。标注"书外"的部分需要额外阅读论文与源码。 -->

* [首页](/)
* [路线与使用说明](guide/roadmap.md)
* [实验环境](guide/lab-setup.md)

---

* **第 0 部分　基础与估算**
  * [0.1 模型服务概览](stage0/01-serving-landscape.md)
  * [0.2 GPU 规格解读](stage0/02-gpu-specs.md)
  * [0.3 显存估算：权重与 KV Cache](stage0/03-memory-budget.md)
  * [0.4 算术强度与 Roofline 分析](stage0/04-roofline.md)
  * [0.5 性能指标：TTFT、TPOT、吞吐与 Goodput](stage0/05-metrics.md)

* **第 1 部分　推理引擎与高效 Serving**
  * [1.1 Decoder-only Transformer 与注意力](stage1/01-transformer-anatomy.md)
  * [1.2 自回归生成的执行过程](stage1/02-autoregressive-walkthrough.md)
  * [1.3 KV Cache 与 Prefill / Decode](stage1/03-kv-cache-prefill-decode.md)
  * [1.4 从零实现在线服务：批处理与流式输出](stage1/04-serving-from-scratch.md)
  * [1.5 Continuous Batching 与 Chunked Prefill](stage1/05-continuous-batching.md)
  * [1.6 vLLM 架构与调度器](stage1/06-vllm-internals.md)
  * [1.7 Prefix Caching 与 RadixAttention](stage1/07-prefix-caching.md)
  * [1.8 推理框架对比](stage1/08-framework-comparison.md)
  * [项目 1　MiniServe](stage1/boss-miniserve.md)

* **第 2 部分　压缩与量化**
  * [2.1 量化基础](stage2/01-quant-fundamentals.md)
  * [2.2 GPTQ](stage2/02-gptq.md)
  * [2.3 AWQ](stage2/03-awq.md)
  * [2.4 SmoothQuant（书外）](stage2/04-smoothquant.md)
  * [2.5 FP8、INT4 与 KV Cache 量化](stage2/05-fp8-int4-kv-quant.md)
  * [2.6 蒸馏与剪枝](stage2/06-distillation-pruning.md)
  * [2.7 量化 Kernel：W4A16 与 W8A8（书外）](stage2/07-quant-kernels.md)
  * [项目 2　Quant Arena](stage2/boss-quant-arena.md)

* **第 3 部分　加速算法与算子**
  * [3.1 注意力变体：MQA、GQA、MLA、滑动窗口](stage3/01-attention-variants.md)
  * [3.2 FlashAttention](stage3/02-flashattention.md)
  * [3.3 Kernel Fusion 与 Triton 语言（书外）](stage3/03-triton-kernels.md)
  * [3.4 投机解码原理](stage3/04-speculative-decoding.md)
  * [3.5 投机解码方法与调参](stage3/05-spec-decoding-advanced.md)
  * [3.6 性能剖析](stage3/06-profiling.md)
  * [项目 3　Kernel Forge](stage3/boss-kernel-forge.md)

* **第 4 部分　分布式与大规模 Serving**
  * [4.1 数据并行、张量并行与流水线并行](stage4/01-dp-tp-pp.md)
  * [4.2 MoE 与专家并行](stage4/02-expert-parallelism.md)
  * [4.3 Prefill-Decode 分离](stage4/03-pd-disaggregation.md)
  * [4.4 KV Cache 进阶：长上下文与分层存储](stage4/04-advanced-kv-cache.md)
  * [4.5 多模型服务](stage4/05-multi-model-serving.md)
  * [4.6 负载均衡与缓存感知路由](stage4/06-load-balancing-routing.md)
  * [4.7 Multi-LoRA Serving](stage4/07-multi-lora.md)
  * [4.8 企业级部署架构](stage4/08-enterprise-architecture.md)
  * [项目 4　Cluster Commander](stage4/boss-cluster-commander.md)

* **第 5 部分　实践与前沿**
  * [5.1 端到端优化流程](final/01-optimization-playbook.md)
  * [5.2 优化权衡](final/02-tradeoffs.md)
  * [5.3 语义缓存](final/03-semantic-caching.md)
  * [5.4 多模态 Serving](final/04-multimodal-serving.md)
  * [5.5 端侧推理](final/05-edge-ai.md)
  * [5.6 强化学习中的推理服务](final/06-rl-serving.md)
  * [5.7 Agent 场景下的推理服务](final/07-agentic-serving.md)

* **第 6 部分　2026 年生产实践（书外）**
  * [6.1 DeepInfer 导读](stage6/01-deepinfer-overview.md)
  * [6.2 NVFP4 与 MXFP4](stage6/02-fp4-formats.md)
  * [6.3 Wide EP 与 EPLB](stage6/03-wide-ep-eplb.md)
  * [6.4 Attention-FFN 分离](stage6/04-attention-ffn-disaggregation.md)
  * [6.5 MTP 与投机验证调度](stage6/05-mtp-speculative.md)
  * [6.6 长上下文：稀疏注意力与 Context Parallelism](stage6/06-long-context.md)
  * [6.7 批不变推理与确定性采样](stage6/07-batch-invariance.md)
  * [6.8 异步调度](stage6/08-async-scheduling.md)
  * [6.9 算子 DSL：TileLang 与 CuTe DSL](stage6/09-kernel-dsl.md)
  * [毕业项目](final/boss-graduation.md)

---

* **附录**
  * [术语表](appendix/glossary.md)
  * [论文与资料](appendix/papers.md)
  * [常用估算公式](appendix/cheatsheet.md)
  * [学习日志](appendix/journal.md)
