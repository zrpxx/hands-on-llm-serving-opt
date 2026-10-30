# 术语表

> 按主题分组。每个术语给出一句话的解释（有常见英文写法的放在解释开头），以及详细讨论它的章节。本教程正文中保留英文的术语（如 prefill、decode、batch），这里也按英文列出。

## 推理的基本过程

| 术语 | 解释 | 章节 |
|---|---|---|
| 自回归生成 | autoregressive generation。每次生成一个 token，把它追加到输入中，再生成下一个 | [0.3](stage0/03-autoregressive-generation.md) |
| prefill | 一次性处理整个输入、生成第一个 token 的阶段，受算力限制 | [0.4](stage0/04-kv-cache-prefill-decode.md) |
| decode | 逐个生成后续 token 的阶段，batch 小时受显存带宽限制 | [0.4](stage0/04-kv-cache-prefill-decode.md) |
| KV Cache | 缓存已处理 token 的 Key 和 Value，避免每一步重算整个序列 | [0.4](stage0/04-kv-cache-prefill-decode.md) |
| 采样 | sampling。从模型输出的概率分布中选出下一个 token；温度、top-p、top-k 控制随机性 | [0.3](stage0/03-autoregressive-generation.md) |
| 停止条件 | stop condition。生成结束符、达到最大长度或遇到停止词时结束生成 | [0.3](stage0/03-autoregressive-generation.md) |

## 性能指标

| 术语 | 解释 | 章节 |
|---|---|---|
| TTFT | Time To First Token。从请求到达到第一个 token 返回的时间，主要由排队和 prefill 决定 | [0.8](stage0/08-metrics.md) |
| ITL | Inter-Token Latency。相邻两个输出 token 之间的间隔 | [0.8](stage0/08-metrics.md) |
| TPOT | Time Per Output Token。一个请求除第一个 token 外，平均每个 token 的耗时 | [0.8](stage0/08-metrics.md) |
| 吞吐 | throughput。单位时间处理的 token 数或请求数；要说明是输入、输出还是总 token | [0.8](stage0/08-metrics.md) |
| goodput | 同时满足延迟目标的请求所贡献的吞吐，或这类请求所占的比例 | [0.8](stage0/08-metrics.md) |
| SLO | Service Level Objective。服务的延迟目标，如 TTFT P99 ≤ 1 秒 | [0.8](stage0/08-metrics.md) |
| P50 / P90 / P99 | 分位数：50%、90%、99% 的请求不超过这个值 | [0.8](stage0/08-metrics.md) |
| Little 定律 | Little's law。系统中的平均请求数 = 到达速率 × 平均停留时间 | [5.1](final/01-optimization-playbook.md) |

## 硬件与性能模型

| 术语 | 解释 | 章节 |
|---|---|---|
| HBM | GPU 的高带宽显存；decode 的速度主要取决于它的带宽 | [0.5](stage0/05-gpu-specs.md) |
| Tensor Core | GPU 上做矩阵乘法的专用单元，支持的精度决定了能用哪些量化格式 | [0.5](stage0/05-gpu-specs.md) |
| SM | Streaming Multiprocessor。GPU 的计算单元，一个线程块在一个 SM 上运行 | [0.5](stage0/05-gpu-specs.md) |
| 算术强度 | arithmetic intensity。每读写一个字节做多少次浮点运算 | [0.7](stage0/07-roofline.md) |
| Roofline 模型 | 用算术强度判断算子受带宽还是算力限制，并估算它的耗时下限 | [0.7](stage0/07-roofline.md) |
| MFU / MBU | 实际达到的算力或带宽占峰值的比例 | [0.7](stage0/07-roofline.md) |
| NVLink | GPU 之间的高速互联；NVL72 把 72 张卡放在一个 NVLink 域内 | [4.1](stage4/01-dp-tp-pp.md) |
| RDMA | 跨机器直接读写内存的网络技术，用于 PD 分离的 KV 传输和跨机的 EP | [4.3](stage4/03-pd-disaggregation.md) |

## 调度与内存管理

| 术语 | 解释 | 章节 |
|---|---|---|
| 静态批处理 | static batching。整批请求一起开始、一起结束 | [1.2](stage1/02-continuous-batching.md) |
| 连续批处理 | continuous batching。每一步都可以加入新请求、移出已完成的请求，也称迭代级调度 | [1.2](stage1/02-continuous-batching.md) |
| chunked prefill | 把长 prompt 切成几块，与 decode 混在同一步中计算 | [1.2](stage1/02-continuous-batching.md) |
| PagedAttention | 把 KV Cache 分成固定大小的块，按需分配，用块表记录位置 | [1.3](stage1/03-paged-attention-vllm.md) |
| 块表 | block table。记录一个序列的逻辑块对应哪些物理块 | [1.3](stage1/03-paged-attention-vllm.md) |
| 抢占 | preemption。KV 空间不足时暂停一些请求，释放或换出它们的 KV | [1.3](stage1/03-paged-attention-vllm.md) |
| 前缀缓存 | prefix caching。缓存已算过的前缀的 KV，新请求的前缀相同时直接复用 | [1.4](stage1/04-prefix-caching.md) |
| RadixAttention | SGLang 用基数树管理前缀缓存的方法 | [1.4](stage1/04-prefix-caching.md) |
| CUDA Graph | 把一串 kernel 的发射录制下来整体重放，消除 CPU 的发射开销 | [3.6](stage3/06-profiling.md) |
| 异步调度 | async scheduling。GPU 计算第 N 步时，CPU 已在准备第 N+1 步 | [6.8](stage6/08-async-scheduling.md) |

## 量化与压缩

| 术语 | 解释 | 章节 |
|---|---|---|
| 量化 | quantization。用更少的位数表示权重、激活或 KV | [2.1](stage2/01-quant-fundamentals.md) |
| 缩放因子 | scale。把浮点数映射到低精度网格时使用的比例 | [2.1](stage2/01-quant-fundamentals.md) |
| 分组量化 | group-wise quantization。每 g 个元素共享一个缩放因子，g 越小误差越小、开销越大 | [2.1](stage2/01-quant-fundamentals.md) |
| W4A16 / W8A8 | 权重 4 位、激活 16 位 / 权重和激活都是 8 位 | [2.1](stage2/01-quant-fundamentals.md) |
| weight-only 量化 | 只量化权重，计算前反量化；省显存和带宽，不省算力 | [2.1](stage2/01-quant-fundamentals.md) |
| GPTQ | 利用二阶信息逐列量化，并用剩余的列补偿误差 | [2.2](stage2/02-gptq.md) |
| AWQ | 根据激活的大小保护重要的通道 | [2.3](stage2/03-awq.md) |
| SmoothQuant | 把激活的离群值按通道迁移到权重上，使 W8A8 可行 | [2.4](stage2/04-activation-quant.md) |
| FP8 | E4M3 / E5M2。8 位浮点；E4M3 精度高、E5M2 范围大 | [2.4](stage2/04-activation-quant.md) |
| 离群值 | outlier。少数幅度远大于其他值的元素或通道，是激活量化的主要难点 | [2.4](stage2/04-activation-quant.md) |
| KV Cache 量化 | 用 FP8 或更低的精度存储 KV | [2.5](stage2/05-kv-cache-quant.md) |
| Marlin | 高效的 INT4 × FP16 矩阵乘法 kernel | [2.6](stage2/06-quant-kernels.md) |
| MXFP4 / NVFP4 | 两种 4 位浮点格式，分别每 32 个、16 个元素共享一个缩放因子 | [6.2](stage6/02-fp4-formats.md) |
| 蒸馏 | distillation。用大模型的输出训练小模型 | [2.8](stage2/08-distillation-pruning.md) |
| 剪枝 | pruning。去掉不重要的权重；2:4 结构化稀疏可以被 Tensor Core 加速 | [2.8](stage2/08-distillation-pruning.md) |

## 注意力与算子

| 术语 | 解释 | 章节 |
|---|---|---|
| MHA / MQA / GQA | 每个 Q 头有自己的 KV / 所有头共用一组 KV / 每组 Q 头共用一组 KV | [3.1](stage3/01-attention-variants.md) |
| MLA | Multi-head Latent Attention。DeepSeek 把 KV 压缩成低维潜在向量的注意力 | [3.1](stage3/01-attention-variants.md) |
| 滑动窗口注意力 | SWA。每个 token 只看最近的 w 个 token | [3.1](stage3/01-attention-variants.md) |
| FlashAttention | 分块计算注意力、不把注意力矩阵写回显存 | [3.2](stage3/02-flashattention.md) |
| 在线 softmax | online softmax。分块计算 softmax 时，遇到更大的值就修正之前的结果 | [3.2](stage3/02-flashattention.md) |
| 算子融合 | kernel fusion。把多个算子合并为一个 kernel，减少显存读写和发射 | [3.3](stage3/03-triton-kernels.md) |
| Triton | 以线程块为单位编写 GPU kernel 的 Python 语言 | [3.3](stage3/03-triton-kernels.md) |
| split-K / split-KV | 把归约维度切成几段并行计算再合并；段数随形状变化会破坏批不变性 | [6.7](stage6/07-batch-invariance.md) |
| 稀疏注意力 | NSA、DSA。每个 query 只对挑选出的部分位置做注意力 | [6.6](stage6/06-long-context.md) |
| TileLang / CuTe DSL | Triton 之后的两种 kernel 编程语言 | [6.9](stage6/09-kernel-dsl.md) |

## 投机解码

| 术语 | 解释 | 章节 |
|---|---|---|
| 投机解码 | speculative decoding。先由便宜的方式提出几个草稿 token，再由目标模型一次验证 | [3.4](stage3/04-speculative-decoding.md) |
| 接受率 | acceptance rate。草稿 token 被目标模型接受的比例 | [3.4](stage3/04-speculative-decoding.md) |
| 拒绝采样 | rejection sampling。保证投机解码的输出分布与目标模型完全相同的验证方法 | [3.4](stage3/04-speculative-decoding.md) |
| n-gram 投机 | 从已有的上下文中查找与结尾相同的片段作为草稿 | [3.5](stage3/05-spec-decoding-advanced.md) |
| Medusa / EAGLE | 在目标模型上附加草稿头 | [3.5](stage3/05-spec-decoding-advanced.md) |
| MTP | Multi-Token Prediction。预训练时就带上的、预测更远位置的模块，推理时用作草稿 | [3.5](stage3/05-spec-decoding-advanced.md)、[6.5](stage6/05-mtp-speculative.md) |
| 树形验证 | tree verification。一次验证多条候选路径 | [3.5](stage3/05-spec-decoding-advanced.md) |

## 分布式与集群

| 术语 | 解释 | 章节 |
|---|---|---|
| 数据并行 | DP。多个完整的副本各自处理一部分请求 | [4.1](stage4/01-dp-tp-pp.md) |
| 张量并行 | TP。把每一层的矩阵切到多张卡上，每层需要 all-reduce | [4.1](stage4/01-dp-tp-pp.md) |
| 流水线并行 | PP。把不同的层放到不同的卡上 | [4.1](stage4/01-dp-tp-pp.md) |
| MoE | Mixture of Experts。每个 token 只激活少数几个专家 | [4.2](stage4/02-expert-parallelism.md) |
| 专家并行 | EP。把不同的专家放到不同的卡上，用 all-to-all 收发 token | [4.2](stage4/02-expert-parallelism.md) |
| Wide EP | 把专家铺到几十上百张卡上的大规模专家并行 | [6.3](stage6/03-wide-ep-eplb.md) |
| EPLB | 用冗余专家和重新放置平衡各卡负载的算法 | [6.3](stage6/03-wide-ep-eplb.md) |
| dispatch / combine | MoE 中把 token 发给专家、把结果收回来的两次 all-to-all | [6.3](stage6/03-wide-ep-eplb.md) |
| PD 分离 | Prefill-Decode disaggregation。prefill 和 decode 在不同的实例上进行，中间传输 KV | [4.3](stage4/03-pd-disaggregation.md) |
| AFD | Attention-FFN disaggregation。一层中的 Attention 和 FFN 放在不同的卡上 | [6.4](stage6/04-attention-ffn-disaggregation.md) |
| 上下文并行 | CP。把一条长序列沿长度切到多张卡上 | [6.6](stage6/06-long-context.md) |
| KV 分层存储 | KV offloading。把 KV 存到 CPU 内存、SSD 或远端存储，需要时再加载 | [4.4](stage4/04-advanced-kv-cache.md) |
| 冷启动 | cold start。新实例从启动到能服务请求的时间：拉镜像、加载权重、编译、预热 | [4.5](stage4/05-multi-model-serving.md) |
| Multi-LoRA | 一个基础模型同时服务多个 LoRA 适配器 | [4.7](stage4/07-multi-lora.md) |
| 缓存感知路由 | cache-aware routing。把请求发给前缀缓存命中最多、又不过载的实例 | [4.6](stage4/06-load-balancing-routing.md) |
| 一致性哈希 | consistent hashing。增减实例时只有约 1/N 的键改变位置的哈希方法 | [4.6](stage4/06-load-balancing-routing.md) |

## 应用场景

| 术语 | 解释 | 章节 |
|---|---|---|
| 语义缓存 | semantic caching。按语义相似度复用以前的回答 | [5.3](final/03-semantic-caching.md) |
| 语义路由 | semantic routing。根据请求的内容选择模型、工具或是否开启推理 | [5.3](final/03-semantic-caching.md) |
| 视觉 token | 图片经视觉编码器转换后送入 LLM 的 token | [5.4](final/04-multimodal-serving.md) |
| EPD 分离 | 多模态编码器、prefill、decode 三者分开部署 | [5.4](final/04-multimodal-serving.md) |
| GGUF | llama.cpp 使用的模型格式 | [5.5](final/05-edge-ai.md) |
| rollout | RL 训练中由当前策略生成样本的阶段 | [5.6](final/06-rl-serving.md) |
| 同策略 / 离策略 | on-policy / off-policy。训练样本是否由当前的参数生成 | [5.6](final/06-rl-serving.md) |
| 约束解码 | constrained decoding。采样时屏蔽不合法的 token，保证输出符合 JSON Schema 或语法 | [5.7](final/07-agentic-serving.md) |
| 批不变性 | batch invariance。一个请求的计算结果不随它所在的 batch 而变化 | [6.7](stage6/07-batch-invariance.md) |
