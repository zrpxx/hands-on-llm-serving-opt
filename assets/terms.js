/**
 * 术语提示：在正文中第一次出现的术语下加虚线，悬停、聚焦或点按时显示一句话解释，
 * 并链接到详细讨论它的章节，或外部的补充资料（维基百科、官方文档）。
 *
 * 规则：
 * - 每页每个术语只标注第一次出现的位置；
 * - 不处理标题、代码、链接、公式、交互组件、测验中的文字；
 * - 正在讲解这个术语的那一节不标注。
 *
 * 新增术语：在 TERMS 中加一项。match 是正文中可能出现的写法（按长度从长到短匹配），
 * see 是本教程的路由（如 'stage0/07-roofline'），url 是外部链接，二者可以都有。
 */
(function () {
  // 中文条目用「搜索并直达」的地址：标题完全匹配时直接打开条目，否则显示搜索结果，不会出现死链
  const WP = (t) => 'https://zh.wikipedia.org/w/index.php?go=Go&search=' + encodeURIComponent(t);
  const WPE = (t) => 'https://en.wikipedia.org/wiki/' + t;

  const TERMS = [
    // ---------------------------------------------------------------- 模型与生成
    { match: ['token'], def: '模型处理文本的基本单位。分词器把文字切成 token，一个汉字通常是一到两个 token，一个英文单词常常是一个。', see: 'stage0/02-transformer-anatomy' },
    { match: ['分词器', 'tokenizer'], def: '把文本切成 token、并把 token 映射为整数 ID 的程序。主流模型用字节级 BPE 算法训练分词器。', url: WPE('Byte_pair_encoding') },
    { match: ['logits'], def: '模型输出头给每个候选 token 打的原始分数，经过 softmax 才变成概率。', see: 'stage0/03-autoregressive-generation' },
    { match: ['softmax'], def: '把一组任意实数变成和为 1 的概率分布：先取指数，再除以总和。注意力和采样都要用到它。', url: WP('Softmax函数') },
    { match: ['词嵌入', '嵌入表', '嵌入向量', '嵌入模型', 'embedding'], def: '把离散的 token ID 映射成一个稠密向量。模型的第一层就是一张嵌入表，按 ID 查出对应的行。', url: WP('词嵌入') },
    { match: ['自回归'], def: '逐个生成：每次根据已有的全部 token 预测下一个，把它接到末尾，再预测下一个。', see: 'stage0/03-autoregressive-generation' },
    { match: ['Transformer'], def: '当前大模型的基础结构，由注意力层和前馈网络层交替堆叠而成。', see: 'stage0/02-transformer-anatomy', url: WP('Transformer模型') },
    { match: ['Decoder-only'], def: '只保留 Transformer 解码器部分、用因果掩码逐个生成 token 的结构。GPT、Llama、Qwen 都属于这一类。', see: 'stage0/02-transformer-anatomy' },
    { match: ['自注意力', '注意力机制'], def: '让每个位置按相关程度从其他位置读取信息的计算：用 Q 和 K 的点积算权重，再对 V 加权求和。', see: 'stage0/02-transformer-anatomy' },
    { match: ['前馈网络', 'FFN'], def: '每个位置独立经过的两三层全连接网络，占了模型的大部分参数。', see: 'stage0/02-transformer-anatomy' },
    { match: ['RMSNorm'], def: '一种归一化：每个向量除以它各元素平方均值的平方根，再乘一个可学习的缩放。比 LayerNorm 少了减均值这一步。', see: 'stage0/02-transformer-anatomy' },
    { match: ['RoPE', '旋转位置编码'], def: '旋转位置编码：把 Q、K 的分量两两成对旋转一个与位置成正比的角度，使注意力分数只与相对距离有关。', see: 'stage0/02-transformer-anatomy' },
    { match: ['SwiGLU'], def: '一种门控前馈网络：一路经过 SiLU 激活作为门，与另一路逐元素相乘，再降维输出。', see: 'stage0/02-transformer-anatomy' },
    { match: ['因果掩码'], def: '让每个位置只能看到自己和之前的位置，看不到之后的位置。生成时不能偷看答案。', see: 'stage0/02-transformer-anatomy' },
    { match: ['贪心解码'], def: '每一步都选概率最高的 token，不做随机采样。输出确定，常用于评测和对比实现。', see: 'stage0/03-autoregressive-generation' },
    { match: ['温度'], skip: ['final/05-edge-ai'], def: '采样前把 logits 除以的一个数。小于 1 让分布更尖锐、输出更确定，大于 1 让分布更平、输出更随机。', see: 'stage0/03-autoregressive-generation' },
    { match: ['top-p', 'Top-p'], def: '核采样：按概率从高到低累加，累计达到 p 就截断，只在这些候选里采样。', see: 'stage0/03-autoregressive-generation' },
    { match: ['top-k', 'Top-k'], def: '只在概率最高的 k 个候选里采样。', see: 'stage0/03-autoregressive-generation' },
    { match: ['困惑度'], def: '衡量语言模型预测能力的指标，等于平均每个 token 交叉熵的指数。越低说明模型对文本越不"意外"。', url: WP('困惑度') },
    { match: ['MoE', '混合专家'], def: '混合专家：把前馈网络换成许多个"专家"，每个 token 只激活其中几个。参数量大，每个 token 的计算量小。', see: 'stage4/02-expert-parallelism', url: WP('混合专家模型') },
    { match: ['LoRA'], def: '一种参数高效的微调方法：冻结原权重，只训练两个很小的低秩矩阵作为修正。', see: 'stage4/07-multi-lora' },

    // ---------------------------------------------------------------- 推理过程
    { match: ['KV Cache'], def: '把已经处理过的 token 在每一层算出的 K、V 缓存起来，生成新 token 时直接复用，不必重算整个序列。', see: 'stage0/04-kv-cache-prefill-decode' },
    { match: ['prefill'], def: '推理的第一阶段：一次性处理整个输入，算出所有位置的 KV 并产出第一个 token。计算量大，受算力限制。', see: 'stage0/04-kv-cache-prefill-decode' },
    { match: ['decode'], def: '推理的第二阶段：每步只处理一个新 token。每步都要读一遍全部权重，受显存带宽限制。', see: 'stage0/04-kv-cache-prefill-decode' },
    { match: ['batch'], def: '一起送进模型计算的一组请求或 token。batch 越大，读一次权重能服务的请求越多。', see: 'stage1/02-continuous-batching' },
    { match: ['连续批处理', 'continuous batching'], def: '每一步都可以加入新请求、移出已完成的请求，而不是整批同进同出。', see: 'stage1/02-continuous-batching' },
    { match: ['chunked prefill'], def: '把长输入切成几块，和其他请求的 decode 混在同一步里计算，避免长 prompt 让所有人卡住。', see: 'stage1/02-continuous-batching' },
    { match: ['PagedAttention'], def: '把 KV Cache 切成固定大小的块，按需分配，用块表记录每个序列的块在哪里，像操作系统的虚拟内存分页。', see: 'stage1/03-paged-attention-vllm' },
    { match: ['前缀缓存'], def: '把算过的前缀的 KV 留在显存里，新请求的开头相同时直接复用，省掉这部分 prefill。', see: 'stage1/04-prefix-caching' },
    { match: ['抢占'], def: 'KV 空间不够时，暂停一部分正在运行的请求，释放或换出它们的 KV，稍后再恢复。', see: 'stage1/03-paged-attention-vllm' },
    { match: ['投机解码'], def: '先用便宜的方式猜出几个 token，再让大模型一次前向验证，接受猜对的部分。输出分布与不用时完全相同。', see: 'stage3/04-speculative-decoding' },
    { match: ['量化'], def: '用更少的位数存储和计算权重、激活或 KV，例如从 16 位降到 8 位或 4 位，换取显存、带宽和速度。', see: 'stage2/01-quant-fundamentals' },
    { match: ['反量化'], def: '把低精度的整数或浮点数乘回缩放因子，恢复成近似的高精度数值。', see: 'stage2/01-quant-fundamentals' },
    { match: ['校准数据'], def: '量化时喂给模型的少量样本，用来统计激活的范围和分布，决定缩放因子。', see: 'stage2/02-gptq' },
    { match: ['蒸馏'], def: '用大模型（教师）的输出训练小模型（学生），让小模型学会大模型的行为。', see: 'stage2/08-distillation-pruning', url: WPE('Knowledge_distillation') },
    { match: ['剪枝'], def: '去掉模型中不重要的权重或结构，让模型变小、变快。', see: 'stage2/08-distillation-pruning' },

    // ---------------------------------------------------------------- 指标
    { match: ['TTFT'], def: 'Time To First Token：从请求到达到第一个 token 返回的时间，决定用户等多久才看到回应。', see: 'stage0/08-metrics' },
    { match: ['TPOT'], def: 'Time Per Output Token：第一个 token 之后，平均每个 token 的生成时间。', see: 'stage0/08-metrics' },
    { match: ['ITL'], def: 'Inter-Token Latency：相邻两个输出 token 之间的间隔，反映流式输出是否平滑。', see: 'stage0/08-metrics' },
    { match: ['goodput'], def: '同时满足延迟目标的请求所贡献的吞吐。只算"合格"的部分，比单纯的吞吐更能反映服务能力。', see: 'stage0/08-metrics' },
    { match: ['SLO'], def: 'Service Level Objective：服务承诺的指标目标，例如 99% 的请求 TTFT 不超过 1 秒。', see: 'stage0/08-metrics', url: WPE('Service-level_objective') },
    { match: ['P99', 'P90', 'P50'], def: '分位数：把所有请求的延迟从小到大排列，P99 是第 99% 位置上的值，也就是最慢的 1% 请求的门槛。', see: 'stage0/08-metrics', url: WP('分位数') },
    { match: ['Little 定律'], def: '排队论的基本定律：系统中的平均请求数 = 到达速率 × 平均停留时间。', see: 'final/01-optimization-playbook', url: WP('利特爾法則') },

    // ---------------------------------------------------------------- 硬件与性能
    { match: ['HBM'], def: 'High Bandwidth Memory：GPU 上的高带宽显存，带宽是普通内存的十几倍，decode 的速度主要取决于它。', see: 'stage0/05-gpu-specs', url: WP('高頻寬記憶體') },
    { match: ['显存带宽'], def: 'GPU 每秒能从显存读写多少字节，H100 约 3.35 TB/s。读权重和 KV 的速度由它决定。', see: 'stage0/05-gpu-specs' },
    { match: ['Tensor Core'], def: 'NVIDIA GPU 上专门做矩阵乘加的计算单元，算力远高于普通 CUDA 核心，支持的精度决定了能用哪些量化格式。', see: 'stage0/05-gpu-specs' },
    { match: ['SM'], def: 'Streaming Multiprocessor：GPU 的基本计算单元，H100 有 132 个。一个线程块在一个 SM 上运行。', see: 'stage0/05-gpu-specs' },
    { match: ['FLOPs'], def: '浮点运算次数（floating-point operations），衡量计算量。乘加一次算两次运算。', see: 'stage0/07-roofline', url: WP('每秒浮點運算次數') },
    { match: ['TFLOPS'], def: '每秒万亿次浮点运算，衡量算力。H100 的 BF16 稠密算力约 989 TFLOPS。', see: 'stage0/05-gpu-specs' },
    { match: ['算术强度'], def: '每读写一个字节做多少次浮点运算。低于硬件的平衡点就受带宽限制，高于则受算力限制。', see: 'stage0/07-roofline' },
    { match: ['Roofline'], def: '一种性能模型：用算术强度判断算子受带宽还是算力限制，并估算耗时的下限。', see: 'stage0/07-roofline', url: WPE('Roofline_model') },
    { match: ['MFU'], def: 'Model FLOPs Utilization：实际达到的算力占峰值的比例。', see: 'stage0/07-roofline' },
    { match: ['BF16', 'bfloat16'], def: '16 位浮点格式，指数位与 FP32 相同（8 位），尾数只有 7 位。范围大、精度低，是大模型训练和推理的常用精度。', see: 'stage2/01-quant-fundamentals', url: WPE('Bfloat16_floating-point_format') },
    { match: ['FP8'], def: '8 位浮点格式，常用 E4M3（4 位指数、3 位尾数）。Hopper 及之后的 GPU 有 FP8 Tensor Core。', see: 'stage2/04-activation-quant' },
    { match: ['NVLink'], def: 'NVIDIA GPU 之间的高速直连，H100 单向 450 GB/s，比 PCIe 快一个数量级。张量并行依赖它。', see: 'stage4/01-dp-tp-pp', url: WPE('NVLink') },
    { match: ['PCIe'], def: '主板上连接 CPU、GPU、网卡的通用总线。PCIe 5.0 x16 单向约 64 GB/s，远低于 NVLink。', url: WP('PCI_Express') },
    { match: ['RDMA'], def: '远程直接内存访问：网卡直接读写另一台机器的内存，不经过对方 CPU，延迟低、带宽高。', see: 'stage4/03-pd-disaggregation', url: WP('远程直接内存访问') },
    { match: ['OOM'], def: 'Out Of Memory：显存或内存不够，分配失败，程序报错退出。', url: WPE('Out_of_memory') },
    { match: ['kernel'], def: '在 GPU 上运行的一个函数。每次 kernel 启动都有几微秒的固定开销，小 kernel 太多时这部分开销不可忽略。', see: 'stage3/03-triton-kernels' },
    { match: ['CUDA Graph'], def: '把一串 GPU 操作录制下来、之后一次重放的机制，省掉每个 kernel 从 CPU 发起的开销。', see: 'stage3/06-profiling' },
    { match: ['共享内存'], def: 'GPU 每个 SM 上很小（几百 KB）但很快的片上存储，同一线程块内的线程可以共享。分块计算靠它减少显存读写。', see: 'stage3/02-flashattention' },
    { match: ['线程块'], def: 'GPU 上一起调度的一组线程，运行在同一个 SM 上，可以共享共享内存、互相同步。', see: 'stage3/03-triton-kernels' },
    { match: ['算子融合', 'kernel fusion'], def: '把几个连续的操作合并到一个 kernel 里完成，中间结果不写回显存，减少读写和启动开销。', see: 'stage3/03-triton-kernels' },

    // ---------------------------------------------------------------- 注意力与算子
    { match: ['MHA'], def: '多头注意力：每个查询头都有自己的一组 K、V。', see: 'stage3/01-attention-variants' },
    { match: ['GQA'], def: '分组查询注意力：几个查询头共用一组 K、V，KV Cache 按比例变小。', see: 'stage3/01-attention-variants' },
    { match: ['MQA'], def: '多查询注意力：所有查询头共用同一组 K、V，KV Cache 最小。', see: 'stage3/01-attention-variants' },
    { match: ['MLA'], def: 'DeepSeek 的多头潜在注意力：把 K、V 压缩成一个低维向量缓存，KV Cache 比 GQA 还小。', see: 'stage3/01-attention-variants' },
    { match: ['FlashAttention'], def: '分块计算注意力、不把 N×N 的注意力矩阵写回显存的算法，显存从平方降为线性，速度也更快。', see: 'stage3/02-flashattention' },
    { match: ['在线 softmax'], def: '分块计算 softmax 的技巧：边读边维护当前最大值和分母，遇到更大的值就修正之前的结果。', see: 'stage3/02-flashattention' },
    { match: ['Triton Inference Server'], def: 'NVIDIA 的通用模型服务框架，负责请求调度、动态批处理和多模型管理。与写 GPU kernel 的 Triton 语言无关。', url: 'https://github.com/triton-inference-server/server' },
    { match: ['Triton'], def: '这里指 OpenAI 的 Triton 语言：用 Python 以"块"为单位写 GPU kernel。注意与 NVIDIA 的 Triton Inference Server 无关。', see: 'stage3/03-triton-kernels' },

    // ---------------------------------------------------------------- 分布式
    { match: ['张量并行'], def: '把每一层的权重矩阵切开分到多张卡上，每层计算后需要 all-reduce 汇总结果。', see: 'stage4/01-dp-tp-pp' },
    { match: ['流水线并行'], def: '把模型的不同层放到不同的卡上，数据依次流过，只在交界处传一次激活。', see: 'stage4/01-dp-tp-pp' },
    { match: ['数据并行'], def: '部署多个完整的模型副本，各自处理一部分请求。', see: 'stage4/01-dp-tp-pp' },
    { match: ['专家并行'], def: '把 MoE 的不同专家放到不同的卡上，token 通过 all-to-all 通信发往它选中的专家。', see: 'stage4/02-expert-parallelism' },
    { match: ['all-reduce'], def: '一种集合通信：每张卡都拿到所有卡上数据的总和。张量并行每层要做两次。', see: 'stage4/01-dp-tp-pp', url: WPE('Collective_operation') },
    { match: ['all-to-all'], def: '一种集合通信：每张卡都向每张卡发送一份不同的数据。MoE 的专家并行用它分发 token。', see: 'stage4/02-expert-parallelism', url: WPE('Collective_operation') },
    { match: ['NCCL'], def: 'NVIDIA 的多 GPU 集合通信库，提供 all-reduce、all-gather 等操作。', url: 'https://developer.nvidia.com/nccl' },
    { match: ['PD 分离'], def: 'prefill 和 decode 放在不同的实例上运行，prefill 算完后把 KV 传给 decode 实例。', see: 'stage4/03-pd-disaggregation' },
    { match: ['一致性哈希'], def: '一种把键分配到节点的哈希方法：增减一个节点时，只有约 1/N 的键需要换位置。', see: 'stage4/06-load-balancing-routing', url: WP('一致哈希') },
    { match: ['LRU'], def: 'Least Recently Used：最近最少使用。缓存满时淘汰最久没有被访问的条目。', url: WP('缓存文件置换机制') },
    { match: ['冷启动'], def: '新实例从启动到能够服务请求的过程：拉镜像、加载权重、编译、预热。', see: 'stage4/05-multi-model-serving' },
    { match: ['Kubernetes'], def: '容器编排系统，负责部署、扩缩容和管理成批的服务实例。', url: WP('Kubernetes') },

    // ---------------------------------------------------------------- 数学与算法背景
    { match: ['Hessian'], def: '海森矩阵：函数对各变量的二阶偏导数组成的矩阵，描述误差曲面的弯曲程度。GPTQ 用它判断哪些权重的误差影响大。', see: 'stage2/02-gptq', url: WP('黑塞矩陣') },
    { match: ['Cholesky'], def: 'Cholesky 分解：把对称正定矩阵分解为一个下三角矩阵与其转置的乘积，比直接求逆更快、数值更稳定。', url: WP('科列斯基分解') },
    { match: ['KL 散度'], def: '衡量两个概率分布差异的量，两个分布相同时为 0。', url: WP('相对熵') },
    { match: ['重要性采样'], def: '用一个分布的样本估计另一个分布下的期望：给每个样本乘上两个分布的概率之比作为权重。', url: WP('重要性采样') },
    { match: ['拒绝采样'], def: '先从一个容易采样的分布中抽样，再按一定概率接受或拒绝，使最终结果服从目标分布。投机解码的验证就是这样做的。', see: 'stage3/04-speculative-decoding', url: WP('拒絕採樣') },
    { match: ['泊松过程'], def: '随机事件以恒定平均速率独立发生的过程，常用来模拟请求的到达。', url: WP('泊松过程') },
    { match: ['离散事件模拟'], def: '只在"事件"（请求到达、一步计算完成）发生的时刻推进时钟的模拟方法，比逐毫秒模拟快得多。', see: 'stage4/boss-cluster-commander', url: WP('离散事件模拟') },
    { match: ['Zipf 分布', 'Zipf'], def: '一种长尾分布：第 k 名的频率与 1/k^s 成正比，少数几项占了大部分。', url: WP('齊夫定律') },

    // ---------------------------------------------------------------- 应用
    { match: ['RAG'], def: '检索增强生成：先从知识库里检索相关文档，拼进提示词，再让模型回答。', see: 'stage4/04-advanced-kv-cache', url: WP('检索增强生成') },
    { match: ['Agent'], def: '由大模型驱动、能调用工具、多步完成任务的系统。模型在一个循环中被反复调用。', see: 'final/07-agentic-serving' },
    { match: ['MCP'], def: 'Model Context Protocol：Anthropic 提出的开放协议，规定模型怎样发现和调用外部工具与数据源。', url: 'https://modelcontextprotocol.io/' },
    { match: ['JSON Schema'], def: '描述 JSON 数据应有哪些字段、什么类型的规范。约束解码据此保证模型输出合法。', url: 'https://json-schema.org/' },
    { match: ['RLHF'], def: '基于人类反馈的强化学习：用人类偏好训练一个奖励模型，再用强化学习让模型生成高奖励的回答。', see: 'final/06-rl-serving', url: WPE('Reinforcement_learning_from_human_feedback') },
    { match: ['GRPO'], def: '一种强化学习算法：对同一个提示词生成一组回答，用组内的相对好坏作为优势，省掉价值模型。', see: 'final/06-rl-serving' },
    { match: ['PPO'], def: '近端策略优化，最常用的强化学习算法之一，限制每次更新的幅度以保持训练稳定。', url: WPE('Proximal_policy_optimization') },
    { match: ['safetensors'], def: 'Hugging Face 的权重文件格式，安全（不执行代码）、可以零拷贝地映射到内存。', url: 'https://huggingface.co/docs/safetensors' },
    { match: ['GGUF'], def: 'llama.cpp 使用的模型文件格式，把权重、分词器和元数据放在一个文件里，支持多种量化。', see: 'final/05-edge-ai' },
    { match: ['SSE'], def: 'Server-Sent Events：服务端通过一个保持打开的 HTTP 连接持续推送数据，流式输出 token 就用它。', see: 'stage1/01-serving-from-scratch', url: WPE('Server-sent_events') },
  ];

  const SKIP = '.markdown-section > blockquote:first-of-type, em, pre, code, a, h1, h2, h3, h4, h5, h6, .llm-widget, .katex, .katex-display, .mermaid, script, style, .term, button, summary, .docsify-tabs__tab, .count';
  const ascii = (s) => /^[\x00-\x7f]+$/.test(s);
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // 所有写法按长度从长到短，组成一个正则
  const forms = [];
  TERMS.forEach((t, i) => t.match.forEach((m) => forms.push({ m, i })));
  forms.sort((a, b) => b.m.length - a.m.length);
  const formIndex = new Map(forms.map((f) => [f.m, f.i]));
  const RE = new RegExp(forms.map((f) => (ascii(f.m) ? `(?<![A-Za-z0-9_\\-/.])${esc(f.m)}(?![A-Za-z0-9_\\-])` : esc(f.m))).join('|'), 'g');

  let pop = null, current = null;

  function sectionTitle(route) {
    const a = document.querySelector(`.sidebar-nav a[href="#/${route}"]`);
    return a ? a.textContent.replace(/\s+/g, ' ').trim() : route;
  }

  function ensurePop() {
    if (pop) return pop;
    pop = document.createElement('div');
    pop.className = 'term-pop';
    pop.setAttribute('role', 'tooltip');
    pop.id = 'term-pop';
    pop.hidden = true;
    document.body.appendChild(pop);
    pop.addEventListener('mouseenter', () => clearTimeout(pop._t));
    pop.addEventListener('mouseleave', hideSoon);
    return pop;
  }

  function show(el) {
    const t = TERMS[+el.dataset.term];
    const p = ensurePop();
    clearTimeout(p._t);
    const here = (location.hash.replace(/^#\//, '').split('?')[0] || '').replace(/\.md$/, '');
    const links = [];
    if (t.see && t.see !== here) links.push(`<a href="#/${t.see}">${sectionTitle(t.see)}</a>`);
    if (t.url) links.push(`<a href="${t.url}" target="_blank" rel="noopener">${/wikipedia/.test(t.url) ? '维基百科' : '外部资料'}<span aria-hidden="true"> ↗</span></a>`);
    p.innerHTML = `<div class="term-pop-name">${el.dataset.label}</div><div class="term-pop-def">${t.def}</div>${links.length ? `<div class="term-pop-links">${links.join('')}</div>` : ''}`;
    p.hidden = false;
    const r = el.getBoundingClientRect();
    const w = Math.min(320, window.innerWidth - 24);
    p.style.width = w + 'px';
    let left = r.left + r.width / 2 - w / 2;
    left = Math.max(12, Math.min(left, window.innerWidth - w - 12));
    const ph = p.offsetHeight;
    const below = r.bottom + 8 + ph < window.innerHeight || r.top - ph - 8 < 0;
    p.style.left = left + window.scrollX + 'px';
    p.style.top = (below ? r.bottom + 8 : r.top - ph - 8) + window.scrollY + 'px';
    p.classList.toggle('is-above', !below);
    el.setAttribute('aria-describedby', 'term-pop');
    if (current !== el) p._shownAt = Date.now();
    current = el;
  }
  function hide() {
    if (!pop) return;
    pop.hidden = true;
    if (current) current.removeAttribute('aria-describedby');
    current = null;
  }
  function hideSoon() {
    if (!pop) return;
    clearTimeout(pop._t);
    pop._t = setTimeout(hide, 180);
  }

  function apply(root, route) {
    if (!root) return;
    hide();
    const seen = new Set();
    // 跳过：指定不标注的页面；以及正是讲这个术语的那一节
    TERMS.forEach((t, i) => { if ((t.skip && t.skip.includes(route)) || t.see === route) seen.add(i); });
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const p = n.parentElement;
        if (!p || p.closest(SKIP)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      const text = node.nodeValue;
      RE.lastIndex = 0;
      let m, last = 0;
      const frag = document.createDocumentFragment();
      let changed = false;
      while ((m = RE.exec(text))) {
        const i = formIndex.get(m[0]);
        if (i === undefined || seen.has(i)) continue;
        seen.add(i);
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const span = document.createElement('span');
        span.className = 'term';
        span.tabIndex = 0;
        span.dataset.term = i;
        span.dataset.label = m[0];
        span.textContent = m[0];
        frag.appendChild(span);
        last = m.index + m[0].length;
        changed = true;
      }
      if (!changed) return;
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  // 事件委托：悬停、聚焦、点按。触屏上点按后浏览器会补发 mouseout，不能据此隐藏
  let lastTouch = 0;
  document.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') lastTouch = Date.now(); }, true);
  const touching = () => Date.now() - lastTouch < 1500;
  document.addEventListener('mouseover', (e) => { const t = e.target.closest && e.target.closest('.term'); if (t) show(t); });
  document.addEventListener('mouseout', (e) => { const t = e.target.closest && e.target.closest('.term'); if (t && !touching()) hideSoon(); });
  document.addEventListener('focusin', (e) => { if (e.target.classList && e.target.classList.contains('term')) show(e.target); });
  document.addEventListener('focusout', (e) => { if (e.target.classList && e.target.classList.contains('term') && !touching()) hideSoon(); });
  document.addEventListener('click', (e) => {
    const t = e.target.closest && e.target.closest('.term');
    if (t) { if (current === t && pop && !pop.hidden && Date.now() - (pop._shownAt || 0) > 500) hide(); else show(t); return; }
    if (pop && !pop.hidden && !e.target.closest('.term-pop')) hide();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
  window.addEventListener('scroll', () => { if (current && !touching() && !current.matches(':hover')) hide(); }, { passive: true });

  window.LLMTerms = { apply, TERMS };
})();
