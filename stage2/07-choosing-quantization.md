# 2.7 怎样选择和评估量化方案

> 对应原书第 6 章 *Hands-on quantization*、*Accuracy trade-offs*、*Quantization-aware training*，以及第 9 章第 6 步。

前面几节分别介绍了各种量化方法。实际部署时要回答的问题是：对于这个模型、这批显卡、这种负载，用哪种方案？量化之后，怎样确认精度还够用、性能确实提高了？这一节把前面的内容串起来。

## 1. 先看瓶颈，再选方案

量化方案的选择取决于三件事：

- **显存够不够**。模型放不下，或者放下之后留给 KV Cache 的空间太少，并发上不去。这时首先需要的是压缩权重，W4A16 最有效。
- **瓶颈在带宽还是算力**。并发低、输出长的场景（聊天、Agent 的多轮调用），decode 是带宽瓶颈，W4A16 最快。并发高、输入长的场景（批处理、RAG），计算的比重变大，W8A8 更合适（2.6 节）。
- **显卡支持什么**。Hopper、Ada、Blackwell 支持 FP8 计算；Ampere 只能用 INT8 做 W8A8；Blackwell 还支持 FP4。

原书给出的经验是：如果模型需要压缩到 1/4 才能放进一张卡或一台机器，避免跨卡或跨机的通信，就用 W4A16；如果 W8A8 已经能满足延迟的要求，就尽量用 W8A8，把 batch 推大，提高每个实例的吞吐，降低成本。下面的组件按这个思路做了一个粗略的估算。

<div class="llm-widget" data-widget="quant-chooser"></div>

可以试试几个典型的组合：Qwen3-8B 放在 H100 上，各种方案都放得下，吞吐优先时 FP8 最合适；Qwen3-32B 放在 L40S 上，FP8 之后 KV Cache 的空间很紧张；Llama-3.1-70B 放在 H100 上，只有 4 位量化才能单卡运行。

还有一个容易被忽略的选项：**量化一个更大的模型，而不是用原精度的小模型**。原书举的例子是，FP8 的 12B 模型和 FP16 的 8B 模型延迟相当，但吞吐更高、效果更好。同样的显存预算，大模型的 4 位版本通常也比小模型的 BF16 版本效果更好。

## 2. 先找现成的量化模型

很多模型的发布方会同时发布量化版本，这些版本通常经过了更充分的校准和评测：

- Qwen 官方为 Qwen3 系列发布了 FP8（按块量化）、AWQ 和 GGUF 等版本；
- DeepSeek-V3、R1 系列的官方权重本身就是 FP8；
- Red Hat AI（原 Neural Magic）在 Hugging Face 上维护了大量 FP8、W4A16、W8A8 的模型，附有与原模型的精度对比。

只有在需要用自己的数据校准，或者模型是自己微调的时候，才需要自己量化（2.2 到 2.4 节的工具）。

## 3. 发布即量化：gpt-oss

OpenAI 在 2025 年 8 月发布的 gpt-oss 是一个值得单独说的例子。它有两个尺寸：gpt-oss-120b 共约 1170 亿参数，gpt-oss-20b 共约 210 亿参数，都是 MoE 模型。OpenAI 在后训练阶段就把 MoE 层的权重量化为 MXFP4，让模型在训练中适应这个精度。发布的 checkpoint 直接就是 4 位的，部署时不需要再选量化方法、做校准。

MXFP4 是一种微缩放格式：每 32 个数共用一个 8 位的指数 scale（E8M0），每个数本身是 FP4 E2M1，平均每个参数约 4.25 位。原书按每个参数 0.5 字节估算，120b 约 58.5 GB，可以放进一张 80 GB 的卡；20b 约 10.5 GB，可以在 16 GB 显存的设备上运行。实际上只有 MoE 层是 MXFP4，注意力等部分仍是 BF16，再加上 scale 的开销，官方 checkpoint 分别约为 60.8 GiB 和 12.8 GiB，结论不变。

这种做法的前提是硬件和推理引擎能高效地执行 MXFP4：

| 显卡 | 情况 |
|---|---|
| Blackwell（B200 等） | 有原生的 FP4 Tensor Core，效果最好 |
| Hopper（H100、H200） | 没有 FP4 计算单元，靠专门的混合精度 kernel 运行，性能仍然不错 |
| Ampere（A100 等） | 原书建议换用 Qwen 等其他模型自己量化。实际上 vLLM 也能在 Ampere 上用 Marlin 以 W4A16 的方式运行 MXFP4 权重，只是性能不如前两者 |

6.2 节会详细比较 MXFP4 和 NVFP4。

## 4. 评估精度

### 4.1 用任务评测，而不只是困惑度

前几节用困惑度比较量化方法，因为它算起来快、对误差敏感，适合在开发时快速比较。但最终要看的是模型在实际任务上的表现。原书推荐用 lm-evaluation-harness：

```bash
lm_eval --model vllm \
  --model_args pretrained=Qwen/Qwen3-8B-AWQ,max_model_len=4096 \
  --tasks gsm8k,mmlu \
  --batch_size auto
```

> [!NOTE]
> 原书的命令写成了 `--model_args Qwen/Qwen2-7B-Instruct`，缺少 `pretrained=`，直接运行会报错。用 `--model vllm` 时也不需要 `--device` 参数。

评测时有几点要注意：

- **和原模型在相同条件下对比**。同样的任务、同样的 prompt 模板、同样的采样参数，最好用贪心解码。常用的指标是恢复率，即量化模型的得分除以原模型的得分。
- **不要只看平均分**。平均分接近，不代表回答一样。有研究统计了量化前后每道题答案的变化（flips），发现即使平均分相差不到 1%，也可能有百分之几的题目由对变错、由错变对。如果业务对一致性敏感，应该统计这个比例，或者直接比较两个模型输出分布的 KL 散度。
- **覆盖实际的使用场景**。长上下文、长输出的推理模型、代码生成、多语言，这些场景对量化误差的敏感程度不同。推理模型一次生成几千个 token，每一步的小误差会累积。最可靠的评测集来自自己的业务数据。

### 4.2 一般的规律

Red Hat 的团队（Kurtić 等人）在 Llama-3.1 系列上做了超过 50 万次评测，原书图 6-20 引用的就是这项工作。主要结论：

- FP8 W8A8 在所有尺寸上都几乎无损；
- 调好的 INT8 W8A8 损失约 1% 到 3%；
- INT4 的 W4A16 与 INT8 W8A8 相当。

> [!NOTE]
> 原书在这里写道"模型越大，精度对权重和 KV Cache 量化越敏感"。多数公开的实验结论正好相反：参数量越大的模型通常越耐量化，同样的方案在小模型上损失更大。可以在 GPU 上对 Qwen3-0.6B 和 Qwen3-8B 用同一个方案量化，比较两者的恢复率来验证。

KV Cache 的 FP8 量化对精度的影响通常比权重量化小。原书的建议是：遇到 KV Cache 空间或 ITL 的问题时再用它，优先级排在权重量化之后。

## 5. 评估性能

性能要在**目标负载**下测，而不是只看单个请求的速度。原书的做法是用 `vllm bench serve` 在多个并发下分别测试原模型和各个量化版本，画出 TTFT、TPOT 和吞吐随并发变化的曲线：

```bash
for c in 1 4 16 64 128; do
  vllm bench serve --model Qwen/Qwen3-8B-AWQ \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts $((c * 8)) --max-concurrency $c \
    --save-result --result-filename awq_c${c}.json
done
```

需要关注的几点：

- **曲线的交叉点**。并发低时 W4A16 往往最快，并发高时可能被 FP8 甚至原模型超过。自己的业务落在曲线的哪一段，决定了该选哪个方案。
- **KV Cache 容量**。启动日志中 KV Cache 能容纳的 token 数，量化后应该明显增加。原书第 9 章的实验中，Qwen3-14B 换成 AWQ 版本后，这个数从约 7.2 万增加到约 19.1 万，吞吐提高了约 2.7 倍。
- **kernel 是否正确**。日志里确认用的是 Marlin、Machete 或 CUTLASS 的 FP8 kernel，而不是某个兜底的慢实现（2.6 节）。

## 6. PTQ 还是 QAT

到目前为止的方法都是训练后量化（PTQ），只需要模型权重和少量校准数据。量化感知训练（QAT）在训练或微调时就模拟量化的误差，让模型自己适应，精度更好，但需要完整的训练流程和算力。

原书表 6-5 认为，PTQ 在 4 位及以下"通常不可接受"。这个说法过于笼统。4 位的**权重**量化（GPTQ、AWQ 的 W4A16）用 PTQ 就能做到很小的损失，这正是它们被广泛使用的原因。PTQ 真正吃力的是 3 位及以下的权重，以及 4 位的激活（W4A4）。gpt-oss 用 QAT 的方式发布 MXFP4，正是因为 FP4 的格式更粗糙，并且 MoE 层占了绝大部分的参数。

对于大多数部署来说，实际的选择是：用别人已经量化好的模型（无论是 PTQ 还是 QAT 得到的），或者自己用 PTQ 量化。自己做 QAT 只有在需要极低位数、且掌握训练流程时才值得。

## 7. 一份检查清单

1. 估算显存：模型能不能放下，KV Cache 能容纳多少 token（0.6 节）。
2. 判断瓶颈：目标并发下是带宽瓶颈还是计算瓶颈（0.7 节、2.6 节）。
3. 确认显卡支持的格式：FP8、INT8、FP4。
4. 先找官方或可信来源的量化模型；没有再自己量化。
5. 在业务相关的任务上评测精度，和原模型对比，看恢复率和答案变化的比例。
6. 在目标负载下压测性能，确认 kernel 和 KV Cache 容量。
7. 仍不满足要求时，再考虑 FP8 KV Cache、更大的量化模型，或者多卡部署。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "一个聊天应用，并发通常只有几个请求，最在意每个请求的输出速度。显卡是 H100，模型放得下。优先考虑哪种方案？", "options": ["W4A16", "INT8 W8A8", "只量化 KV Cache", "不量化"], "answer": 0, "explain": "并发低时 decode 是带宽瓶颈，每一步的耗时主要是读权重。W4A16 读的字节最少，单个请求最快。并发升高后再考虑 FP8 W8A8。"}, {"q": "关于 gpt-oss 的 MXFP4，下面哪个说法正确？", "options": ["全部参数都是 4 位", "MoE 层的权重是 MXFP4，每 32 个数共用一个 8 位指数 scale，平均约 4.25 位；其他部分仍是 BF16", "MXFP4 只能在 Blackwell 上运行", "需要用户自己做 GPTQ 校准"], "answer": 1, "explain": "MoE 层占了 90% 以上的参数，所以只量化这部分就足以让 120b 放进一张 80 GB 的卡。Hopper 和 Ampere 上也能通过专门的 kernel 运行，只是没有原生的 FP4 计算单元。"}, {"q": "量化模型在一个评测集上的平均分只比原模型低 0.5%。还应该关注什么？", "options": ["不需要再关注什么，可以上线", "每道题的答案有多少发生了变化，以及在业务相关的长输出、长上下文任务上的表现", "只需要再测一次困惑度", "模型文件的大小"], "answer": 1, "explain": "平均分接近时，仍可能有一部分题目由对变错、由错变对，互相抵消。推理模型、长上下文等场景对误差更敏感，最可靠的评测来自自己的业务数据。"}, {"q": "原书表 6-5 认为 PTQ 在 4 位及以下通常不可接受。更准确的说法是：", "options": ["这个说法完全正确", "4 位的权重量化（W4A16）用 PTQ 就能做到很小的损失，PTQ 真正吃力的是 3 位以下的权重和 4 位的激活", "PTQ 在 8 位也不可接受", "QAT 不能用于 4 位"], "answer": 1, "explain": "GPTQ、AWQ 都是 PTQ，4 位权重量化已经广泛用于生产。权重和激活都到 4 位、或者权重到 2、3 位时，才更需要 QAT 或旋转等方法。"}]'></div>

## 延伸阅读

- Kurtić et al., ["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](https://arxiv.org/abs/2411.02355), ACL 2025。
- Dutta et al., [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141), NeurIPS 2024。讨论了量化前后答案变化（flips）的统计。
- OpenAI, [gpt-oss-120b & gpt-oss-20b Model Card](https://openai.com/index/gpt-oss-model-card/), 2025。
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
