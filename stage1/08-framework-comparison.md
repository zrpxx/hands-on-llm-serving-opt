# 1.8 推理框架对比

> 对应书中第 8 章 Why We Need Specialized LLM Serving Frameworks、TensorRT-LLM、SGLang、Llama.cpp、Selecting the Right Framework

## 学习目标

- 说明通用模型服务框架为什么不适合 LLM，专用框架补上了哪些能力。
- 了解 vLLM、SGLang、TensorRT-LLM、llama.cpp 的定位与适用场景。
- 掌握书中推荐的框架评估方法，并用同一套压测工具做一次公平对比。

## 1. 为什么需要专用框架

TensorFlow Serving、TorchServe、Triton Inference Server 等通用框架为图像分类、推荐等传统模型设计：输入短、张量形状固定、延迟可预测，主要优化手段是批处理。LLM 带来的新问题（书中归纳）：

| 问题 | 说明 |
|---|---|
| 自回归生成 | 一个请求要持续几秒到几分钟 |
| 上下文长度跨度大 | 从几个 token 到上百万 token，KV Cache 管理成为瓶颈 |
| 需要连续批处理 | 输入输出长度差异大，静态批处理浪费 GPU |
| 流式输出 | 用户期望几百毫秒内看到首 token，之后连续输出 |
| 资源利用率 | GPU 昂贵，碎片和空转不可接受 |

专用框架普遍具备分页 KV Cache、连续批处理、LLM 专用量化、投机解码等能力，这些在前几节已经逐一介绍。

## 2. 四个主流框架

### 2.1 vLLM

本笔记 1.6 节已经详细介绍。书中对它的评价：部署快、模型覆盖广、默认配置下就有不错的性能；架构清晰，便于集成新的研究成果；社区和生态最大。

### 2.2 SGLang

定位是结构化生成和 Agent 类应用。主要特性：RadixAttention 前缀复用、连续批处理、分页 KV、投机解码（EAGLE 系列）、chunked prefill、结构化输出（JSON / 正则 / EBNF）、Multi-LoRA 批处理、多种并行方式。

```python
import sglang as sgl
llm = sgl.Engine(model_path="Qwen/Qwen3-8B")
outputs = llm.generate(["The capital of France is"], {"temperature": 0.8, "top_p": 0.95})
```

硬件覆盖面比 TensorRT-LLM 广，官方文档涵盖 NVIDIA、AMD、CPU、TPU、Jetson、昇腾。书中认为它在功能上与 vLLM 对等，部分负载下性能更好；vLLM 的社区规模目前更大。

### 2.3 TensorRT-LLM

NVIDIA 的开源推理库，目标是在 NVIDIA GPU 上发挥出最高性能。支持 in-flight batching（即连续批处理）、分页 KV、投机解码、FP8 / FP4 / INT4 / INT8 等多种精度、张量与流水线并行，与 NVIDIA 的 Triton 和 Dynamo 深度集成。

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-8B")
for out in llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8, top_p=0.95)):
    print(out.outputs[0].text)
```

书中的建议：已经全面采用 NVIDIA 软硬件栈、追求单位成本最优的团队适合选它。

### 2.4 llama.cpp

轻量的 C/C++ 推理实现，能在笔记本、工作站、边缘设备上运行。模型使用 GGUF 格式，支持 2 到 8 位的整数量化，后端覆盖 CPU（SIMD）、Metal、CUDA / ROCm、Vulkan，自带 OpenAI 兼容的 HTTP 服务。Ollama 是在它之上封装的更易用的本地工具。

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="Qwen/Qwen3-8B-GGUF", filename="*Q8_0.gguf", verbose=False)
print(llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"]))
```

书中指出，本地推理的目标与数据中心不同：并发低（常常只有一个用户），大批量和复杂调度意义不大，重点是延迟、资源占用、成本、隐私与离线可用。llama.cpp 正是为这类场景设计的。

### 2.5 对比

| | vLLM | SGLang | TensorRT-LLM | llama.cpp |
|---|---|---|---|---|
| 主要语言 | Python（核心算子为 CUDA / Triton） | Python（核心算子为 CUDA / Triton） | Python / C++ | C / C++ |
| 定位 | 通用在线服务 | 结构化生成、Agent、多步调用 | NVIDIA GPU 上的极致性能 | 本地、边缘、低资源 |
| 前缀复用 | 块哈希 | RadixAttention | 支持 | 支持（单会话为主） |
| 硬件 | NVIDIA、AMD、TPU、CPU 等 | NVIDIA、AMD、TPU、CPU、昇腾等 | NVIDIA | 几乎所有 |
| 上手难度 | 低 | 低 | 中 | 低 |
| 书中推荐场景 | 最快上线、模型覆盖广 | Agent、严格结构化输出、多厂商硬件 | 全栈 NVIDIA、追求单位成本 | 本地开发、私有部署、边缘 |

## 3. 评估方法

书中的原则是：**按负载选，不按热度选。**

1. **从 SLO 出发，而不是从功能清单出发。** 写下延迟目标（TTFT、P95 / P99）、吞吐、单位成本、质量约束（如结构化输出）和可用性要求。
2. **分析真实的 prompt。** 是 prefill 为主还是 decode 为主？上下文多长？有没有工具调用和多轮链式调用？
3. **公平比较。** 相同的模型、精度、最大长度、并发设置、是否流式。
4. **考察可运维性。** 冷启动时间、可观测性、自动扩缩容行为、多租户公平性、升级成本、故障模式。
5. **考虑硬件与厂商锁定。** 如果使用多种硬件，可移植性权重要高。
6. **为变化做准备。** 模型和解码技术更新很快，选择容易升级的框架。

书中作者的做法是每三到六个月重新评估一次，并在服务中保留一层框架抽象，随时可以替换。

> [!TIP]
> OpenAI 兼容接口本身就是一层很好的抽象。上层业务只依赖这个接口，下层的推理框架可以随时替换，也便于用同一个压测工具横向比较。

## 动手实验

用项目 1 的 `bench.py` 对 vLLM 和 SGLang 做一次公平对比。两者都使用同一个模型、BF16、相同的最大长度：

```bash
# vLLM
vllm serve Qwen/Qwen3-8B --max-model-len 8192 --port 8000
# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --context-length 8192 --port 30000
```

设计两组负载：

| 负载 | 参数 |
|---|---|
| 聊天型（decode 为主） | 输入 256、输出 512，泊松到达，速率 1 → 16 |
| 长文档问答（prefill 为主，有共享前缀） | 共享前缀 4096、输入 128、输出 128，速率 1 → 16 |

| 负载 | 框架 | 吞吐峰值 | Goodput 峰值 | 该点 TTFT P99 | 该点 TPOT P99 |
|---|---|---|---|---|---|
| 聊天型 | vLLM | | | | |
| 聊天型 | SGLang | | | | |
| 长文档 | vLLM | | | | |
| 长文档 | SGLang | | | | |

注意记录两个框架的版本号和启动参数，并说明差异可能来自哪里（默认的 token 预算、前缀缓存实现、CUDA Graph 覆盖的批大小等）。

## 自测题

1. 一个团队要在内网的 CPU 服务器上为 20 个人提供文档问答，应优先考虑哪个框架？
2. 为什么说"在别人的 benchmark 上更快"不能作为选型依据？
3. 一个 Agent 应用需要模型严格输出 JSON，并且大量请求共享很长的系统提示词。书中推荐哪个框架？理由是什么？
4. 书中建议保留"框架抽象层"。在实践中可以怎样做？

<details>
<summary>参考答案</summary>

1. llama.cpp（或基于它的 Ollama）。CPU 推理、并发低、部署简单，配合 4 位或 8 位量化可以在普通服务器上运行。
2. 别人的负载（输入输出长度分布、并发、是否有共享前缀）、硬件、精度、参数可能与你完全不同，而性能对这些因素非常敏感。必须在自己的负载和 SLO 下测。
3. SGLang。它对结构化输出和多步 Agent 调用有专门的优化，RadixAttention 对长系统提示词的复用很有效。
4. 让业务只依赖 OpenAI 兼容接口；把模型名、地址、采样参数放在配置里；保持一套与框架无关的压测与回归测试；在网关层做路由，便于灰度切换。

</details>

## 延伸阅读

- 各框架的官方文档与 GitHub 仓库
- 书中第 9 章：用 vLLM 端到端优化 Qwen3-14B（对应本笔记 5.1 节）
