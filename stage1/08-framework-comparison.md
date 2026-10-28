# 1.8 推理框架对比：vLLM、SGLang、TensorRT-LLM 与 llama.cpp

## 学习目标

完成本节学习后，你将能够：
- **总结为什么传统深度学习服务框架（如 TorchServe、TF Serving）无法胜任大模型推理**。
- **准确把握四大主流开源框架（vLLM、SGLang、TensorRT-LLM、llama.cpp）的架构侧重与选型边界**。
- **掌握企业级框架选型的六步评估法**，学会用 OpenAI 兼容接口构建可随时平替的抽象层。

---

## 1. 通用模型服务框架为什么不够用

经过前 7 节的学习，你已经把现代大模型推理引擎的五脏六腑摸得一清二楚了。现在回头看传统的通用深度学习服务框架（如 TorchServe、TensorFlow Serving），你就明白为什么它们在大模型时代几乎无人问津：

| 大模型服务的独特挑战 | 传统通用服务框架的做法 | 专用大模型推理引擎（vLLM / SGLang / TRT-LLM）的做法 |
|---|---|---|
| **自回归多步循环（1.2）** | 把整个请求当成 1 次黑盒函数调用 | 拆解为**单步迭代（Step）**级别调度，每生成 1 个词重新编批 |
| **动态膨胀的 KV Cache（1.3 & 1.6）** | 每次分配连续大张量，碎片浪费 > 60% | **PagedAttention 分页块池**，消灭外部碎片，显存利用率接近 100% |
| **长短不一的输入输出（1.5）** | 静态或动态整批等待，产生大量空泡 | **Continuous Batching + Chunked Prefill**，随完随走、切块混跑 |
| **大量共享前缀（1.7）** | 每次从头重算整段 Prompt | **块哈希或 RadixAttention 前缀缓存**，自动跳过重复 Prefill |
| **严苛的显存带宽瓶颈（0.4）** | 标准 FP32/FP16 算子 | 深度定制 **W4A16 / FP8 / NVFP4 量化算子**、FlashAttention 与投机采样 |

---

## 2. 主流推理框架对比

目前工业界与开源社区最核心的四个推理框架，各自占据着非常清晰的生态位：

```mermaid
flowchart TB
  subgraph Cloud["数据中心 / 云端高并发 GPU 服务"]
    V["vLLM<br/>【通用标准 · 生态最大】<br/>模型覆盖最广，架构清晰易扩展"]
    S["SGLang<br/>【Agent · 结构化输出利器】<br/>RadixAttention + 快速正则/JSON 约束解码"]
    T["TensorRT-LLM<br/>【NVIDIA 硬件极致压榨】<br/>深度绑定 CUDA / FP8 / NVFP4 编译优化"]
  end
  subgraph Edge["本地电脑 / 边缘端 / CPU 低并发服务"]
    L["llama.cpp (及 Ollama)<br/>【轻量纯 C/C++ · 跨平台全能】<br/>GGUF 量化，Mac Metal / CPU / 消费级显卡开箱即跑"]
  end
```

| 对比维度 | **vLLM** | **SGLang** | **TensorRT-LLM** | **llama.cpp** |
|---|---|---|---|---|
| **核心定位** | 最通用的高吞吐在线服务引擎 | 面向复杂 Agent、多轮分支与结构化输出 | 在 NVIDIA GPU 上压榨极致性能与最低成本 | 笔记本、CPU、边缘设备的低并发本地推理 |
| **实现语言** | Python 调度 + CUDA/Triton 算子 | Python 调度 + CUDA/Triton 算子 | C++ 核心运行时 + Python 封装 | 纯 C / C++（零重型依赖） |
| **前缀缓存机制** | 块级链式哈希（Block Hashing） | **RadixAttention（基数树）** + 缓存感知调度 | 块级 KV 复用 | 会话级 Prompt 缓存 |
| **杀手锏特性** | 新模型 Day-0 支持、社区插件最全、V1 架构极其稳健 | 极速 JSON/EBNF 约束解码、高效多模态与 DeepSeek 深度优化 | NVIDIA 官方算子库（FP4/FP8/通信）最快落地 | GGUF 格式 2~8 bit 量化、支持 CPU+GPU 混合卸载 |
| **支持的硬件** | NVIDIA、AMD、TPU、昇腾、CPU 等 | NVIDIA、AMD、TPU、昇腾、CPU 等 | **仅限 NVIDIA GPU** | 几乎所有硬件（Mac M系列、x86/ARM CPU、各家 GPU） |
| **上手与排障难度** | （极易上手） | （极易上手） | （较重，需对齐驱动与编译环境） | （单二进制文件即跑） |

---

## 3. 组件：按场景选框架

点击下方选型器中的不同业务场景，查看推荐的推理框架、核心理由以及对应的启动命令：

<div class="llm-widget" data-widget="framework-advisor"></div>

### 代码写法对比：同一实验在四大框架中的调用方式

<!-- tabs:start -->
#### **vLLM**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-8B", dtype="bfloat16")
out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64))
print(out[0].outputs[0].text)
```
#### **SGLang**
```python
import sglang as sgl

llm = sgl.Engine(model_path="Qwen/Qwen3-8B", dtype="bfloat16")
out = llm.generate(["The capital of France is"], {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 64})
print(out[0]["text"])
```
#### **TensorRT-LLM**
```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-8B")
for out in llm.generate(["The capital of France is"], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)):
    print(out.outputs[0].text)
```
#### **llama.cpp (Python 绑定)**
```python
from llama_cpp import Llama

llm = Llama.from_pretrained(repo_id="Qwen/Qwen3-8B-GGUF", filename="*Q4_K_M.gguf", verbose=False)
print(llm("The capital of France is", max_tokens=64)["choices"][0]["text"])
```
<!-- tabs:end -->

---

## 4. 选型步骤

不要因为网上某篇测评说“框架 X 比框架 Y 快 20%”就盲目跟风切换。大模型框架的性能对**硬件型号、模型架构（稠密 vs MoE、MHA vs GQA/MLA）、输入输出长度比、前缀重合度**极度敏感。科学的选型必须遵循以下六步：

1. **先写下硬性的 SLO 与质量约束**：比如 `TTFT P99 < 800ms`、`TPOT P99 < 40ms`、是否必须 100% 符合 JSON Schema。
2. **用自己的真实业务流量画像做测试集**：统计线上真实的 Prompt 长度分布、输出长度分布、以及多轮对话的前缀重合比例（千万别拿别人的合成短数据结论套自己的长文档业务）。
3. **对齐一切基准参数做苹果对苹果（Apples-to-Apples）对比**：统一模型权重、统一精度（BF16 或 FP8）、统一 `--max-model-len`、统一开启或关闭前缀缓存，用同一个外部压测客户端（如本仓库的 `code/project1/bench.py`）施压。
4. **考察生产可运维性**：冷启动要多久？崩溃报错好不好查？Prometheus 监控指标全不全？
5. **评估多硬件可移植性**：如果公司既有 NVIDIA 卡又有国产卡或 AMD 卡，优先选 vLLM / SGLang。
6. **用 OpenAI 兼容接口做抽象隔离**：业务代码永远只调标准的 `/v1/chat/completions` 接口。这样每隔半年框架格局变化时，只需改一行 Docker 镜像配置就能无缝切换后端。

---

## 5. 实验：用同一套脚本对比 vLLM 与 SGLang

在 GPU 上分别用完全相同的参数启动 vLLM 和 SGLang，然后用 `code/project1/bench.py` 在两组典型负载下（**负载 A：短输入长输出聊天** vs **负载 B：4096 共享长前缀文档问答**）同台竞技：

```bash
# 分别启动（测试完一个关掉再启动另一个，确保独占同一张 GPU）
vllm serve Qwen/Qwen3-8B --max-model-len 8192 --port 8000
# 或
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --context-length 8192 --port 8000

# 负载 A：聊天型（输入 256，输出 512，无共享前缀）
python code/project1/bench.py --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
  --num-prompts 200 --input-len 256 --output-len 512 --request-rate 8 --vocab-size 32000

# 负载 B：长文档问答（共享前缀 4096，独立问题 128，输出 128）
python code/project1/bench.py --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
  --num-prompts 200 --shared-prefix-len 4096 --input-len 128 --output-len 128 --request-rate 8 --vocab-size 32000
```

---

## 6. 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "要在没有独立 GPU 的服务器或 MacBook 上，为十几个人私有部署一个低并发的 8B 模型，最合适的选择是：", "options": ["vLLM 的 CPU 后端，开启张量并行", "llama.cpp（或基于它的 Ollama），配合 GGUF 的 4~8 位量化", "SGLang，开启 RadixAttention 共享前缀", "TensorRT-LLM，先把模型编译成 CPU 引擎"], "answer": 1, "explain": "llama.cpp 针对 CPU 与 Apple Silicon 做了大量优化，GGUF 量化能把 8B 模型压到 5~9 GB，低并发下体验最好。vLLM 与 SGLang 的优势在 GPU 上的高并发调度，TensorRT-LLM 只支持 NVIDIA GPU。"}, {"q": "看到一篇评测说框架 A 在 ShareGPT 数据集上比框架 B 快 15%，为什么不应该据此直接迁移线上服务？", "options": ["ShareGPT 数据集已经过时，评测结果不再有参考价值", "性能高度依赖负载（输入输出长度、前缀共享率、结构化输出）、模型结构、硬件和并发度；应该用自己的流量和 SLO 复现对比", "框架之间的性能差距通常小于测量误差", "迁移框架需要重新训练模型"], "answer": 1, "explain": "同一组框架在不同负载下的排名经常互换。评测时还要对齐版本、量化方式和关键参数，并比较满足 SLO 的 Goodput，而不只是峰值吞吐。"}, {"q": "为了避免业务代码与某一个推理框架绑定，最简单有效的做法是：", "options": ["在业务代码中同时引入多个框架的 Python SDK，按配置切换", "业务侧只调用 OpenAI 兼容的 HTTP/SSE 接口，框架差异封装在网关之后", "把模型导出成 ONNX，所有框架都从 ONNX 加载", "固定使用一个框架的某个版本，不再升级"], "answer": 1, "explain": "主流框架都提供 OpenAI 兼容接口，换框架时只需替换后端与网关配置。直接依赖各框架的 Python 接口会把引擎细节带进业务代码。ONNX 并不被这些 LLM 服务框架普遍作为主加载格式。"}]'></div>

---

## 延伸阅读

- Chi Wang & Peiheng Hu, *Hands-On LLM Serving and Optimization*, O'Reilly, Chapter 8 (*LLM Serving Frameworks*)
- 下一站实战闯关：**[项目 1 · MiniServe：亲手从零实现一个迷你推理引擎](stage1/boss-miniserve.md)**。
