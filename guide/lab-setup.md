# 实验环境准备指南

在开始动手实验前，花 15 分钟把本地或云端开发环境配置好。本教程的设计原则是：**原理验证与代码开发可在普通 CPU 笔记本上完成，性能压测在 GPU 上运行。**

---

## 1. 硬件档位与可完成的实验对照

| 硬件档位 | 典型配置 | 可完成的教程内容 | 租用/使用建议 |
|---|---|---|---|
| **零 GPU（纯 CPU / Mac）** | 任意笔记本（8GB+ 内存） | 全部交互组件、`code/stage0/estimate_memory.py`、**项目 1 MiniServe 全部代码实现与 `pytest` 正确性测试** | 日常看教程、写代码、跑单元测试首选 |
| **入门单卡（24 GB 显存）** | RTX 4090 / 3090 / NVIDIA A10 / L4 | 第 0、1 部分全部 GPU 实验（使用 `Qwen3-0.6B` 或量化后的 `8B`）、项目 1 GPU 压测、项目 2 量化评测 | 云平台按量租用约 ¥1.5~3 / 小时 |
| **标准单卡（48~80 GB 显存）** | L40S (48GB) / A100 / A800 / H100 (80GB) | 全精度 `Qwen3-8B` / `14B` 高并发压测、项目 3 Triton 算子与投机采样 | 跑正式压测曲线时租用 2~3 小时即可 |
| **多卡环境（≥ 2× GPU）** | 2× A10 / 2× A100 / 8× H200 | 第 4 部分（TP/PP/DP 对比、PD 分离）、项目 4 Cluster Commander | 仅在做第 4 部分时按需开启 |

> [!TIP]
> **硬件架构小贴士**：
> - **FP8 张量计算**需要 NVIDIA Ada Lovelace（RTX 4090、L40S，计算能力 SM89）或 Hopper（H100、H200，计算能力 SM90）及以上架构。
> - Ampere 架构（如 A100、A800、RTX 3090，SM80）**不支持原生 FP8 Tensor Core 计算**——在这些卡上开启 FP8 KV Cache 或 W8A8 只能节省显存和带宽，计算时仍会反量化回 BF16/FP16。这一点在第 0.2 节和第 2 部分会重点剖析。

---

## 2. Python 环境与依赖安装

推荐使用 [`uv`](https://docs.astral.sh/uv/) 或 `conda` 创建独立的 Python 3.11 虚拟环境，避免包版本冲突：

```bash
# 1. 创建并激活 Python 3.11 虚拟环境
uv venv --python 3.11
source .venv/bin/activate

# 2. 安装基础深度学习依赖与项目 1 测试依赖
uv pip install torch transformers accelerate fastapi uvicorn httpx pytest

# 3. （有 NVIDIA GPU 时）安装主流推理引擎与算子工具
uv pip install vllm                 # 第 0、1 部分起的主力对比引擎
uv pip install "sglang[all]"        # 1.7 RadixAttention 与 1.8 框架对比实验
uv pip install triton               # 第 3 部分手写 GPU 算子（通常随 PyTorch 自动安装）
uv pip install lm-eval              # 第 2 部分量化精度评测
```

---

## 3. 模型权重下载与国内加速技巧

教程中最常使用的两个实验模型是：
- **`Qwen/Qwen3-0.6B`**（约 1.2 GB）：轻量极速，适合第一次跑通流程、笔记本或小显存卡调试。
- **`Qwen/Qwen3-8B`**（约 16.4 GB）：主流生产级尺寸，适合观察真实的显存瓶颈与并发曲线。

如果直接访问 Hugging Face 较慢，可以使用以下两种加速方式之一：

<!-- tabs:start -->
#### **方式 A：HF 镜像环境变量**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```
#### **方式 B：使用 ModelScope（魔搭）**
```bash
uv pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```
<!-- tabs:end -->

---

## 4. 5 分钟冒烟验证（Smoke Test）

### 4.1 纯 CPU 环境冒烟测试（无需下载大模型）

即使你现在用的是没有独立显卡的轻薄本，也可以立刻跑通我们内置的显存估算器和 MiniServe 引擎：

```bash
# 1. 运行显存与并发容量估算器
python code/stage0/estimate_memory.py code/stage0/configs/qwen3-8b.json --gpu-mem 80 --max-len 4096

# 2. 生成一个随机权重的微型 Qwen3 模型（仅几 MB）并在 CPU 上运行 MiniServe 测试套件
python code/project1/tools/make_tiny_model.py --out /tmp/tiny --layers 4 --hidden 128
MINISERVE_IMPL=solution pytest -q code/project1/tests
```

看到 `passed` 输出，说明你的本地代码环境已经完全就绪。

### 4.2 GPU 环境第一次启动 vLLM

如果你在 GPU 机器上，运行以下命令启动你的第一个在线推理服务：

```bash
# 启动 vLLM 服务（加载 Qwen3-0.6B）
vllm serve Qwen/Qwen3-0.6B --max-model-len 4096 --port 8000
```

在另一个终端窗口发送一条标准 OpenAI 格式的请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "请用一句话向初学者解释什么是 KV Cache"}],
    "max_tokens": 64
  }'
```

收到模型返回的 JSON 响应后，恭喜你完成了环境准备。接下来请进入 [0.1 模型服务概览](stage0/01-serving-landscape.md)。
