# 实验环境

原理验证和代码开发在普通笔记本的 CPU 上就能完成，性能实验需要 GPU。这一页列出不同硬件能完成哪些内容，以及安装和验证的步骤。

## 1. 硬件与可完成的内容

| 硬件 | 例子 | 可以完成 | 建议 |
|---|---|---|---|
| 只有 CPU | 任意笔记本，8 GB 以上内存 | 全部交互组件，`estimate_memory.py`，项目 1、2 的全部开发和正确性测试，项目 2 的小模型对比实验 | 日常阅读和写代码 |
| 一张 24 GB 的卡 | RTX 4090、RTX 3090、A10、L4 | 第 0、1 部分的 GPU 实验（Qwen3-0.6B 或量化后的 8B），项目 1 的性能实验，项目 2 的真实模型实验 | 云平台按小时租用 |
| 一张 48～80 GB 的卡 | L40S、A100、A800、H100 | BF16 的 Qwen3-8B、14B 高并发压测，项目 3 | 需要完整的压测曲线时租几个小时 |
| 多张卡 | 2 张 A100，8 张 H200 等 | 第 4 部分的并行与 PD 分离实验，项目 4 | 只在做第 4 部分时需要 |

关于 FP8：Ada（RTX 4090、L40S、L4，计算能力 8.9）、Hopper（H100、H200、H20，9.0）和 Blackwell 支持 FP8 计算。Ampere（A100、A800 为 8.0，RTX 3090、A10 为 8.6）不支持，在这些卡上使用 FP8 的 KV Cache 或权重只能节省显存和带宽，计算前要转换回 BF16（0.5 节）。

## 2. 安装

建议用 [uv](https://docs.astral.sh/uv/) 或 conda 建立单独的 Python 3.11 或 3.12 环境：

```bash
uv venv --python 3.11
source .venv/bin/activate

# 基础依赖，以及项目 1 的测试依赖
uv pip install torch transformers safetensors tokenizers fastapi uvicorn httpx pytest

# 有 NVIDIA GPU 时
uv pip install vllm              # 第 0 部分起的对照引擎
uv pip install "sglang[all]"     # 1.4、1.5 节的对比实验
uv pip install lm-eval           # 第 2 部分的精度评测
```

Triton 一般随 PyTorch 一起安装，第 3 部分会用到。

## 3. 下载模型

最常用的两个模型：

- **Qwen/Qwen3-0.6B**：权重约 1.2 GB，适合第一次跑通流程，以及在小显存的卡上调试。
- **Qwen/Qwen3-8B**：权重约 16.4 GB，用来观察真实的显存瓶颈和并发曲线。

在国内直接访问 Hugging Face 较慢，可以用镜像或魔搭：

<!-- tabs:start -->
#### **Hugging Face 镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
# 较旧版本的 huggingface_hub 使用 huggingface-cli download
```
#### **魔搭 ModelScope**
```bash
uv pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```
<!-- tabs:end -->

vLLM 也可以直接从魔搭加载：设置环境变量 `VLLM_USE_MODELSCOPE=True`。

## 4. 验证环境

### 4.1 CPU

不需要下载任何模型：

```bash
# 显存估算
python code/stage0/estimate_memory.py code/stage0/configs/qwen3-8b.json --gpu-mem 80 --max-len 4096

# 生成一个随机权重的小模型，运行项目 1 参考实现的测试
python code/project1/tools/make_tiny_model.py --out /tmp/tiny --layers 4 --hidden 128
MINISERVE_IMPL=solution pytest -q code/project1/tests
```

测试全部通过，说明 CPU 上的环境已经就绪。

### 4.2 GPU

```bash
vllm serve Qwen/Qwen3-0.6B --max-model-len 4096 --port 8000
```

另开一个终端：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用一句话解释什么是 KV Cache"}],
    "max_tokens": 64
  }'
```

收到 JSON 格式的回答，环境就准备好了。接下来从 [0.1 模型服务概览](stage0/01-serving-landscape.md) 开始。
