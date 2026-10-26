# 🧰 实验环境搭建

## 硬件分级

| 档位 | 配置 | 能打的关 |
|---|---|---|
| 入门 | 1× 24GB（4090 / A10） | 阶段 ⓪ ①、项目 1、项目 2（8B 模型） |
| 标准 | 1× 80GB（A100 / H100 / A800） | 全部单卡内容、项目 3 |
| 集群 | ≥ 2× 80GB | 阶段 ④、项目 4、毕业项目 A |

> [!TIP]
> FP8 计算需要 Hopper（H100/H800）或 Ada（4090/L40S）及以上；Ampere（A100/A800）只能做 FP8 **存储**（权重/KV），计算会回落。做项目 2 之前先确认你的卡。

> [!WARNING]
> 阶段 ⑥ 的 DeepInfer 目前只支持 NVIDIA SM100（Blackwell，CUDA 13）和华为昇腾 950（CANN 9.2），A100/A800/H100/4090 都跑不了。这一阶段以读源码为主，再把思路移植到手头的硬件上验证。

## 软件栈

```bash
# 建议用 uv 管理环境
uv venv --python 3.11 && source .venv/bin/activate

uv pip install torch transformers accelerate
uv pip install vllm                 # 阶段一起的主力框架
uv pip install "sglang[all]"        # RadixAttention / 对比实验
uv pip install triton               # 阶段三手写算子（通常随 torch 一起装）
uv pip install lm-eval              # 阶段二量化质量评估
```

## 第一次冒烟测试

```bash
vllm serve Qwen/Qwen3-0.6B --max-model-len 4096

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"用一句话解释 KV Cache"}]}'
```

## 压测工具

```bash
# vLLM 自带的 serving benchmark
vllm bench serve --model Qwen/Qwen3-0.6B \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 200 --request-rate 8
```

> [!NOTE]
> 不同 vLLM 版本的 benchmark 入口有变化（老版本是 `benchmarks/benchmark_serving.py`），以你安装的版本 `--help` 为准，并在学习日志里记下版本号。
