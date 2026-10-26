# 项目 1：MiniServe

说明文档见站点中的「项目 1 MiniServe」页面（`stage1/boss-miniserve.md`）。

## 目录

| 路径 | 内容 |
|---|---|
| `starter/miniserve/` | 待完成的骨架，关键函数标有 `TODO(1-x)` |
| `solution/miniserve/` | 参考实现。建议自己完成后再对照 |
| `baseline_hf.py` | Hugging Face 逐请求基线（步骤 1-1） |
| `bench.py` | 压测客户端，适用于任意 OpenAI 兼容服务 |
| `tools/make_tiny_model.py` | 生成随机初始化的微型 Qwen3，用于 CPU 上开发 |
| `tests/` | 正确性测试：以 Hugging Face 的贪心输出为参照 |

## 常用命令

```bash
pip install -r requirements.txt

# 测试自己的实现（默认 starter）；只测调度器：pytest tests/test_scheduler.py
pytest -q tests
# 测试参考实现
MINISERVE_IMPL=solution pytest -q tests

# CPU 上开发
python tools/make_tiny_model.py --out /tmp/tiny --layers 4 --hidden 128
PYTHONPATH=starter python -m miniserve.server --model /tmp/tiny --device cpu --num-gpu-blocks 1024 --port 8000

# GPU 上运行真实模型（先把模型下载到本地目录）
PYTHONPATH=starter python -m miniserve.server --model /path/to/Qwen3-0.6B --port 8000

# 压测
python bench.py --base-url http://localhost:8000 --num-prompts 200 --input-len 512 --output-len 128 \
    --request-rate 8 --vocab-size 32000
```
