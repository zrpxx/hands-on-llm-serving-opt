import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认测试 starter（你自己的实现）；设置 QUANTLAB_IMPL=solution 可以测试参考实现
sys.path.insert(0, os.path.join(ROOT, os.environ.get("QUANTLAB_IMPL", "starter")))


@pytest.fixture
def layer_data():
    """一个模拟的线性层：输入特征之间相关，并且有 8 个数值特别大的离群通道，接近真实 LLM 激活的样子。"""
    g = torch.Generator().manual_seed(0)
    n_in, n_out, n = 256, 128, 1024
    mix = torch.randn(n_in, n_in, generator=g) / n_in ** 0.5
    x = torch.randn(n, n_in, generator=g) @ mix + 0.1 * torch.randn(n, n_in, generator=g)
    x[:, :8] *= 20
    w = torch.randn(n_out, n_in, generator=g) * 0.02
    return w, x


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """复用项目 1 的工具生成一个随机权重的 Qwen3 小模型，只用来检查流程能否跑通。"""
    path = str(tmp_path_factory.mktemp("tiny-qwen3"))
    tool = os.path.join(ROOT, "..", "project1", "tools", "make_tiny_model.py")
    subprocess.run([sys.executable, tool, "--out", path, "--layers", "2", "--hidden", "128"], check=True, capture_output=True)
    return path


def rel_err(a, b):
    return ((a - b) ** 2).mean().item() / (b ** 2).mean().item()
