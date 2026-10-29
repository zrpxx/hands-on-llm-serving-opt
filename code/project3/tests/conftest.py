import os
import sys

# 没有 GPU 时用 Triton 的解释器在 CPU 上运行 kernel。必须在 import triton 之前设置。
try:
    import torch
    _cuda = torch.cuda.is_available()
except ImportError:
    _cuda = False
if not _cuda:
    os.environ.setdefault("TRITON_INTERPRET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, os.environ.get("FORGE_IMPL", "starter")))

import pytest  # noqa: E402

DEVICE = "cuda" if _cuda else "cpu"


@pytest.fixture
def device():
    return DEVICE


def close(a, b, tol=2e-3):
    # GPU 上 fp32 的 tl.dot 默认使用 TF32，误差比 CPU 的解释器大一些
    if _cuda:
        tol *= 5
    return (a.float() - b.float()).abs().max().item() < tol
