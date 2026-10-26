import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认测试 starter（你自己的实现）；设置 MINISERVE_IMPL=solution 可以测试参考实现
sys.path.insert(0, os.path.join(ROOT, os.environ.get("MINISERVE_IMPL", "starter")))


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("tiny-qwen3"))
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_model.py"), "--out", path],
                   check=True, capture_output=True)
    return path
