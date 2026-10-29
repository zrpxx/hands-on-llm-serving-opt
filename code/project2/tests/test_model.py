"""端到端：在随机权重的小模型上跑通每种方法。全部步骤完成后运行。"""
import copy

import pytest
import torch
from transformers import AutoModelForCausalLM

from conftest import rel_err


@pytest.mark.parametrize("method", ["rtn", "gptq", "awq", "w8a8", "smoothquant", "fp8"])
def test_quantize_model_runs(tiny_model, method):
    from quantlab.model_quant import quantize_model
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(tiny_model).eval()
    ids = torch.randint(5, 400, (1, 64))
    calib = [torch.randint(5, 400, (1, 64)) for _ in range(4)]
    with torch.no_grad():
        ref = base(ids).logits
        m = quantize_model(copy.deepcopy(base), method, calib, bits=8, group_size=64)
        out = m(ids).logits
    assert rel_err(out, ref) < 0.01, method


def test_kv_quant_hooks(tiny_model):
    from quantlab.model_quant import add_kv_quant
    base = AutoModelForCausalLM.from_pretrained(tiny_model).eval()
    ids = torch.randint(5, 400, (1, 64))
    with torch.no_grad():
        ref = base(ids).logits
        hooks = add_kv_quant(base, bits=8)
        out8 = base(ids).logits
        for h in hooks:
            h.remove()
        hooks = add_kv_quant(base, bits=2)
        out2 = base(ids).logits
    assert rel_err(out8, ref) < rel_err(out2, ref)
