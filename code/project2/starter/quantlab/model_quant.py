"""把量化方法应用到整个模型。逐个解码层进行：先用校准数据收集这一层每个线性层的输入统计量，
再量化这一层，然后处理下一层。这样后面的层看到的是前面已经量化过的输出，和 GPTQ 原论文的做法一致。
"""
import torch
from torch import nn

from .awq import awq_search
from .formats import rtn
from .gptq import gptq_quantize
from .kv import fake_quant_kv
from .smoothquant import W8A8Linear, smooth_scales


class _Stop(Exception):
    pass


def _set_module(root: nn.Module, name: str, new: nn.Module):
    parent = root
    *path, last = name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, new)


@torch.no_grad()
def _collect(model, layer, linears, calib, need):
    """跑一遍校准数据，到当前层结束为止，收集它的线性层的输入统计量。"""
    stats = {n: {"H": None, "n": 0, "rows": [], "absmax": None} for n in linears}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            st = stats[name]
            if "H" in need:
                h = x.T @ x
                st["H"] = h if st["H"] is None else st["H"] + h
                st["n"] += x.shape[0]
            if "rows" in need and sum(r.shape[0] for r in st["rows"]) < 4096:
                st["rows"].append(x)
            if "absmax" in need:
                a = x.abs().amax(dim=0)
                st["absmax"] = a if st["absmax"] is None else torch.maximum(st["absmax"], a)
        return hook

    for n, m in linears.items():
        hooks.append(m.register_forward_hook(make_hook(n)))

    def stop(mod, inp, out):
        raise _Stop()
    hooks.append(layer.register_forward_hook(stop))
    for ids in calib:
        try:
            model(ids)
        except _Stop:
            pass
    for h in hooks:
        h.remove()
    for st in stats.values():
        if st["H"] is not None:
            st["H"] = 2.0 / st["n"] * st["H"]
        if st["rows"]:
            st["rows"] = torch.cat(st["rows"])[:4096]
    return stats


@torch.no_grad()
def quantize_model(model, method: str, calib=None, bits: int = 4, group_size: int = 128,
                   alpha: float = 0.5, act: str = "token", act_order: bool = False):
    """method ∈ {rtn, gptq, awq, w8a8, smoothquant, fp8}。返回原地修改后的 model。"""
    need = {"gptq": {"H"}, "awq": {"rows"}, "smoothquant": {"absmax"}}.get(method, set())
    for layer in model.model.layers:
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        stats = _collect(model, layer, linears, calib, need) if need else None
        for name, lin in linears.items():
            W = lin.weight.data
            if method == "rtn":
                lin.weight.data = rtn(W, bits, group_size)
            elif method == "gptq":
                lin.weight.data = gptq_quantize(W, stats[name]["H"], bits, group_size, act_order=act_order)
            elif method == "awq":
                lin.weight.data = awq_search(W, stats[name]["rows"], bits, group_size)[0]
            elif method == "w8a8":
                _set_module(layer, name, W8A8Linear(lin, None, act, "int8"))
            elif method == "smoothquant":
                s = smooth_scales(stats[name]["absmax"], W, alpha)
                _set_module(layer, name, W8A8Linear(lin, s, act, "int8"))
            elif method == "fp8":
                _set_module(layer, name, W8A8Linear(lin, None, act, "fp8"))
            else:
                raise ValueError(method)
    return model


def add_kv_quant(model, bits: int = 4, k_axis: str = "channel", v_axis: str = "token", fmt: str = "int"):
    """在每层注意力上挂钩子，把 K、V 伪量化。返回钩子列表，调用 h.remove() 可以撤销。

    说明：钩子挂在 k_norm（Qwen3）或 k_proj 的输出上，此时还没有做 RoPE。真实的 KV Cache 存的是
    RoPE 之后的 K。RoPE 是逐对旋转，会把一个通道的离群值分摊到它配对的通道上，所以这里按通道量化 K
    的效果会比真实情况略好一些，用于比较不同方案的相对好坏是足够的。
    """
    hooks = []
    for layer in model.model.layers:
        attn = layer.self_attn
        n_kv = model.config.num_key_value_heads
        d = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)

        def q(t, axis):
            shape = t.shape
            x = t.reshape(-1, n_kv, d)               # [tokens, heads, head_dim]
            return fake_quant_kv(x, bits, axis, fmt).reshape(shape)

        k_mod = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
        hooks.append(k_mod.register_forward_hook(lambda m, i, o: q(o, k_axis)))
        hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o: q(o, v_axis)))
    return hooks


def weight_bits(bits: int, group_size: int, symmetric: bool = True) -> float:
    """每个权重平均占多少比特：数值本身，加上每组一个 FP16 的 scale（非对称时再加一个 zero）。"""
    extra = 16 if symmetric else 16 + bits
    return bits + (extra / group_size if group_size > 0 else 0)
