"""对小模型做 2:4 稀疏剪枝，比较两种准则的困惑度。对应 2.8 节。

    python tools/prune_24.py --model /tmp/tiny-lm --data /tmp/qcorpus

准则：
- magnitude：每 4 个权重中保留绝对值最大的 2 个；
- wanda：保留 |W_ij| * ||X_j|| 最大的 2 个，||X_j|| 是第 j 个输入通道在校准数据上的 L2 范数
  （Sun et al., A Simple and Effective Pruning Approach for Large Language Models, ICLR 2024）。
两种都不更新剩下的权重，所以误差比 SparseGPT 大，用来比较准则本身的好坏。
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solution"))
from quantlab.eval import perplexity  # noqa: E402


def mask_24(score: torch.Tensor) -> torch.Tensor:
    """score: [out, in]，in 是 4 的倍数。每行每 4 个中分数最高的 2 个为 True。"""
    out, n = score.shape
    s = score.reshape(out, n // 4, 4)
    keep = s.topk(2, dim=-1).indices
    m = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, keep, True)
    return m.reshape(out, n)


@torch.no_grad()
def input_norms(model, calib):
    """一次前向，统计每个线性层每个输入通道的 L2 范数。"""
    sq, hooks = {}, []
    for name, mod in model.model.layers.named_modules():
        if isinstance(mod, torch.nn.Linear):
            def hook(m, inp, out, name=name):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                sq[name] = sq.get(name, 0) + (x * x).sum(0)
            hooks.append(mod.register_forward_hook(hook))
    for x in calib:
        model(x)
    for h in hooks:
        h.remove()
    return {k: v.sqrt() for k, v in sq.items()}


@torch.no_grad()
def prune(model, criterion, norms=None):
    for name, mod in model.model.layers.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data
            score = w.abs().float()
            if criterion == "wanda":
                score = score * norms[name][None, :]
            w.mul_(mask_24(score).to(w.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--calib-samples", type=int, default=32)
    ap.add_argument("--eval-windows", type=int, default=60)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    read = lambda f: torch.tensor(tok(open(os.path.join(a.data, f), encoding="utf-8").read())["input_ids"])
    train, test = read("train.txt"), read("test.txt")
    calib = [train[i:i + a.seq_len].unsqueeze(0) for i in range(0, a.seq_len * a.calib_samples, a.seq_len)]

    for crit in ["none", "magnitude", "wanda"]:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        if crit != "none":
            prune(model, crit, input_norms(model, calib) if crit == "wanda" else None)
        print(f"{crit:10s} 困惑度 {perplexity(model, test, a.seq_len, a.eval_windows):.2f}")


if __name__ == "__main__":
    main()
