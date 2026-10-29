"""量化一个模型并计算困惑度。

    python run.py --model /tmp/tiny-lm --data data --method gptq --bits 3 --group-size 128
    python run.py --model /tmp/tiny-lm --data data --method fp8
    python run.py --model /tmp/tiny-lm --data data --method none --kv-bits 4 --k-axis channel

方法：none（不量化，作为基准）、rtn、gptq、awq（仅权重）；w8a8、smoothquant、fp8（权重和激活）。
真实模型（如 Qwen3-0.6B）也可以直接使用，CPU 上会比较慢，建议加 --device cuda。
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("QUANTLAB_IMPL", "starter")))
from quantlab.eval import perplexity                       # noqa: E402
from quantlab.model_quant import add_kv_quant, quantize_model, weight_bits   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data", help="make_corpus.py 生成的目录，含 train.txt 和 test.txt")
    ap.add_argument("--method", default="none", choices=["none", "rtn", "gptq", "awq", "w8a8", "smoothquant", "fp8"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant 的迁移强度")
    ap.add_argument("--act", default="token", choices=["token", "tensor"], help="激活量化的粒度")
    ap.add_argument("--kv-bits", type=int, default=0)
    ap.add_argument("--k-axis", default="channel", choices=["token", "channel"])
    ap.add_argument("--calib-samples", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--eval-windows", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    read = lambda name: torch.tensor(tok(open(os.path.join(a.data, name), encoding="utf-8").read())["input_ids"])
    train, test = read("train.txt").to(a.device), read("test.txt").to(a.device)
    calib = [train[i:i + a.seq_len].unsqueeze(0) for i in range(0, a.seq_len * a.calib_samples, a.seq_len)]

    t0 = time.time()
    if a.method != "none":
        quantize_model(model, a.method, calib, a.bits, a.group_size, a.alpha, a.act, a.act_order)
    if a.kv_bits:
        add_kv_quant(model, a.kv_bits, a.k_axis, "token")
    cost = time.time() - t0
    ppl = perplexity(model, test, a.seq_len, a.eval_windows)
    desc = a.method
    if a.method in ("rtn", "gptq", "awq"):
        desc += f" W{a.bits} g{a.group_size}（平均 {weight_bits(a.bits, a.group_size):.2f} 比特/权重）"
    if a.kv_bits:
        desc += f" KV{a.kv_bits}（K 按 {a.k_axis}）"
    print(f"{desc}  困惑度 {ppl:.2f}  量化用时 {cost:.1f}s")


if __name__ == "__main__":
    main()
