"""在自己的电脑上验证：batch 为 1 的 decode 受限于内存带宽。对应 5.5 节。

1. 用一次大数组拷贝粗略测出内存带宽；
2. 生成一个随机权重的 Qwen3 结构模型，测量逐 token decode 的速度；
3. 与"带宽 ÷ 权重字节数"的估算比较。

    python code/stage5/cpu_decode_speed.py --hidden 1024 --layers 12
小模型时每步的 Python 和框架开销占比高，实测会明显低于估算；模型越大越接近。
"""
import argparse
import time

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM


def measure_bandwidth(mb=1024):
    a = torch.empty(mb * 2**20 // 4, dtype=torch.float32).fill_(1.0)
    b = torch.empty_like(a)
    b.copy_(a)
    t0 = time.perf_counter()
    for _ in range(5):
        b.copy_(a)
    dt = (time.perf_counter() - t0) / 5
    return 2 * a.numel() * 4 / dt / 1e9          # 读一遍加写一遍


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    bw = measure_bandwidth()
    dt = getattr(torch, a.dtype)
    cfg = Qwen3Config(vocab_size=32000, hidden_size=a.hidden, intermediate_size=a.hidden * 3, num_hidden_layers=a.layers,
                      num_attention_heads=a.hidden // 64, num_key_value_heads=max(1, a.hidden // 256), head_dim=64,
                      tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).to(dt).eval()
    n = sum(p.numel() for p in model.parameters())
    w_bytes = (n - cfg.vocab_size * a.hidden) * dt.itemsize + cfg.vocab_size * a.hidden * dt.itemsize   # 嵌入只读一行，输出头读全部
    with torch.no_grad():
        cache = DynamicCache()
        out = model(torch.tensor([[1, 2, 3, 4]]), past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(3):
            out = model(nxt, past_key_values=cache, use_cache=True); nxt = out.logits[:, -1:].argmax(-1)
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            out = model(nxt, past_key_values=cache, use_cache=True); nxt = out.logits[:, -1:].argmax(-1)
        tps = a.tokens / (time.perf_counter() - t0)
    print(f"测得的内存带宽：约 {bw:.1f} GB/s（大数组拷贝，偏乐观）")
    print(f"模型：{n / 1e6:.0f}M 参数，{a.dtype}，每步要读约 {w_bytes / 1e9:.2f} GB 权重")
    print(f"带宽上限估算：{bw * 1e9 / w_bytes:.1f} token/s；实测：{tps:.1f} token/s（{tps / (bw * 1e9 / w_bytes):.0%}）")


if __name__ == "__main__":
    main()
