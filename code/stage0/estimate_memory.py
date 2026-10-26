"""估算 Decoder-only 稠密模型的权重显存、每 token KV Cache 大小与可容纳的并发。

只适用于 Llama / Qwen 一类的稠密结构（GQA + SwiGLU MLP）。MoE、MLA、滑动窗口
等结构需要另行处理。

用法：
    python estimate_memory.py configs/qwen3-8b.json --gpu-mem 80 --max-len 4096
    python estimate_memory.py path/to/config.json --weight-bytes 1 --kv-bytes 1   # FP8 权重与 KV
"""
import argparse
import json

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def count_params(c):
    h, inter, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    n_q, n_kv = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
    d = c.get("head_dim", h // n_q)
    attn = h * n_q * d + 2 * h * n_kv * d + n_q * d * h      # q, k, v, o 投影
    mlp = 3 * h * inter                                        # gate, up, down
    norms = 2 * h
    embed = V * h
    lm_head = 0 if c.get("tie_word_embeddings", False) else V * h
    return L * (attn + mlp + norms) + embed + lm_head + h


def kv_bytes_per_token(c, kv_bytes):
    n_q = c["num_attention_heads"]
    n_kv = c.get("num_key_value_heads", n_q)
    d = c.get("head_dim", c["hidden_size"] // n_q)
    return 2 * c["num_hidden_layers"] * n_kv * d * kv_bytes   # 2 = K 和 V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpu-mem", type=float, default=80, help="单卡显存，GB")
    ap.add_argument("--util", type=float, default=0.9, help="对应 vLLM 的 gpu_memory_utilization")
    ap.add_argument("--overhead", type=float, default=2.0, help="激活、CUDA graph 等预留，GB（经验值）")
    ap.add_argument("--max-len", type=int, default=4096, help="每个请求的最大 token 数（输入 + 输出）")
    ap.add_argument("--tp", type=int, default=1, help="张量并行卡数")
    ap.add_argument("--weight-bytes", type=float, default=None)
    ap.add_argument("--kv-bytes", type=float, default=None)
    a = ap.parse_args()

    c = json.load(open(a.config))
    default_bytes = DTYPE_BYTES.get(c.get("torch_dtype", "bfloat16"), 2)
    wb = a.weight_bytes or default_bytes
    kb = a.kv_bytes or default_bytes

    P = count_params(c)
    weights_gb = P * wb / 1e9 / a.tp
    kv_tok = kv_bytes_per_token(c, kb) / a.tp           # TP 按 KV 头切分（假设 n_kv 能被 tp 整除）
    budget_gb = a.gpu_mem * a.util - weights_gb - a.overhead
    tokens = budget_gb * 1e9 / kv_tok if budget_gb > 0 else 0

    print(f"参数量            {P / 1e9:.2f} B")
    print(f"权重（每卡）      {weights_gb:.1f} GB   ({wb} B/参数, tp={a.tp})")
    print(f"KV / token（每卡）{kv_tok / 1024:.0f} KiB  ({kb} B/元素)")
    print(f"KV 预算（每卡）   {budget_gb:.1f} GB = {a.gpu_mem}×{a.util} - 权重 - {a.overhead} 预留")
    print(f"可缓存 token 数   {tokens:,.0f}")
    print(f"并发上限          {tokens / a.max_len:.1f} 个请求（每个 {a.max_len} token 用满时）")


if __name__ == "__main__":
    main()
