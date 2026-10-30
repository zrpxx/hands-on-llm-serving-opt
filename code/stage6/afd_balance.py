"""用 Roofline 估算 decode 时 Attention 与 MoE 各自的算术强度。对应 6.4 节。

两个模型各取一层：
- Qwen3-235B-A22B：GQA，64 个 Q 头、4 个 KV 头，头维度 128；128 个专家选 8 个，每个专家 3 × 4096 × 1536 个参数
- DeepSeek-V3：MLA（吸收后 128 个头共享 576 维的压缩 KV）；256 个专家选 8 个，每个专家 3 × 7168 × 2048 个参数

KV Cache 用 BF16，专家权重用 FP8。硬件取 H800。只估算单层的主要部分，数字只用来看趋势。

    python code/stage6/afd_balance.py
"""
import argparse

BW, FLOPS = 3.35e12, 1979e12        # H800：HBM 带宽、FP8 稠密算力

MODELS = {
    "Qwen3-235B-A22B": dict(kv_bytes=2 * 4 * 128 * 2, attn_flops=4 * 64 * 128, n_exp=128, topk=8, exp_params=3 * 4096 * 1536),
    "DeepSeek-V3": dict(kv_bytes=576 * 2, attn_flops=2 * 128 * (576 + 512), n_exp=256, topk=8, exp_params=3 * 7168 * 2048),
}


def attn(m, batch, ctx):
    """每个请求读自己的 KV。返回（耗时，算术强度）。"""
    by = batch * ctx * m["kv_bytes"]
    fl = batch * ctx * m["attn_flops"]
    return max(by / BW, fl / FLOPS), fl / by


def moe(m, tokens, n_gpu):
    """tokens 个 token 经过 MoE，专家平均分到 n_gpu 张卡，负载完全均衡。返回（耗时，算术强度）。"""
    exp_per_gpu = m["n_exp"] / n_gpu
    tok_per_exp = tokens * m["topk"] / m["n_exp"]
    by = exp_per_gpu * m["exp_params"]
    fl = exp_per_gpu * tok_per_exp * 2 * m["exp_params"]
    return max(by / BW, fl / FLOPS), fl / by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096, help="平均上下文长度")
    ap.add_argument("--ep", type=int, default=32, help="专家并行的卡数")
    args = ap.parse_args()
    print(f"上下文 {args.ctx} token，EP = {args.ep}；H800 的平衡点约 {FLOPS / BW:.0f} FLOP/字节\n")
    for name, m in MODELS.items():
        print(name)
        print(f"{'每卡 batch':>10s}{'Attention 强度':>16s}{'MoE 强度':>10s}{'Attention':>12s}{'MoE':>10s}")
        for b in [16, 64, 256, 1024]:
            ta, ia = attn(m, b, args.ctx)
            tm, im = moe(m, b * args.ep, args.ep)
            print(f"{b:>10d}{ia:>16.0f}{im:>10.0f}{ta * 1e6:>10.0f} us{tm * 1e6:>8.0f} us")
        print()
    print("Attention 的算术强度与 batch 无关：每个请求读的是自己的 KV，batch 再大也不能复用。")
    print("MoE 的算术强度随 batch 线性增长：同一份专家权重被更多 token 复用。")
    print("要让 MoE 达到平衡点，每卡 batch 要几百上千，而 KV Cache 的显存限制了 batch，这是 AFD 的出发点。")


if __name__ == "__main__":
    main()
