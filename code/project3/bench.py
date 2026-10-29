"""在 GPU 上比较自己写的 Triton kernel 与 PyTorch 的实现。需要 NVIDIA GPU。

    python bench.py                 # 测试 starter 中你的实现
    FORGE_IMPL=solution python bench.py

对带宽瓶颈的算子（RMSNorm、SwiGLU、softmax、decode 注意力）报告有效带宽，
即"最少需要读写的字节数 / 耗时"，可以和显卡的峰值带宽比较（0.5 节）。
对 prefill 注意力报告 TFLOPS。
"""
import math
import os
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("bench.py 需要 GPU。CPU 上请用 python -m pytest tests 检查正确性。")

import triton  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FORGE_IMPL", "starter")))
from forge.flash_attn import flash_attention  # noqa: E402
from forge.paged_decode import paged_decode_attention, paged_decode_ref  # noqa: E402
from forge.rmsnorm import rmsnorm, rmsnorm_ref  # noqa: E402
from forge.softmax import online_softmax  # noqa: E402
from forge.swiglu import swiglu, swiglu_ref  # noqa: E402

dev, dt = "cuda", torch.bfloat16
bench = lambda f: triton.testing.do_bench(f, warmup=25, rep=100)   # 毫秒


def report(name, ms_ours, ms_torch, nbytes=None, flops=None):
    extra = ""
    if nbytes:
        extra = f"  有效带宽 {nbytes / ms_ours / 1e6:7.0f} GB/s（PyTorch {nbytes / ms_torch / 1e6:6.0f}）"
    if flops:
        extra = f"  {flops / ms_ours / 1e9:6.1f} TFLOPS（PyTorch {flops / ms_torch / 1e9:6.1f}）"
    print(f"{name:<34}{ms_ours * 1e3:9.1f} us  PyTorch {ms_torch * 1e3:9.1f} us{extra}")


def main():
    print(torch.cuda.get_device_name())
    N, D = 4096, 4096                                       # 一批 4096 个 token，hidden 4096
    x, r = (torch.randn(N, D, device=dev, dtype=dt) for _ in range(2))
    w = torch.randn(D, device=dev, dtype=dt)
    report("RMSNorm + 残差 [4096, 4096]", bench(lambda: rmsnorm(x, w, residual=r)),
           bench(lambda: rmsnorm_ref(x, w, residual=r)), nbytes=4 * N * D * 2)

    gu = torch.randn(N, 2 * 12288, device=dev, dtype=dt)
    report("SwiGLU [4096, 2×12288]", bench(lambda: swiglu(gu)), bench(lambda: swiglu_ref(gu)), nbytes=3 * N * 12288 * 2)

    s = torch.randn(1024, 32768, device=dev, dtype=dt)
    report("softmax [1024, 32768]", bench(lambda: online_softmax(s, block=1024)), bench(lambda: s.softmax(-1)),
           nbytes=2 * s.numel() * 2)

    H, Hkv, T, Dh = 32, 8, 4096, 128
    q = torch.randn(H, T, Dh, device=dev, dtype=dt)
    k, v = (torch.randn(Hkv, T, Dh, device=dev, dtype=dt) for _ in range(2))
    sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(
        q[None], k.repeat_interleave(H // Hkv, 0)[None], v.repeat_interleave(H // Hkv, 0)[None], is_causal=True)
    flops = 4 * H * T * T * Dh / 2                          # 因果掩码，约一半
    for bm, bn in [(64, 64), (128, 64)]:
        report(f"FlashAttention 因果 T=4096（{bm}×{bn}）",
               bench(lambda: flash_attention(q, k, v, causal=True, block_m=bm, block_n=bn)), bench(sdpa), flops=flops)

    B, bs, ctx = 64, 16, 2048
    nb = B * ctx // bs
    kc, vc = (torch.randn(nb, bs, Hkv, Dh, device=dev, dtype=dt) for _ in range(2))
    bt = torch.randperm(nb, device=dev).reshape(B, -1).to(torch.int32)
    lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    qd = torch.randn(B, H, Dh, device=dev, dtype=dt)
    report(f"分页 decode 注意力 B={B} ctx={ctx}", bench(lambda: paged_decode_attention(qd, kc, vc, bt, lens)),
           bench(lambda: paged_decode_ref(qd, kc, vc, bt, lens)), nbytes=2 * kc.numel() * 2)
    print("\n参考实现中的 decode 注意力用 Python 循环处理每个序列，只用来对照正确性，不代表 PyTorch 的最好水平。")


if __name__ == "__main__":
    main()
