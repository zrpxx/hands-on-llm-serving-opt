"""在 CPU 上演示 Multi-LoRA 的批量计算：同一个 batch 中的请求各自使用不同的适配器。对应 4.7 节。

    y = x W^T + scale · (x A_i^T) B_i^T          i 是这个请求使用的适配器

基座权重 W 所有请求共用，一次矩阵乘法即可；LoRA 部分每个请求不同。几种算法：
1. 逐个请求计算：每个请求一次小矩阵乘法，调用次数随 batch 增长；
2. 按适配器分组：同一个适配器的请求放在一起算（Punica 的 SGMV kernel 把这些分组合并在一次调用里）；
3. 按请求收集各自的 A、B，用批量矩阵乘法一次算完（类似 Punica 的 BGMV kernel）；
4. 合并权重：把 BA 加到 W 上，只适用于整个 batch 用同一个适配器的情况。

    python code/stage4/lora_batch.py
"""
import time

import torch

torch.manual_seed(0)
B, d_in, d_out, r, n_adapters = 64, 1024, 1024, 16, 8
W = torch.randn(d_out, d_in) / d_in ** 0.5
A = torch.randn(n_adapters, r, d_in) / d_in ** 0.5
Bm = torch.randn(n_adapters, d_out, r) * 0.01
x = torch.randn(B, d_in)
idx = torch.randint(0, n_adapters, (B,))                  # 每个请求用哪个适配器
scale = 2.0

base = x @ W.T

t0 = time.perf_counter()
y1 = base.clone()
for i in range(B):                                        # 1. 逐个请求
    y1[i] += scale * (x[i] @ A[idx[i]].T) @ Bm[idx[i]].T
t1 = time.perf_counter()

y2 = base.clone()
for a in range(n_adapters):                               # 2. 按适配器分组
    sel = (idx == a).nonzero(as_tuple=True)[0]
    if sel.numel():
        y2[sel] += scale * (x[sel] @ A[a].T) @ Bm[a].T
t2 = time.perf_counter()

y3 = torch.bmm(torch.bmm(x[:, None, :], A[idx].transpose(1, 2)), Bm[idx].transpose(1, 2))[:, 0] * scale + base
t3 = time.perf_counter()                                  # 3. 批量矩阵乘法

print(f"batch {B}，{n_adapters} 个适配器，秩 {r}")
print(f"逐个请求与按适配器分组的最大差：{(y1 - y2).abs().max().item():.2e}；与批量矩阵乘法：{(y1 - y3).abs().max().item():.2e}")
print(f"矩阵乘法调用次数：逐个请求 {2 * B} 次；按适配器分组 {2 * n_adapters} 次；批量矩阵乘法 2 次")
print(f"CPU 耗时（仅供参考）：逐个 {(t1 - t0) * 1e3:.1f} ms，分组 {(t2 - t1) * 1e3:.1f} ms，批量 {(t3 - t2) * 1e3:.1f} ms")
lora_flops, base_flops = 2 * B * r * (d_in + d_out), 2 * B * d_in * d_out
print(f"LoRA 部分的计算量是基座矩阵乘法的 {lora_flops / base_flops:.1%}")

# 4. 合并权重：只有一个适配器时，W' = W + scale·BA，推理时没有任何额外开销
W_merged = W + scale * Bm[0] @ A[0]
y_single = base + scale * (x @ A[0].T) @ Bm[0].T
print(f"合并权重与不合并的最大差（全部使用适配器 0）：{(x @ W_merged.T - y_single).abs().max().item():.2e}")
