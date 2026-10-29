"""验证 MLA 的"矩阵吸收"：decode 时不必把隐向量解压成每个头的 K、V。对应 3.1 节。

DeepSeek-V3 的一层注意力（略去 RoPE 部分）：
    c   = 隐向量，每个 token 512 维，KV Cache 只存它
    K_h = c @ W_UK[h]    每个头 128 维
    V_h = c @ W_UV[h]
    score_h = q_h · K_h，   out_h = softmax(score_h) @ V_h
吸收之后：
    score_h = (q_h @ W_UK[h]ᵀ) · c      先把查询变换到隐空间，K 就是 c 本身
    out_h   = (softmax(score_h) @ c) @ W_UV[h]
128 个头共用同一份 c，相当于一个 K、V 都是 512 维的 MQA。

    python code/stage3/mla_absorb.py
"""
import torch

torch.manual_seed(0)
H, d_c, d_h, T = 128, 512, 128, 1024          # 头数、隐向量维度、每头维度、上下文长度
W_UK = torch.randn(H, d_c, d_h) / d_c ** 0.5
W_UV = torch.randn(H, d_c, d_h) / d_c ** 0.5
c = torch.randn(T, d_c)                       # KV Cache：T 个 token 的隐向量
q = torch.randn(H, d_h)                       # decode：每个头一个查询

# 朴素做法：先解压出每个头的 K、V
K = torch.einsum("tc,hcd->htd", c, W_UK)      # [H, T, d_h]
V = torch.einsum("tc,hcd->htd", c, W_UV)
p = torch.softmax(torch.einsum("hd,htd->ht", q, K) / d_h ** 0.5, -1)
out_naive = torch.einsum("ht,htd->hd", p, V)

# 吸收：查询变换到隐空间，直接与 c 计算
q_lat = torch.einsum("hd,hcd->hc", q, W_UK)   # [H, d_c]
p2 = torch.softmax(q_lat @ c.T / d_h ** 0.5, -1)
out_absorb = torch.einsum("hc,hcd->hd", p2 @ c, W_UV)

print("两种做法的最大差：", (out_naive - out_absorb).abs().max().item())
print(f"朴素做法要临时生成的 K、V：{2 * H * T * d_h * 2 / 2**20:.0f} MiB（BF16）；吸收后只读 KV Cache 本身：{T * d_c * 2 / 2**20:.1f} MiB")
flops_attn = 2 * H * T * d_c * 2              # 分数和加权求和，都在 512 维上
print(f"吸收后注意力部分每读 1 字节 KV 的计算量：约 {flops_attn / (T * d_c * 2):.0f} FLOPs，"
      f"同样头数的 GQA（8 个 KV 头）约为 {2 * 128 * T * 128 * 2 / (2 * 8 * T * 128 * 2):.0f}")
