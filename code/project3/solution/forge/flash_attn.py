"""第 4 步：FlashAttention 的前向（prefill）。对应 3.2 节。

每个 program 负责一个头的 BLOCK_M 个查询。它按 BLOCK_N 个 key 一块遍历 K、V，
在片上维护每个查询的 m（最大值）、l（指数和）和 acc（未归一化的输出），
从不把 [M, N] 的注意力矩阵写回显存。

支持：
- 因果掩码，且查询可以只是序列的最后 M 个位置（chunked prefill 时 M < N），
  第 i 个查询的绝对位置是 N - M + i；
- GQA：第 h 个查询头使用第 h // (H / H_kv) 个 KV 头。
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, M, N, sm_scale,
                      GROUP: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // GROUP
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + h * M * D + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < M, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    q_pos = N - M + offs_m                        # 查询的绝对位置

    end = N
    if CAUSAL:                                    # 这一块查询最远只能看到 q_pos 的最大值
        end = tl.minimum(N, N - M + (pid_m + 1) * BLOCK_M)
    for start in range(0, end, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N
        k = tl.load(k_ptr + kvh * N * D + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + kvh * N * D + offs_n[:, None] * D + offs_d[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale                     # [BLOCK_M, BLOCK_N]
        valid = offs_n[None, :] < N
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_pos[:, None])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)                              # 旧结果的修正系数
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(o_ptr + h * M * D + offs_m[:, None] * D + offs_d[None, :], out, mask=offs_m[:, None] < M)


def flash_attention(q, k, v, causal: bool = True, sm_scale: float = None, block_m: int = 16, block_n: int = 16):
    """q: [H, M, D]，k、v: [H_kv, N, D]，M <= N。返回 [H, M, D]。"""
    H, M, D = q.shape
    Hkv, N, _ = k.shape
    assert H % Hkv == 0 and M <= N and D in (16, 32, 64, 128)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    sm_scale = sm_scale or 1.0 / math.sqrt(D)
    grid = (triton.cdiv(M, block_m), H)
    _flash_fwd_kernel[grid](q, k, v, o, M, N, sm_scale, GROUP=H // Hkv, D=D, CAUSAL=causal,
                            BLOCK_M=block_m, BLOCK_N=block_n)
    return o


def attention_ref(q, k, v, causal=True, sm_scale=None):
    H, M, D = q.shape
    Hkv, N, _ = k.shape
    k = k.repeat_interleave(H // Hkv, 0).float()
    v = v.repeat_interleave(H // Hkv, 0).float()
    s = (q.float() @ k.transpose(1, 2)) * (sm_scale or 1.0 / math.sqrt(D))
    if causal:
        qpos = torch.arange(N - M, N)[:, None]
        s = s.masked_fill(torch.arange(N)[None, :] > qpos, float("-inf"))
    return (s.softmax(-1) @ v).to(q.dtype)
