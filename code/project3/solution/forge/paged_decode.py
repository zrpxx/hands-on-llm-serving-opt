"""第 5 步：分页 KV Cache 上的 decode 注意力。对应 3.2 节，与项目 1 的 KV 布局相同。

decode 时每个序列只有一个查询 token，但要读完它全部的 KV。KV 按块存放在一个大的缓存里：
k_cache、v_cache 的形状是 [num_blocks, block_size, H_kv, D]，block_tables[b, j] 是序列 b 的
第 j 个逻辑块对应的物理块号。每个 program 负责一个序列的一个查询头，按块遍历它的 KV。
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, lens_ptr, o_ptr, sm_scale, max_blocks,
                         H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // (H // HKV)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, BLOCK_SIZE)
    q = tl.load(q_ptr + (b * H + h) * D + offs_d).to(tl.float32)
    seq_len = tl.load(lens_ptr + b)

    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    n_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for j in range(0, n_blocks):
        phys = tl.load(bt_ptr + b * max_blocks + j).to(tl.int64)
        tok = j * BLOCK_SIZE + offs_t
        # [BLOCK_SIZE, D]：物理块 phys 中，第 kvh 个头的所有 token
        base = (phys * BLOCK_SIZE + offs_t[:, None]) * HKV * D + kvh * D + offs_d[None, :]
        k = tl.load(k_ptr + base, mask=tok[:, None] < seq_len, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base, mask=tok[:, None] < seq_len, other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * sm_scale
        s = tl.where(tok < seq_len, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    tl.store(o_ptr + (b * H + h) * D + offs_d, acc / l)


def paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, sm_scale: float = None):
    """q: [B, H, D]；k_cache、v_cache: [num_blocks, block_size, H_kv, D]；
    block_tables: [B, max_blocks] int32；seq_lens: [B] int32。返回 [B, H, D]。"""
    B, H, D = q.shape
    _, block_size, Hkv, _ = k_cache.shape
    o = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
    sm_scale = sm_scale or 1.0 / math.sqrt(D)
    _paged_decode_kernel[(B, H)](q.contiguous(), k_cache, v_cache, block_tables.contiguous(), seq_lens, o,
                                 sm_scale, block_tables.shape[1], H=H, HKV=Hkv, D=D, BLOCK_SIZE=block_size)
    return o.to(q.dtype)


def paged_decode_ref(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=None):
    B, H, D = q.shape
    _, bs, Hkv, _ = k_cache.shape
    out = []
    for b in range(B):
        n = int(seq_lens[b])
        blocks = block_tables[b, : (n + bs - 1) // bs].long()
        k = k_cache[blocks].reshape(-1, Hkv, D)[:n].transpose(0, 1)   # [Hkv, n, D]
        v = v_cache[blocks].reshape(-1, Hkv, D)[:n].transpose(0, 1)
        k = k.repeat_interleave(H // Hkv, 0).float()
        v = v.repeat_interleave(H // Hkv, 0).float()
        s = (q[b].float()[:, None, :] @ k.transpose(1, 2)).squeeze(1) * (sm_scale or 1.0 / math.sqrt(D))
        out.append((s.softmax(-1)[:, None, :] @ v).squeeze(1))
    return torch.stack(out).to(q.dtype)
