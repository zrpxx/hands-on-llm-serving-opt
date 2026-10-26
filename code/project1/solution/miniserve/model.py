"""Llama / Qwen2 / Qwen3 的前向实现，使用分页 KV Cache。

输入不是 [batch, seq] 的二维张量，而是把本步所有请求要处理的 token 拼成一维：
    input_ids: [T]      T = 本步所有请求的新 token 数之和
    positions: [T]      每个 token 在自己序列中的位置
线性层、归一化、MLP 都在 [T, hidden] 上一次算完；只有注意力需要按请求区分。
"""
import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor                 # [T]   每个新 token 写入 KV 池的位置
    # prefill（包括 chunked prefill 的一个分块）：逐请求计算
    prefill_q_starts: List[int]                # 该请求的新 token 在扁平 batch 中的起点
    prefill_q_lens: List[int]
    prefill_ctx_slots: List[torch.Tensor]      # 每个请求 [ctx_len]：从位置 0 到当前的全部 KV 槽位
    # decode（每个请求只有 1 个新 token）：合并成一个带 padding 的批
    decode_rows: Optional[torch.Tensor]        # [B]   这些 token 在扁平 batch 中的行号
    decode_slots: Optional[torch.Tensor]       # [B, max_ctx]
    decode_mask: Optional[torch.Tensor]        # [B, max_ctx]，padding 位置为 False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h, d = cfg.hidden_size, cfg.head_dim
        self.n_q, self.n_kv, self.d = cfg.num_heads, cfg.num_kv_heads, d
        self.q_proj = nn.Linear(h, self.n_q * d, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(h, self.n_kv * d, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(h, self.n_kv * d, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.n_q * d, h, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(d, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(d, cfg.rms_norm_eps)
        self.qk_norm = cfg.qk_norm
        self.scale = d ** -0.5

    def forward(self, x, cos, sin, k_cache, v_cache, meta: AttentionMetadata):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_q, self.d)
        k = self.k_proj(x).view(T, self.n_kv, self.d)
        v = self.v_proj(x).view(T, self.n_kv, self.d)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)          # [T, 1, d]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # 1) 把新 token 的 K、V 写进分页池
        k_cache[meta.slot_mapping] = k
        v_cache[meta.slot_mapping] = v

        # 2) 从池中按 block table 取出完整上下文，计算注意力
        out = torch.empty_like(q)
        for start, n, slots in zip(meta.prefill_q_starts, meta.prefill_q_lens, meta.prefill_ctx_slots):
            ctx = slots.numel()
            qs = q[start:start + n].transpose(0, 1)                 # [n_q, n, d]
            ks = k_cache[slots].transpose(0, 1)                     # [n_kv, ctx, d]
            vs = v_cache[slots].transpose(0, 1)
            # 第 i 个新 token 的位置是 ctx - n + i，只能看到位置不超过它的 key
            q_pos = torch.arange(ctx - n, ctx, device=x.device)
            mask = torch.arange(ctx, device=x.device)[None, :] <= q_pos[:, None]
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, scale=self.scale, enable_gqa=True)
            out[start:start + n] = o.transpose(0, 1)

        if meta.decode_rows is not None:
            qd = q[meta.decode_rows].unsqueeze(2)                   # [B, n_q, 1, d]
            kd = k_cache[meta.decode_slots].permute(0, 2, 1, 3)     # [B, n_kv, max_ctx, d]
            vd = v_cache[meta.decode_slots].permute(0, 2, 1, 3)
            mask = meta.decode_mask[:, None, None, :]               # [B, 1, 1, max_ctx]
            o = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask, scale=self.scale, enable_gqa=True)
            out[meta.decode_rows] = o.squeeze(2)

        return self.o_proj(out.reshape(T, self.n_q * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, k_cache, v_cache, meta):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, k_cache, v_cache, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class CausalLM(nn.Module):
    """参数命名与 Hugging Face 一致，可以直接加载 safetensors 权重。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._inv_freq = None

    def rope(self, positions: torch.Tensor, dtype):
        if self._inv_freq is None or self._inv_freq.device != positions.device:
            d = self.cfg.head_dim
            exp = torch.arange(0, d, 2, device=positions.device, dtype=torch.float32) / d
            self._inv_freq = 1.0 / (self.cfg.rope_theta ** exp)
        freqs = positions.float()[:, None] * self._inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, input_ids, positions, kv_caches, meta: AttentionMetadata):
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.rope(positions, x.dtype)
        for layer, (k_cache, v_cache) in zip(self.model.layers, kv_caches):
            x = layer(x, cos, sin, k_cache, v_cache, meta)
        return self.model.norm(x)

    def compute_logits(self, hidden):
        return self.lm_head(hidden).float()

    @classmethod
    def from_pretrained(cls, path: str, cfg: ModelConfig, dtype: torch.dtype, device: str) -> "CausalLM":
        # 先在 meta 设备上建空壳，再按目标精度分配显存，避免先建 FP32 副本
        with torch.device("meta"):
            model = cls(cfg)
        model = model.to_empty(device=device).to(dtype)
        from safetensors.torch import load_file

        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"{path} 下没有 safetensors 权重")
        loaded = set()
        for f in files:
            sd = load_file(f, device=device)
            model.load_state_dict(sd, strict=False)
            loaded.update(sd.keys())
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
            loaded.add("lm_head.weight")
        missing = set(model.state_dict().keys()) - loaded
        if missing:
            raise RuntimeError(f"权重缺失：{sorted(missing)[:5]} ...")
        return model.eval()
