"""分步测试。每一步完成后运行：python -m pytest tests/test_steps.py -k step01
没有 GPU 时，kernel 在 Triton 解释器中运行，比较慢，所以这里的张量都很小。"""
import torch

from conftest import close


# ------------------------------------------------------------------ 第 1 步：RMSNorm
def test_step01_rmsnorm(device):
    from forge.rmsnorm import rmsnorm, rmsnorm_ref
    torch.manual_seed(0)
    x = torch.randn(5, 96, device=device)          # 96 不是 2 的幂，检查 mask
    w = torch.randn(96, device=device)
    y, r = rmsnorm(x, w)
    assert r is None and close(y, rmsnorm_ref(x, w)[0])


def test_step01_rmsnorm_with_residual(device):
    from forge.rmsnorm import rmsnorm, rmsnorm_ref
    torch.manual_seed(1)
    x, res = torch.randn(3, 64, device=device), torch.randn(3, 64, device=device)
    w = torch.rand(64, device=device)
    y, r = rmsnorm(x, w, residual=res)
    y_ref, r_ref = rmsnorm_ref(x, w, residual=res)
    assert close(y, y_ref) and close(r, r_ref)


# ------------------------------------------------------------------ 第 2 步：SwiGLU
def test_step02_swiglu(device):
    from forge.swiglu import swiglu, swiglu_ref
    torch.manual_seed(2)
    x = torch.randn(4, 2 * 200, device=device)      # 200 列，block=128 时需要两块，第二块不满
    assert close(swiglu(x, block=128), swiglu_ref(x))


# ------------------------------------------------------------------ 第 3 步：在线 softmax
def test_step03_online_softmax_long_rows(device):
    from forge.softmax import online_softmax
    torch.manual_seed(3)
    x = torch.randn(3, 1000, device=device) * 5
    assert close(online_softmax(x, block=128), x.softmax(-1), 1e-5)


def test_step03_online_softmax_is_stable(device):
    from forge.softmax import online_softmax
    x = torch.tensor([[1000.0, 1001.0, 999.0] + [0.0] * 200], device=device)
    y = online_softmax(x, block=64)
    assert torch.isfinite(y).all() and close(y, x.softmax(-1), 1e-5)


# ------------------------------------------------------------------ 第 4 步：FlashAttention
def test_step04_flash_noncausal(device):
    from forge.flash_attn import flash_attention, attention_ref
    torch.manual_seed(4)
    q, k, v = (torch.randn(2, 40, 16, device=device) for _ in range(3))
    assert close(flash_attention(q, k, v, causal=False), attention_ref(q, k, v, causal=False))


def test_step04_flash_causal_gqa_chunk(device):
    from forge.flash_attn import flash_attention, attention_ref
    torch.manual_seed(5)
    q = torch.randn(4, 20, 32, device=device)       # 4 个查询头
    k = torch.randn(2, 50, 32, device=device)       # 2 个 KV 头；查询是 50 个位置中的最后 20 个
    v = torch.randn(2, 50, 32, device=device)
    assert close(flash_attention(q, k, v, causal=True), attention_ref(q, k, v, causal=True))


def test_step04_flash_causal_full(device):
    from forge.flash_attn import flash_attention, attention_ref
    torch.manual_seed(6)
    q, k, v = (torch.randn(1, 37, 16, device=device) for _ in range(3))
    assert close(flash_attention(q, k, v, causal=True), attention_ref(q, k, v, causal=True))


# ------------------------------------------------------------------ 第 5 步：分页 decode 注意力
def test_step05_paged_decode(device):
    from forge.paged_decode import paged_decode_attention, paged_decode_ref
    torch.manual_seed(7)
    B, H, Hkv, D, bs, nblocks = 3, 4, 2, 16, 16, 12
    k_cache = torch.randn(nblocks, bs, Hkv, D, device=device)
    v_cache = torch.randn(nblocks, bs, Hkv, D, device=device)
    perm = torch.randperm(nblocks)                  # 物理块的顺序是打乱的
    block_tables = perm[:12].reshape(3, 4).to(torch.int32).to(device)
    seq_lens = torch.tensor([1, 33, 64], dtype=torch.int32, device=device)
    q = torch.randn(B, H, D, device=device)
    out = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
    assert close(out, paged_decode_ref(q, k_cache, v_cache, block_tables, seq_lens))


# ------------------------------------------------------------------ 第 6 步：投机解码
def test_step06_rejection_sampling_preserves_distribution():
    from forge.spec import rejection_sample
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])        # 目标模型
    q = torch.tensor([0.1, 0.2, 0.3, 0.4])          # 一个很差的草稿模型
    counts = torch.zeros(4)
    n = 6000
    for _ in range(n):
        t = torch.multinomial(q, 1, generator=g)
        first = rejection_sample(torch.stack([p, p]), q[None], t, g)[0]
        counts[first] += 1
    assert (counts / n - p).abs().max() < 0.025


def test_step06_greedy_verify():
    from forge.spec import greedy_verify
    logits = torch.zeros(4, 10)
    logits[0, 3] = logits[1, 5] = logits[2, 7] = logits[3, 1] = 1
    assert greedy_verify(logits, torch.tensor([3, 5, 7])) == [3, 5, 7, 1]   # 全部接受，外加一个
    assert greedy_verify(logits, torch.tensor([3, 4, 7])) == [3, 5]         # 第二个被拒绝


def test_step06_ngram_propose():
    from forge.spec import ngram_propose
    toks = [1, 2, 3, 4, 5, 9, 9, 2, 3]
    assert ngram_propose(toks, k=3) == [4, 5, 9]
    assert ngram_propose([1, 2, 3], k=2) == []


def test_step06_speculative_greedy_matches_target():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from forge.spec import greedy_generate, speculative_generate
    torch.manual_seed(0)
    cfg = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
               head_dim=16, max_position_embeddings=128)
    target = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, **cfg)).eval()
    draft = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=1, **cfg)).eval()
    prompt = [1, 5, 9, 13]
    ref = greedy_generate(target, prompt, 12)
    out, stats = speculative_generate(target, prompt, 12, k=3, draft=draft)
    assert out == ref
    out2, _ = speculative_generate(target, prompt, 12, k=3, method="ngram")
    assert out2 == ref
    assert stats.steps <= 12
