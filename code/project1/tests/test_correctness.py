"""用 Hugging Face 的实现作为参照，验证 MiniServe 的模型与引擎输出一致。"""
import torch
from transformers import AutoModelForCausalLM

from miniserve import EngineConfig, LLMEngine, SamplingParams

PROMPTS = [
    [5, 17, 99, 210, 33],
    [44] * 3,
    list(range(30, 90)),
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    [300, 301, 302],
]


def hf_greedy(path, prompt, n):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    ids = torch.tensor([prompt])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False, eos_token_id=None,
                             pad_token_id=0, min_new_tokens=n)
    return out[0, len(prompt):].tolist()


def engine(path, **kw):
    cfg = dict(model=path, device="cpu", num_gpu_blocks=64, block_size=4, max_model_len=128,
               max_num_batched_tokens=32, max_num_seqs=8)
    cfg.update(kw)
    return LLMEngine(EngineConfig(**cfg))


def test_prefill_logits_match_hf(tiny_model):
    hf = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32).eval()
    eng = engine(tiny_model, max_num_batched_tokens=128)
    prompt = PROMPTS[2]
    with torch.no_grad():
        ref = hf(torch.tensor([prompt])).logits[0]
    seq = eng.add_request(prompt, SamplingParams(max_tokens=1, ignore_eos=True))
    sched = eng.scheduler.schedule()
    ids, pos, meta, _, _ = eng._prepare_inputs(sched)
    with torch.no_grad():
        ours = eng.model.compute_logits(eng.model(ids, pos, eng.kv_caches, meta))
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()


def _check_greedy(tiny_model, n_new=20, **kw):
    eng = engine(tiny_model, **kw)
    seqs = eng.generate(PROMPTS, SamplingParams(max_tokens=n_new, ignore_eos=True))
    for prompt, seq in zip(PROMPTS, seqs):
        assert seq.output_token_ids == hf_greedy(tiny_model, prompt, n_new), seq
    return eng


def test_continuous_chunked(tiny_model):
    eng = _check_greedy(tiny_model)
    assert eng.allocator.num_free == eng.num_blocks       # 所有块都已归还


def test_with_preemption(tiny_model):
    # 只有 40 个 4-token 块（160 token），五个请求生成到最后需要约 190 token，一定会触发抢占与重算
    eng = _check_greedy(tiny_model, num_gpu_blocks=40, max_model_len=96)
    assert eng.scheduler.total_preemptions > 0
    assert eng.allocator.num_free == eng.num_blocks


def test_no_chunked_prefill(tiny_model):
    _check_greedy(tiny_model, chunked_prefill=False, max_num_batched_tokens=128)


def test_static_batching(tiny_model):
    _check_greedy(tiny_model, policy="static", max_num_batched_tokens=128)
