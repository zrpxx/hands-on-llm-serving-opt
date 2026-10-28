"""分步测试：每完成一个 TODO，就运行对应的一组测试。

    pytest tests/test_steps.py -k step01     # 只测第 1 步
    pytest tests/test_steps.py -x            # 从头测到第一个失败为止

步骤编号与「项目 1 分步指南」页面一致。后面的步骤会用到前面步骤的实现，所以请按顺序完成。
"""
import asyncio

import pytest
import torch

from miniserve import EngineConfig, LLMEngine, SamplingParams
from miniserve.kv_cache import BlockAllocator
from miniserve.scheduler import Scheduler, SchedulerOutput
from miniserve.sequence import Sequence, SeqStatus


def make_seq(n_prompt, max_tokens=4, first_id=1):
    return Sequence(list(range(first_id, first_id + n_prompt)),
                    SamplingParams(max_tokens=max_tokens, ignore_eos=True))


def make_sched(num_blocks=100, block_size=4, max_seqs=8, budget=16, **kw):
    return Scheduler(BlockAllocator(num_blocks), block_size, max_seqs, budget, **kw)


def pairs(out):
    return [(s.seq_id, n) for s, n in out]


# --------------------------------------------------------------------------- 第 1 步
def test_step01_allocator_allocate_and_free():
    a = BlockAllocator(3)
    got = [a.allocate(), a.allocate()]
    assert all(isinstance(b, int) for b in got), "allocate() 应返回一个 int 块号"
    assert len(set(got)) == 2, "两次 allocate() 不应返回同一个块"
    assert a.num_free == 1, f"分出 2 块后应剩 1 块空闲，实际 {a.num_free}"
    a.free(got)
    assert a.num_free == 3, "free() 之后块应回到空闲列表"


def test_step01_allocator_raises_when_empty():
    a = BlockAllocator(1)
    a.allocate()
    with pytest.raises(RuntimeError):
        a.allocate()


# --------------------------------------------------------------------------- 第 2 步
@pytest.mark.parametrize("computed, have_blocks, num_new, expected", [
    (0, 0, 9, 3),    # 新请求 9 个 token：需要 ceil(9/4)=3 块
    (0, 0, 4, 1),    # 正好一块
    (4, 1, 1, 1),    # 第 5 个 token 落在第 2 块：再要 1 块
    (5, 2, 1, 0),    # 第 6 个 token 仍在第 2 块：不需要新块
    (8, 2, 3, 1),    # 第 9~11 个 token 在第 3 块
])
def test_step02_blocks_needed(computed, have_blocks, num_new, expected):
    sch = make_sched(block_size=4)
    s = make_seq(20)
    s.num_computed_tokens = computed
    s.block_table = list(range(100, 100 + have_blocks))
    got = sch._blocks_needed(s, num_new)
    assert got == expected, (f"已算 {computed} 个、已有 {have_blocks} 块、再算 {num_new} 个："
                             f"应再要 {expected} 块，实际返回 {got}")


def test_step02_allocate_appends_blocks():
    sch = make_sched(num_blocks=10, block_size=4)
    s = make_seq(9)
    sch._allocate(s, 9)
    assert len(s.block_table) == 3, f"9 个 token 应分到 3 块，实际 {s.block_table}"
    assert sch.allocator.num_free == 7
    s.num_computed_tokens = 9
    sch._allocate(s, 1)                       # 第 10 个 token 仍在第 3 块
    assert len(s.block_table) == 3, "第 10 个 token 不应触发新分配"


# --------------------------------------------------------------------------- 第 3 步
def test_step03_admit_partial():
    sch = make_sched(budget=10)
    a, b = make_seq(6), make_seq(6)
    sch.add(a); sch.add(b)
    got = sch._admit(10, allow_partial=True)
    assert pairs(got) == [(a.seq_id, 6), (b.seq_id, 4)], \
        f"预算 10、两个 6-token prompt：应为 a 处理 6 个、b 处理 4 个，实际 {pairs(got)}"
    assert sch.running == [a, b] and not sch.waiting, "被接纳的请求应从 waiting 移到 running"
    assert a.status == SeqStatus.RUNNING and b.status == SeqStatus.RUNNING
    assert len(a.block_table) == 2 and len(b.block_table) == 1, "接纳时要为本步的 token 分配块"


def test_step03_admit_no_partial():
    sch = make_sched(budget=10)
    a, b = make_seq(6), make_seq(6)
    sch.add(a); sch.add(b)
    got = sch._admit(10, allow_partial=False)
    assert pairs(got) == [(a.seq_id, 6)], "不允许切块时，b 放不进剩余预算 4，应留在 waiting"
    assert list(sch.waiting) == [b]


def test_step03_admit_respects_max_seqs_and_blocks():
    sch = make_sched(max_seqs=2, budget=100)
    for _ in range(3):
        sch.add(make_seq(3))
    assert len(sch._admit(100, allow_partial=True)) == 2, "running 已有 max_num_seqs 个时应停止接纳"

    sch = make_sched(num_blocks=2, block_size=4, budget=100)
    a, b = make_seq(8), make_seq(4)
    sch.add(a); sch.add(b)
    got = sch._admit(100, allow_partial=True)
    assert pairs(got) == [(a.seq_id, 8)], "块不够时应停止接纳，而不是抛异常"
    assert list(sch.waiting) == [b]


# --------------------------------------------------------------------------- 第 4 步
def test_step04_schedule_running_basic():
    sch = make_sched(budget=100)
    a, b = make_seq(6), make_seq(10)
    sch.add(a); sch.add(b)
    sch._admit(8, allow_partial=True)              # a 处理 6 个，b 处理 2 个
    a.num_computed_tokens, b.num_computed_tokens = 6, 2
    a.token_ids.append(0)                          # a 生成了 1 个 token，进入 decode
    out = sch._schedule_running(100)
    assert isinstance(out, tuple) and len(out) == 3, "_schedule_running 应返回 (scheduled, 剩余预算, 抢占数)"
    scheduled, budget, preempted = out
    assert pairs(scheduled) == [(a.seq_id, 1), (b.seq_id, 8)], \
        f"a 在 decode 应得 1 个；b 剩余 8 个 prompt token，实际 {pairs(scheduled)}"
    assert budget == 91 and preempted == 0


def test_step04_schedule_running_budget_limit():
    sch = make_sched(budget=100)
    a, b = make_seq(20), make_seq(20)
    sch.add(a); sch.add(b)
    sch._admit(100, allow_partial=True)
    a.num_computed_tokens = b.num_computed_tokens = 4    # 假设各算了 4 个
    scheduled, budget, _ = sch._schedule_running(10)
    assert pairs(scheduled) == [(a.seq_id, 10)], "预算用完后应停止，b 这一步不参与"
    assert budget == 0


# --------------------------------------------------------------------------- 第 5 步
def test_step05_continuous_running_first_then_waiting():
    sch = make_sched(budget=10)
    a = make_seq(4, max_tokens=10)
    sch.add(a)
    out = sch.schedule()
    assert isinstance(out, SchedulerOutput)
    assert pairs(out.scheduled) == [(a.seq_id, 4)]
    a.num_computed_tokens = 4
    a.token_ids.append(0)
    b = make_seq(20)
    sch.add(b)
    out = sch.schedule()
    assert pairs(out.scheduled) == [(a.seq_id, 1), (b.seq_id, 9)], \
        f"应先给 running 的 a 1 个，再用剩余预算 9 接纳 b，实际 {pairs(out.scheduled)}"


# --------------------------------------------------------------------------- 第 6 步
def test_step06_preempt_resets_state():
    sch = make_sched(num_blocks=10, budget=100)
    a, b = make_seq(8), make_seq(8)
    sch.add(a); sch.add(b)
    sch._admit(100, allow_partial=True)
    b.num_computed_tokens = 8
    sch._preempt(b)
    assert b.status == SeqStatus.WAITING
    assert b.num_computed_tokens == 0, "抢占后必须把 num_computed_tokens 清零，否则恢复时会跳过重算"
    assert b.block_table == [], "抢占后 block_table 应清空"
    assert sch.waiting[0] is b, "被抢占的请求应放回等待队列的头部"
    assert b not in sch.running
    assert sch.allocator.num_free == 8, "b 的 2 个块应归还"
    assert sch.total_preemptions == 1


def test_step06_running_preempts_latest():
    # 6 块 = 24 token。两个 8-token 请求生成到一定长度后放不下，后到的 b 应被抢占
    sch = make_sched(num_blocks=6, block_size=4, budget=100)
    a, b = make_seq(8, max_tokens=12), make_seq(8, max_tokens=12)
    sch.add(a); sch.add(b)
    for _ in range(12):
        out = sch.schedule()
        if out.num_preempted:
            break
        for s, n in out.scheduled:
            s.num_computed_tokens += n
            if s.num_computed_tokens == s.num_tokens:
                s.token_ids.append(0)
    assert out.num_preempted >= 1, "块不够时应发生抢占"
    assert b.status == SeqStatus.WAITING and a.status == SeqStatus.RUNNING, "应抢占后到的 b，而不是 a"
    assert pairs(out.scheduled) == [(a.seq_id, 1)], "抢占之后 a 应能继续 decode，且这一步不接纳新请求"


# --------------------------------------------------------------------------- 第 7 步
@pytest.fixture(scope="module")
def engine(tiny_model):
    return LLMEngine(EngineConfig(model=tiny_model, device="cpu", num_gpu_blocks=64, block_size=4,
                                  max_model_len=128, max_num_batched_tokens=32, max_num_seqs=8))


def test_step07_prepare_inputs_worked_example(engine):
    """与分步指南第 7 步中的手算例子完全一致。block_size = 4。"""
    p = SamplingParams(max_tokens=8, ignore_eos=True)
    A = Sequence([11, 12, 13, 14, 15, 16], p)        # prompt 6 个，已算 4 个（第 1 块），本步算 2 个
    A.num_computed_tokens, A.block_table = 4, [3, 7]
    B = Sequence([21, 22, 23], p)                    # prompt 3 个 + 已生成 2 个，本步 decode 1 个
    B.token_ids += [24, 25]
    B.num_computed_tokens, B.block_table = 4, [5, 2]
    C = Sequence([31, 32, 33, 34, 35, 36, 37], p)    # 新请求 prompt 7 个，本步只算前 3 个
    C.num_computed_tokens, C.block_table = 0, [9]
    ids, pos, meta, sample_seqs, sample_rows = engine._prepare_inputs(SchedulerOutput([(A, 2), (B, 1), (C, 3)]))

    assert ids.tolist() == [15, 16, 25, 31, 32, 33], f"input_ids 错误：{ids.tolist()}"
    assert pos.tolist() == [4, 5, 4, 0, 1, 2], f"positions 错误：{pos.tolist()}"
    assert meta.slot_mapping.tolist() == [28, 29, 8, 36, 37, 38], f"slot_mapping 错误：{meta.slot_mapping.tolist()}"
    assert meta.prefill_q_starts == [0, 3] and meta.prefill_q_lens == [2, 3], \
        f"prefill 组应为 A（起点 0、2 个）和 C（起点 3、3 个），实际 {meta.prefill_q_starts}, {meta.prefill_q_lens}"
    assert meta.prefill_ctx_slots[0].tolist() == [12, 13, 14, 15, 28, 29], "A 的上下文槽位错误"
    assert meta.prefill_ctx_slots[1].tolist() == [36, 37, 38], "C 的上下文槽位错误"
    assert meta.decode_rows.tolist() == [2], "decode 组只有 B，位于第 2 行"
    assert meta.decode_slots.shape == (1, 8), "B 有 2 个块，padding 后槽位矩阵为 [1, 8]"
    assert meta.decode_slots[0, :5].tolist() == [20, 21, 22, 23, 8], "B 的前 5 个槽位错误"
    assert meta.decode_mask.tolist() == [[True] * 5 + [False] * 3], "mask 应只覆盖 B 的 5 个有效位置"
    assert [s.seq_id for s in sample_seqs] == [A.seq_id, B.seq_id], "C 的 prompt 没算完，不应采样"
    assert sample_rows == [1, 2], f"A 取第 1 行、B 取第 2 行，实际 {sample_rows}"


# --------------------------------------------------------------------------- 第 8 步
def test_step08a_prefill_logits_match_hf(engine, tiny_model):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32).eval()
    prompt = list(range(30, 70))
    with torch.no_grad():
        ref = hf(torch.tensor([prompt])).logits[0]
    s = Sequence(prompt, SamplingParams(max_tokens=1))
    s.block_table = list(range(10))
    ids, pos, meta, _, _ = engine._prepare_inputs(SchedulerOutput([(s, 40)]))
    with torch.no_grad():
        ours = engine.model.compute_logits(engine.model(ids, pos, engine.kv_caches, meta))
    err = (ours - ref).abs().max().item()
    assert err < 1e-4, f"prefill 的 logits 与 Hugging Face 不一致，最大误差 {err:.2e}（检查 KV 写入与因果掩码）"


def test_step08b_chunked_and_decode_match_hf(tiny_model):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32).eval()
    prompt = list(range(30, 70))
    with torch.no_grad():
        ref = hf.generate(torch.tensor([prompt]), max_new_tokens=6, do_sample=False, eos_token_id=None,
                          pad_token_id=0, min_new_tokens=6)[0, len(prompt):].tolist()
    # 预算 16：40 个 token 的 prompt 会被切成 3 块，覆盖 chunked prefill 与 decode 两条路径
    eng = LLMEngine(EngineConfig(model=tiny_model, device="cpu", num_gpu_blocks=64, block_size=4,
                                 max_model_len=128, max_num_batched_tokens=16, max_num_seqs=8))
    s = eng.generate([prompt], SamplingParams(max_tokens=6, ignore_eos=True))[0]
    assert s.output_token_ids == ref, (f"生成结果与 Hugging Face 不一致：{s.output_token_ids} vs {ref}。"
                                       "第 1 个 token 就错：检查切块时掩码的偏移；之后才错：检查 decode 路径")


# --------------------------------------------------------------------------- 第 10 步（第 9 步是 tests/test_correctness.py）
def test_step10_dispatch_routes_tokens():
    from miniserve.engine import StepOutput
    from miniserve.server import AsyncEngine

    class IdleEngine:                     # 一个永远没有任务的假引擎
        def has_unfinished(self): return False
        def step(self): return []

    async def main():
        ae = AsyncEngine(IdleEngine())
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        s = make_seq(3)
        ae.streams[s.seq_id] = (q, loop)
        ae._dispatch(StepOutput(s, 42, False))
        ae._dispatch(StepOutput(s, None, False))       # 本步没有采样（切块中），不应投递
        s.finish_reason = "length"
        ae._dispatch(StepOutput(s, 43, True))
        got = [await asyncio.wait_for(q.get(), 2), await asyncio.wait_for(q.get(), 2)]
        assert got == [(42, False, None), (43, True, "length")], f"投递内容错误：{got}"
        assert q.empty(), "new_token_id 为 None 的输出不应投递"
        assert s.seq_id not in ae.streams, "请求结束后应从 streams 中移除"
    asyncio.run(main())
