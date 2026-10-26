"""调度器单元测试：不依赖模型，可以在完成 1-2（以及 1-3 的块管理部分）后单独运行。

    pytest tests/test_scheduler.py
"""
from miniserve.kv_cache import BlockAllocator
from miniserve.scheduler import Scheduler
from miniserve.sequence import SamplingParams, Sequence, SeqStatus


def make(num_blocks=100, block_size=4, max_seqs=8, budget=16, **kw):
    return Scheduler(BlockAllocator(num_blocks), block_size, max_seqs, budget, **kw)


def seq(n_prompt, max_tokens=4):
    return Sequence(list(range(1, n_prompt + 1)), SamplingParams(max_tokens=max_tokens, ignore_eos=True))


def advance(sched, out):
    """模拟引擎：更新已计算 token 数；prompt 全部算完的请求追加一个假 token，生成够了就结束。"""
    for s, n in out.scheduled:
        s.num_computed_tokens += n
        if s.num_computed_tokens == s.num_tokens:
            s.token_ids.append(0)
            if s.num_output_tokens >= s.params.max_tokens:
                sched.finish(s, "length")


def test_token_budget_and_chunking():
    sch = make(budget=10)
    a, b = seq(6), seq(6)
    sch.add(a); sch.add(b)
    out = sch.schedule()
    assert [(s.seq_id, n) for s, n in out.scheduled] == [(a.seq_id, 6), (b.seq_id, 4)]   # b 只处理一部分
    advance(sch, out)
    out = sch.schedule()
    assert [(s.seq_id, n) for s, n in out.scheduled] == [(a.seq_id, 1), (b.seq_id, 2)]   # a 进入 decode，b 处理剩余 prompt


def test_max_num_seqs():
    sch = make(max_seqs=2, budget=100)
    for _ in range(3):
        sch.add(seq(3))
    out = sch.schedule()
    assert len(out.scheduled) == 2 and len(sch.waiting) == 1


def test_blocks_allocated_and_freed():
    sch = make(num_blocks=10, block_size=4, budget=100)
    s = seq(9, max_tokens=1)
    sch.add(s)
    out = sch.schedule()
    assert len(s.block_table) == 3            # 9 个 token 需要 3 个 4-token 块
    advance(sch, out)                          # max_tokens=1，生成一个就结束
    assert s.status == SeqStatus.FINISHED and sch.allocator.num_free == 10


def test_continuous_admits_while_running():
    sch = make(budget=100)
    a = seq(4, max_tokens=10)
    sch.add(a)
    advance(sch, sch.schedule())
    b = seq(4)
    sch.add(b)
    out = sch.schedule()
    assert {s.seq_id for s, _ in out.scheduled} == {a.seq_id, b.seq_id}


def test_static_waits_for_batch():
    sch = make(budget=100, policy="static")
    a = seq(4, max_tokens=10)
    sch.add(a)
    advance(sch, sch.schedule())
    sch.add(seq(4))
    out = sch.schedule()
    assert [s.seq_id for s, _ in out.scheduled] == [a.seq_id]


def test_preempts_latest_arrival():
    # 6 个块 = 24 token。两个 8-token 的请求各占 2 块，decode 增长后放不下，后到的被抢占
    sch = make(num_blocks=6, block_size=4, budget=100)
    a, b = seq(8, max_tokens=12), seq(8, max_tokens=12)
    sch.add(a); sch.add(b)
    for _ in range(12):
        out = sch.schedule()
        if out.num_preempted:
            break
        advance(sch, out)
    assert out.num_preempted >= 1
    assert b.status == SeqStatus.WAITING and b.num_computed_tokens == 0 and b.block_table == []
    assert sch.waiting[0] is b and a.status == SeqStatus.RUNNING
