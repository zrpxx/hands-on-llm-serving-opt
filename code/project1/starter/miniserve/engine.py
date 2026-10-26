import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from .config import EngineConfig, ModelConfig
from .kv_cache import BlockAllocator, allocate_kv_cache
from .model import AttentionMetadata, CausalLM
from .scheduler import Scheduler, SchedulerOutput
from .sequence import SamplingParams, Sequence

logger = logging.getLogger("miniserve")

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class StepOutput:
    seq: Sequence
    new_token_id: Optional[int]      # 本步没有采样（例如 prefill 只做了一个分块）时为 None
    finished: bool


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model_cfg = ModelConfig.from_pretrained(cfg.model)
        if cfg.dtype == "auto":
            dtype = DTYPES.get(self.model_cfg.torch_dtype, torch.bfloat16)
            if cfg.device == "cpu" and dtype != torch.float32:
                dtype = torch.float32
        else:
            dtype = DTYPES[cfg.dtype]
        self.dtype = dtype
        if not cfg.chunked_prefill and cfg.max_num_batched_tokens < cfg.max_model_len:
            raise ValueError("关闭 chunked prefill 时，max_num_batched_tokens 必须不小于 max_model_len")
        torch.manual_seed(cfg.seed)

        t0 = time.perf_counter()
        self.model = CausalLM.from_pretrained(cfg.model, self.model_cfg, dtype, cfg.device)
        weight_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        logger.info("权重加载完成：%.2f GB，用时 %.1f s", weight_bytes / 1e9, time.perf_counter() - t0)

        num_blocks = cfg.num_gpu_blocks or self._profile_num_blocks()
        self.kv_caches = allocate_kv_cache(self.model_cfg, num_blocks, cfg.block_size, dtype, cfg.device)
        self.allocator = BlockAllocator(num_blocks)
        self.num_blocks = num_blocks
        logger.info("KV Cache：%d 块 × %d = %d token；按 max_model_len=%d 计可容纳 %.1f 个请求",
                    num_blocks, cfg.block_size, num_blocks * cfg.block_size, cfg.max_model_len,
                    num_blocks * cfg.block_size / cfg.max_model_len)

        self.scheduler = Scheduler(self.allocator, cfg.block_size, cfg.max_num_seqs,
                                   cfg.max_num_batched_tokens, cfg.policy, cfg.chunked_prefill)
        self.eos_ids = set(self.model_cfg.eos_token_ids)
        self.stats = {"steps": 0, "tokens": 0, "forward_time": 0.0}

    # ------------------------------------------------------------------ 显存测算
    @torch.inference_mode()
    def _profile_num_blocks(self) -> int:
        """用一次最大规模的 prefill 测出激活峰值，再把剩余显存全部分给 KV Cache。"""
        if not self.cfg.device.startswith("cuda"):
            raise ValueError("非 CUDA 设备请用 num_gpu_blocks 直接指定 KV 块数")
        bs = self.cfg.block_size
        n = min(self.cfg.max_num_batched_tokens, self.cfg.max_model_len)
        tmp_blocks = math.ceil(n / bs)
        tmp_cache = allocate_kv_cache(self.model_cfg, tmp_blocks, bs, self.dtype, self.cfg.device)
        tmp_bytes = sum(k.numel() * k.element_size() * 2 for k, _ in tmp_cache)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.zeros(n, dtype=torch.long, device=self.cfg.device)
        pos = torch.arange(n, device=self.cfg.device)
        slots = torch.arange(n, device=self.cfg.device)
        meta = AttentionMetadata(slots, [0], [n], [slots], None, None, None)
        hidden = self.model(ids, pos, tmp_cache, meta)
        self.model.compute_logits(hidden[-1:])
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - tmp_bytes
        del tmp_cache, hidden
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        non_torch = total - free - torch.cuda.memory_reserved()      # CUDA 上下文等
        budget = total * self.cfg.gpu_memory_utilization - peak - non_torch
        per_block = self.model_cfg.kv_bytes_per_token(torch.tensor([], dtype=self.dtype).element_size()) * bs
        num_blocks = int(budget // per_block)
        logger.info("显存测算：总 %.1f GB，激活+权重峰值 %.2f GB，其他 %.2f GB，KV 预算 %.2f GB",
                    total / 1e9, peak / 1e9, non_torch / 1e9, budget / 1e9)
        if num_blocks <= 0:
            raise RuntimeError("显存不足以分配 KV Cache，尝试降低 max_num_batched_tokens 或换更小的模型")
        return num_blocks

    # ------------------------------------------------------------------ 请求接口
    def add_request(self, prompt_token_ids: List[int], params: SamplingParams,
                    seq_id: Optional[str] = None) -> Sequence:
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt 不能为空")
        if len(prompt_token_ids) >= self.cfg.max_model_len:
            raise ValueError(f"prompt 长度 {len(prompt_token_ids)} 超过 max_model_len={self.cfg.max_model_len}")
        if math.ceil(self.cfg.max_model_len / self.cfg.block_size) > self.num_blocks:
            raise ValueError("KV 块总数不足以容纳一个 max_model_len 长度的请求")
        seq = Sequence(prompt_token_ids, params, seq_id)
        self.scheduler.add(seq)
        return seq

    def abort_request(self, seq_id: str):
        self.scheduler.abort(seq_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # ------------------------------------------------------------------ 一步
    @torch.inference_mode()
    def step(self) -> List[StepOutput]:
        sched = self.scheduler.schedule()
        if not sched.scheduled:
            return []
        t0 = time.perf_counter()
        input_ids, positions, meta, sample_seqs, sample_rows = self._prepare_inputs(sched)
        hidden = self.model(input_ids, positions, self.kv_caches, meta)
        outputs: List[StepOutput] = []
        new_tokens = []
        if sample_rows:
            logits = self.model.compute_logits(hidden[sample_rows])
            new_tokens = self._sample(logits, [s.params for s in sample_seqs]).tolist()
        if self.cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.stats["forward_time"] += time.perf_counter() - t0
        self.stats["steps"] += 1
        self.stats["tokens"] += sched.num_tokens

        for seq, n in sched.scheduled:
            seq.num_computed_tokens += n
        now = time.perf_counter()
        token_of = dict(zip((s.seq_id for s in sample_seqs), new_tokens))
        for seq, _ in sched.scheduled:
            tok = token_of.get(seq.seq_id)
            if tok is None:
                outputs.append(StepOutput(seq, None, False))
                continue
            seq.token_ids.append(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            reason = self._check_stop(seq, tok)
            if reason:
                self.scheduler.finish(seq, reason)
            outputs.append(StepOutput(seq, tok, reason is not None))
        return outputs

    def _prepare_inputs(self, sched: SchedulerOutput):
        # TODO(1-3)：
        # 把调度结果展平成一次前向的输入：input_ids、positions、slot_mapping，
        # 以及 AttentionMetadata 中 prefill 与 decode 两组所需的槽位信息；同时记录哪些请求需要采样、取哪一行的 hidden。
        # 返回 (input_ids, positions, meta, sample_seqs, sample_rows)。
        raise NotImplementedError("TODO(1-3): LLMEngine._prepare_inputs")

    def _sample(self, logits: torch.Tensor, params: List[SamplingParams]) -> torch.Tensor:
        greedy = logits.argmax(dim=-1)
        temps = torch.tensor([p.temperature for p in params], device=logits.device)
        if (temps == 0).all():
            return greedy
        probs = torch.softmax(logits / temps.clamp(min=1e-5)[:, None], dim=-1)
        top_p = torch.tensor([p.top_p for p in params], device=logits.device)
        if (top_p < 1).any():
            sorted_p, idx = probs.sort(dim=-1, descending=True)
            drop = sorted_p.cumsum(-1) - sorted_p > top_p[:, None]
            sorted_p[drop] = 0
            probs = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)
        sampled = torch.multinomial(probs, 1).squeeze(1)
        return torch.where(temps == 0, greedy, sampled)

    def _check_stop(self, seq: Sequence, tok: int) -> Optional[str]:
        p = seq.params
        if not p.ignore_eos and (tok in self.eos_ids or tok in p.stop_token_ids):
            return "stop"
        if seq.num_output_tokens >= p.max_tokens or seq.num_tokens >= self.cfg.max_model_len:
            return "length"
        return None

    # ------------------------------------------------------------------ 离线批量生成
    def generate(self, prompts: List[List[int]], params: SamplingParams) -> List[Sequence]:
        seqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished():
            self.step()
        return seqs
