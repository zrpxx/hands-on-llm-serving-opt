"""MiniServe：一个用于学习的最小 LLM 推理引擎。

支持 Llama / Qwen2 / Qwen3 一类的稠密 Decoder-only 模型，实现了：
- 自写的模型前向（扁平化的 token 批，与 vLLM 相同的组织方式）
- 分页式 KV Cache（block table + 空闲块池）
- Continuous Batching、Chunked Prefill、基于重算的抢占
- OpenAI 兼容的 HTTP 接口与 SSE 流式输出
"""

from .config import EngineConfig, ModelConfig
from .sequence import SamplingParams, Sequence
from .engine import LLMEngine

__all__ = ["EngineConfig", "ModelConfig", "SamplingParams", "Sequence", "LLMEngine"]
