import json
import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """从 Hugging Face 的 config.json 中读出 MiniServe 需要的字段。"""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool       # Qwen2 的 q/k/v 投影带 bias
    qk_norm: bool              # Qwen3 在 RoPE 之前对 q、k 做逐头 RMSNorm
    torch_dtype: str
    eos_token_ids: list = field(default_factory=list)

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        c = json.load(open(os.path.join(path, "config.json")))
        model_type = c.get("model_type", "llama")
        if model_type not in ("llama", "qwen2", "qwen3", "mistral"):
            raise ValueError(f"MiniServe 只支持稠密的 llama/qwen2/qwen3/mistral 结构，得到 {model_type}")
        n_heads = c["num_attention_heads"]
        rope = c.get("rope_parameters") or {}
        rope_theta = c.get("rope_theta", rope.get("rope_theta", 10000.0))
        eos = c.get("eos_token_id")
        gen_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_path):
            eos = json.load(open(gen_path)).get("eos_token_id", eos)
        eos_ids = [] if eos is None else (list(eos) if isinstance(eos, list) else [eos])
        return cls(
            model_type=model_type,
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=n_heads,
            num_kv_heads=c.get("num_key_value_heads", n_heads),
            head_dim=c.get("head_dim") or c["hidden_size"] // n_heads,
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            max_position_embeddings=c.get("max_position_embeddings", 4096),
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            attention_bias=c.get("attention_bias", model_type == "qwen2"),
            qk_norm=model_type == "qwen3",
            torch_dtype=c.get("torch_dtype") or c.get("dtype") or "bfloat16",
            eos_token_ids=eos_ids,
        )

    def kv_bytes_per_token(self, elem_bytes: int) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * elem_bytes


@dataclass
class EngineConfig:
    model: str
    device: str = "cuda"
    dtype: str = "auto"                      # auto / bfloat16 / float16 / float32
    block_size: int = 16                     # 每个 KV 块容纳的 token 数
    max_num_seqs: int = 64                   # 同时处于 RUNNING 的请求数上限
    max_num_batched_tokens: int = 2048       # 每一步前向最多处理的 token 数
    max_model_len: int = 4096                # 单个请求的最大长度（输入 + 输出）
    gpu_memory_utilization: float = 0.9
    num_gpu_blocks: int = 0                  # >0 时直接指定 KV 块数，跳过显存测算（CPU 上必须指定）
    policy: str = "continuous"               # continuous / static
    chunked_prefill: bool = True
    seed: int = 0
