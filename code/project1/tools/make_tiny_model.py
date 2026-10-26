"""生成一个随机初始化的微型 Qwen3 模型（含分词器），用于在 CPU 上开发和测试 MiniServe。

输出目录的格式与 Hugging Face 一致，可以被 transformers、MiniServe、vLLM 直接加载。
权重是随机的，生成的文本没有意义，只用来验证正确性与流程。

用法：
    python tools/make_tiny_model.py --out /tmp/tiny-qwen3
"""
import argparse

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Large language model serving needs batching, caching and careful scheduling.",
    "KV cache stores keys and values for every token that has been processed.",
    "推理服务需要关注首 token 延迟、每 token 延迟和吞吐。",
] * 50

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>",
                                     pad_token="<|endoftext|>", bos_token=None)
    hf_tok.chat_template = CHAT_TEMPLATE
    eos_id = hf_tok.convert_tokens_to_ids("<|im_end|>")

    torch.manual_seed(a.seed)
    cfg = Qwen3Config(
        vocab_size=len(hf_tok), hidden_size=a.hidden, intermediate_size=a.hidden * 2,
        num_hidden_layers=a.layers, num_attention_heads=4, num_key_value_heads=2, head_dim=a.hidden // 4,
        max_position_embeddings=4096, tie_word_embeddings=False, eos_token_id=eos_id,
    )
    model = Qwen3ForCausalLM(cfg)
    # 默认初始化的输出几乎是均匀分布，放大 lm_head 让贪心解码的结果更稳定、更容易比对
    with torch.no_grad():
        model.lm_head.weight.mul_(20)
    model.save_pretrained(a.out)
    hf_tok.save_pretrained(a.out)
    print(f"已生成：{a.out}  vocab={len(hf_tok)} layers={a.layers} hidden={a.hidden}")


if __name__ == "__main__":
    main()
