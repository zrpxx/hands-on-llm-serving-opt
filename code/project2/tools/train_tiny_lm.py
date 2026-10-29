"""在 CPU 上训练一个小的 Qwen3 结构语言模型，用来做量化实验。

随机权重的模型对量化不敏感，困惑度也没有意义；训练过的小模型才能看出不同量化方法的差别。
默认配置（4 层，hidden 256）在两核 CPU 上训练几分钟即可。
    python tools/train_tiny_lm.py --data data --out /tmp/tiny-lm
"""
import argparse
import math
import os
import time

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None, help="复用已有模型目录中的分词器（项目 3 训练草稿模型时使用）")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(os.cpu_count() or 1)

    text = open(os.path.join(a.data, "train.txt"), encoding="utf-8").read()
    if a.tokenizer:
        hf_tok = PreTrainedTokenizerFast.from_pretrained(a.tokenizer)
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(text.split("\n"), trainers.BpeTrainer(
            vocab_size=a.vocab, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    ids = torch.tensor(hf_tok(text)["input_ids"])

    cfg = Qwen3Config(vocab_size=len(hf_tok), hidden_size=a.hidden, intermediate_size=a.hidden * 3,
                      num_hidden_layers=a.layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=a.hidden // 4, max_position_embeddings=1024, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    t0 = time.time()
    for step in range(a.steps):
        starts = torch.randint(0, ids.numel() - a.seq - 1, (a.batch,))
        x = torch.stack([ids[s:s + a.seq] for s in starts])
        loss = model(x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(a.out)
    hf_tok.save_pretrained(a.out)
    print("已保存到", a.out)


if __name__ == "__main__":
    main()
