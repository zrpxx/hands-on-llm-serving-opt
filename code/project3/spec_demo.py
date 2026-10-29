"""在 CPU 上做投机解码实验：统计接受率、每次目标模型前向产出的 token 数，并按 3.4 节的公式估算加速比。

    python spec_demo.py --target /tmp/tiny-lm --draft /tmp/draft-lm --data ../project2/data
目标模型和草稿模型必须使用同一个分词器（训练草稿模型时加 --tokenizer 指向目标模型目录）。
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FORGE_IMPL", "starter")))
from forge.spec import greedy_generate, speculative_generate  # noqa: E402


def forward_time(model, n_tokens, reps=5):
    x = torch.randint(5, 500, (1, n_tokens))
    with torch.no_grad():
        model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--data", required=True, help="含 test.txt 的目录")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.target)
    target = AutoModelForCausalLM.from_pretrained(a.target).eval()
    draft = AutoModelForCausalLM.from_pretrained(a.draft).eval()
    ids = tok(open(os.path.join(a.data, "test.txt"), encoding="utf-8").read())["input_ids"]
    step = len(ids) // (a.prompts + 1)
    prompts = [ids[i * step:i * step + a.prompt_len] for i in range(a.prompts)]

    c = forward_time(draft, a.prompt_len + a.new_tokens) / forward_time(target, a.prompt_len + a.new_tokens)
    n_t = sum(p.numel() for p in target.parameters()) / 1e6
    n_d = sum(p.numel() for p in draft.parameters()) / 1e6
    print(f"目标模型 {n_t:.1f}M 参数，草稿模型 {n_d:.1f}M 参数；在这台机器上，草稿一次前向的耗时约为目标的 c = {c:.2f}")
    if a.temperature == 0:
        ref = [greedy_generate(target, p, a.new_tokens) for p in prompts]
    print(f"温度 {a.temperature}，{a.prompts} 个 prompt，每个生成 {a.new_tokens} 个 token\n")
    print(f"{'方法':<12}{'K':>3}{'接受率':>9}{'每次前向产出':>12}{'估算加速比':>11}   各位置的接受比例")
    for method, ks in [("draft", [1, 2, 4, 6]), ("ngram", [4])]:
        for k in ks:
            steps = prop = acc = 0
            per_pos = [0] * k
            produced = 0
            for i, p in enumerate(prompts):
                out, st = speculative_generate(target, p, a.new_tokens, k=k, draft=draft,
                                               temperature=a.temperature, method=method, seed=i)
                if a.temperature == 0:
                    assert out == ref[i], "贪心投机解码的输出必须与目标模型逐 token 一致"
                steps += st.steps; prop += st.proposed; acc += st.accepted; produced += len(out)
                per_pos = [x + y for x, y in zip(per_pos, st.accepted_per_pos)]
            tpf = produced / steps
            cost = 1 + (k * c if method == "draft" else 0)     # 每轮：目标一次前向 + K 次草稿前向
            pos = " ".join(f"{x / steps:.2f}" for x in per_pos)
            name = "草稿模型" if method == "draft" else "n-gram"
            print(f"{name:<10}{k:>5}{acc / max(1, prop):>9.2f}{tpf:>12.2f}{tpf / cost:>11.2f}   {pos}")


if __name__ == "__main__":
    main()
