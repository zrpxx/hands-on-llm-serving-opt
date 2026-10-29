"""用 PyTorch Profiler 剖析 decode 的一步。对应 3.6 节。

在 CPU 上也能运行：默认现场生成一个随机权重的 Qwen3 结构小模型。有 GPU 时加 --device cuda，
可以看到每个算子在 GPU 上的耗时，以及 kernel 之间的空隙。

    python code/stage3/profile_decode.py                     # CPU，小模型
    python code/stage3/profile_decode.py --model Qwen/Qwen3-0.6B --device cuda
输出：按耗时排序的算子表，以及 trace.json（用 https://ui.perfetto.dev 打开）。
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, DynamicCache, Qwen3Config, Qwen3ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Hugging Face 模型名或本地目录；不指定时生成一个随机的小模型")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="trace.json")
    a = ap.parse_args()

    if a.model:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.device != "cpu" else torch.float32)
    else:
        cfg = Qwen3Config(vocab_size=8192, hidden_size=512, intermediate_size=1536, num_hidden_layers=8,
                          num_attention_heads=8, num_key_value_heads=4, head_dim=64)
        model = Qwen3ForCausalLM(cfg)
    model = model.to(a.device).eval()
    ids = torch.randint(10, 1000, (1, a.prompt_len), device=a.device)

    with torch.no_grad():
        cache = DynamicCache()
        out = model(ids, past_key_values=cache, use_cache=True)            # prefill，不计入剖析
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(2):                                                 # 预热
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if a.device.startswith("cuda") else [])
        with profile(activities=acts, record_shapes=True) as prof:
            for _ in range(a.steps):
                with record_function("decode_step"):
                    out = model(nxt, past_key_values=cache, use_cache=True)
                    nxt = out.logits[:, -1:].argmax(-1)
                if a.device.startswith("cuda"):
                    torch.cuda.synchronize()

    sort = "cuda_time_total" if a.device.startswith("cuda") else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=12))
    steps = [e for e in prof.events() if e.name == "decode_step"]
    if steps:
        avg = sum(e.cpu_time_total for e in steps) / len(steps)
        n_ops = sum(1 for e in prof.events() if e.name.startswith("aten::")) / len(steps)
        print(f"平均每步 {avg / 1000:.2f} ms，每步约 {n_ops:.0f} 次 aten 算子调用")
    prof.export_chrome_trace(a.out)
    print("时间线已保存到", a.out)


if __name__ == "__main__":
    main()
