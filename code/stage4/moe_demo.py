"""在 CPU 上模拟一个 MoE 层的专家并行：路由、dispatch、专家计算、combine。对应 4.2 节。

    python code/stage4/moe_demo.py --experts 64 --topk 4 --ep 8

步骤与 vLLM、SGLang 中的实现相同：
1. 路由器为每个 token 打分，选出 top-k 个专家及其权重；
2. dispatch：把每个 (token, 专家) 对发往该专家所在的卡，这是一次 all-to-all；
3. 每张卡把发给同一个专家的 token 收集起来，做一次矩阵乘法（grouped GEMM）；
4. combine：把结果送回 token 原来所在的卡，按路由权重加权求和，又是一次 all-to-all。
"""
import argparse

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--inter", type=int, default=128)
    a = ap.parse_args()
    torch.manual_seed(0)
    E, k, H, I = a.experts, a.topk, a.hidden, a.inter
    x = torch.randn(a.tokens, H, dtype=torch.float64)
    router = torch.randn(H, E, dtype=torch.float64) * 0.3
    w1 = torch.randn(E, 2 * I, H, dtype=torch.float64) / H ** 0.5      # 每个专家一个 SwiGLU FFN
    w2 = torch.randn(E, H, I, dtype=torch.float64) / I ** 0.5

    def expert(e, h):
        g, u = (h @ w1[e].T).chunk(2, -1)
        return (F.silu(g) * u) @ w2[e].T

    # 1. 路由
    probs = torch.softmax(x @ router, -1)
    topw, topi = probs.topk(k, -1)
    topw = topw / topw.sum(-1, keepdim=True)

    # 参考结果：逐个 token、逐个专家计算
    ref = torch.zeros_like(x)
    for t in range(a.tokens):
        for j in range(k):
            ref[t] += topw[t, j] * expert(int(topi[t, j]), x[t:t + 1])[0]

    # 2～4. 按专家分组计算，并统计每张卡收到的 token 数
    flat_e = topi.reshape(-1)                                          # 每个 (token, 专家) 对的专家编号
    flat_t = torch.arange(a.tokens).repeat_interleave(k)
    order = torch.argsort(flat_e, stable=True)                         # dispatch：按专家排序
    out = torch.zeros_like(x)
    per_rank = [0] * a.ep
    for e in range(E):
        idx = order[flat_e[order] == e]
        if idx.numel() == 0:
            continue
        per_rank[e // (E // a.ep)] += idx.numel()
        y = expert(e, x[flat_t[idx]])                                  # grouped GEMM 中的一组
        out.index_add_(0, flat_t[idx], y * topw.reshape(-1)[idx, None])  # combine

    print(f"{E} 个专家、每个 token 选 {k} 个、EP = {a.ep}（每张卡 {E // a.ep} 个专家）")
    print(f"分组计算与逐个计算的最大差：{(out - ref).abs().max().item():.2e}")
    mean = a.tokens * k / a.ep
    print("每张卡收到的 (token, 专家) 对：", per_rank, f"；最忙的是平均的 {max(per_rank) / mean:.2f} 倍")
    # 假设 token 均匀分布在各卡上（数据并行的注意力），发往其他卡的比例是 (EP-1)/EP
    bytes_ = 2 * a.tokens * k * H * 2 * (a.ep - 1) / a.ep
    print(f"dispatch + combine 的跨卡数据量（BF16）：{bytes_ / 2**20:.1f} MiB，"
          f"是这一层输入激活（{a.tokens * H * 2 / 2**20:.1f} MiB）的 {bytes_ / (a.tokens * H * 2):.1f} 倍")


if __name__ == "__main__":
    main()
