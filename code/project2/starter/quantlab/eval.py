"""困惑度评测。"""
import math

import torch


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seq_len: int = 256, max_windows: int = 0) -> float:
    """ids: 一维 token 张量。按不重叠的窗口计算平均负对数似然，返回 exp(NLL)。"""
    nll, count = 0.0, 0
    starts = list(range(0, ids.numel() - seq_len - 1, seq_len))
    if max_windows:
        starts = starts[:max_windows]
    for s in starts:
        x = ids[s:s + seq_len].unsqueeze(0)
        y = ids[s + 1:s + seq_len + 1].unsqueeze(0)
        logits = model(x).logits.float()
        nll += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                 reduction="sum").item()
        count += y.numel()
    return math.exp(nll / count)
