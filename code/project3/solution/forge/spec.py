"""第 6 步：投机解码。对应 3.4、3.5 节。

- rejection_sample：给定草稿模型提议的 K 个 token 及其概率分布，以及目标模型在 K+1 个位置上的分布，
  决定接受哪些 token，并在第一个被拒绝的位置从修正后的分布 max(0, p - q) 中重新采样。
  这样得到的 token 与直接从目标模型采样同分布。
- ngram_propose：不需要草稿模型，在已有的 token 中查找与结尾相同的 n-gram，把它后面的 token 作为草稿。
- speculative_generate：用目标模型和草稿模型（或 n-gram）生成，并统计接受率和目标模型的前向次数。

为了让代码短一些，模型每次都对完整序列做前向，不使用 KV Cache。真实的引擎会缓存 KV，
被拒绝的草稿 token 对应的 KV 要丢弃。
"""
from dataclasses import dataclass, field

import torch


def rejection_sample(target_probs: torch.Tensor, draft_probs: torch.Tensor, draft_tokens: torch.Tensor,
                     generator: torch.Generator = None) -> list:
    """target_probs: [K+1, V]，第 i 行是目标模型在看到前 i 个草稿 token 之后的分布；
    draft_probs: [K, V]，第 i 行是草稿模型提议第 i 个 token 时的分布；draft_tokens: [K]。
    返回这一步最终输出的 token 列表，长度在 1 到 K+1 之间。"""
    K = draft_tokens.numel()
    out = []
    for i in range(K):
        t = int(draft_tokens[i])
        p, q = target_probs[i, t], draft_probs[i, t]
        r = torch.rand((), generator=generator)
        if q > 0 and r < torch.clamp(p / q, max=1.0):     # 以 min(1, p/q) 的概率接受
            out.append(t)
            continue
        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
        if residual.sum() <= 0:                          # p 与 q 完全相同，理论上不会走到这里
            residual = target_probs[i]
        out.append(int(torch.multinomial(residual / residual.sum(), 1, generator=generator)))
        return out
    # 全部接受：再从目标模型的第 K+1 个分布中采样一个"额外"的 token
    out.append(int(torch.multinomial(target_probs[K], 1, generator=generator)))
    return out


def greedy_verify(target_logits: torch.Tensor, draft_tokens: torch.Tensor) -> list:
    """贪心解码时的验证：从头比较草稿与目标模型的 argmax，第一个不同的位置用目标的 argmax 替换。"""
    pred = target_logits.argmax(-1)                      # [K+1]
    out = []
    for i in range(draft_tokens.numel()):
        out.append(int(pred[i]))
        if int(pred[i]) != int(draft_tokens[i]):
            return out
    out.append(int(pred[-1]))
    return out


def ngram_propose(tokens: list, k: int, n_max: int = 4, n_min: int = 2) -> list:
    """在 tokens 中查找与结尾 n 个 token（n 从 n_max 到 n_min）相同的、最近一次出现的位置，
    返回它后面最多 k 个 token。找不到时返回空列表。"""
    L = len(tokens)
    for n in range(min(n_max, L - 1), n_min - 1, -1):
        tail = tokens[L - n:]
        for start in range(L - n - 1, -1, -1):           # 从近到远，不包括结尾本身
            if tokens[start:start + n] == tail:
                follow = tokens[start + n:start + n + k]
                if follow:
                    return follow
    return []


@dataclass
class SpecStats:
    steps: int = 0                     # 目标模型的前向次数
    proposed: int = 0
    accepted: int = 0
    accepted_per_pos: list = field(default_factory=list)

    @property
    def acceptance_rate(self):
        return self.accepted / max(1, self.proposed)


def _probs(logits, temperature):
    return torch.softmax(logits.float() / temperature, dim=-1)


@torch.no_grad()
def speculative_generate(target, prompt_ids: list, max_new_tokens: int, k: int = 4, draft=None,
                         temperature: float = 0.0, method: str = "draft", seed: int = 0):
    """method="draft" 使用草稿模型，"ngram" 使用 n-gram 提议。temperature=0 时为贪心解码。
    返回 (生成的 token 列表, SpecStats)。"""
    g = torch.Generator().manual_seed(seed)
    tokens = list(prompt_ids)
    stats = SpecStats(accepted_per_pos=[0] * k)
    while len(tokens) - len(prompt_ids) < max_new_tokens:
        # 1. 提议
        if method == "ngram":
            proposal = ngram_propose(tokens, k)
            draft_probs = None
        else:
            proposal, rows = [], []
            for _ in range(k):
                logits = draft(torch.tensor([tokens + proposal])).logits[0, -1]
                if temperature == 0:
                    proposal.append(int(logits.argmax()))
                else:
                    pr = _probs(logits, temperature)
                    rows.append(pr)
                    proposal.append(int(torch.multinomial(pr, 1, generator=g)))
            draft_probs = torch.stack(rows) if rows else None
        # 2. 目标模型一次前向，得到 len(proposal)+1 个位置的分布
        logits = target(torch.tensor([tokens + proposal])).logits[0, len(tokens) - 1:]
        stats.steps += 1
        d = torch.tensor(proposal, dtype=torch.long)
        if not proposal:
            new = greedy_verify(logits, d) if temperature == 0 else \
                [int(torch.multinomial(_probs(logits[0], temperature), 1, generator=g))]
        elif temperature == 0:
            new = greedy_verify(logits, d)
        elif method == "ngram":                          # n-gram 的提议是确定的，相当于 q 是 one-hot
            q = torch.zeros(len(proposal), logits.shape[-1])
            q[torch.arange(len(proposal)), d] = 1.0
            new = rejection_sample(_probs(logits, temperature), q, d, g)
        else:
            new = rejection_sample(_probs(logits, temperature), draft_probs, d, g)
        # 3. 统计
        n_acc = min(len(new) - 1, len(proposal))
        stats.proposed += len(proposal)
        stats.accepted += n_acc
        for i in range(n_acc):
            stats.accepted_per_pos[i] += 1
        tokens += new
    return tokens[len(prompt_ids):len(prompt_ids) + max_new_tokens], stats


@torch.no_grad()
def greedy_generate(model, prompt_ids: list, max_new_tokens: int) -> list:
    tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        tokens.append(int(model(torch.tensor([tokens])).logits[0, -1].argmax()))
    return tokens[len(prompt_ids):]
