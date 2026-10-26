# 6.7 批不变推理与确定性采样

> 书外内容

> [!NOTE]
> 状态：未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 学习目标

- [ ] 讲清楚同一个请求为什么在不同批组合下输出会不同：归约顺序随 batch 大小变化
- [ ] 了解批不变算子（matmul / RMSNorm / attention）的实现方式和性能代价
- [ ] 理解它对 RL 训推一致性（rollout 与训练前向对齐）的意义

## 要点

## 核心公式

## 动手实验

```bash
# 实验命令
```

| 配置 | 吞吐 (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | 显存 (GB) |
|---|---|---|---|---|
| baseline | | | | |
| 本节优化 | | | | |

**结论（一句话，带数字）**：

## 自测题

1. 
2. 

## 踩坑记录

## 资料

- Thinking Machines Lab：Defeating Nondeterminism in LLM Inference（2025-09）
- vLLM 的 batch invariant 模式文档
- DeepInfer 的 `DEEPINFER_BATCH_INVARIANT`（见 [6.1](stage6/01-deepinfer-overview.md)）
- 书 Ch10「Determinism in RL Serving」
