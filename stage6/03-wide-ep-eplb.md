# 6.3 Wide EP 与 EPLB：大规模 MoE 专家并行

> 📎 书外内容

> [!NOTE]
> 本节状态：⬜ 未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 🎯 学习目标

- [ ] 讲清楚 Wide EP 相比"TP 切 MoE"在显存和通信上的取舍
- [ ] 理解 EPLB 用冗余专家副本平衡热点专家的思路
- [ ] 读 DeepEP 的 dispatch / combine 接口，区分高吞吐模式和低延迟模式

## 📝 要点

## 🧮 核心公式

## 🔬 动手实验

```bash
# 实验命令
```

| 配置 | 吞吐 (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | 显存 (GB) |
|---|---|---|---|---|
| baseline | | | | |
| 本节优化 | | | | |

**结论（一句话，带数字）**：

## ❓ 自测题

1. 
2. 

## 🕳️ 踩坑记录

## 🔗 资料

- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)
- [vLLM WideEP on GB200 博客](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)
- [MegaScale-Infer (arXiv 2504.02263)](https://arxiv.org/abs/2504.02263)
- DeepInfer 中 EP + EPLB 的实现（见 [6.1](stage6/01-deepinfer-overview.md)）
