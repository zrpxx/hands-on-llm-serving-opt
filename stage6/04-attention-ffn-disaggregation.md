# 6.4 Attention-FFN 分离（AFD）

> 📎 书外内容

> [!NOTE]
> 本节状态：⬜ 未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 🎯 学习目标

- [ ] 说明 AFD 和 PD 分离分别解决什么问题：前者按算子类型拆，后者按阶段拆
- [ ] 用算术强度解释 Attention（访存密集）与 FFN/MoE（计算密集）为什么适合放在不同硬件上
- [ ] 理解 ping-pong 流水线如何掩盖 Attention 与 FFN 之间的通信

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

- [Step-3 is Large yet Affordable (arXiv 2507.19427)](https://arxiv.org/abs/2507.19427)
- [MegaScale-Infer (arXiv 2504.02263)](https://arxiv.org/abs/2504.02263)
- [vLLM AFD Plugin 发布博客（2026-07）](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin)
- [How Far Can Disaggregation Go? AFD 设计空间探索](https://www.alphaxiv.org/abs/2605.28302)
