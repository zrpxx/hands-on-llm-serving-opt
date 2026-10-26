# 6.5 投机解码新形态：MTP 与置信度调度验证

> 📎 书外内容

> [!NOTE]
> 本节状态：⬜ 未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 🎯 学习目标

- [ ] 区分三类草稿来源：独立小模型、附加 head（Medusa/EAGLE）、模型自带的 MTP 层
- [ ] 理解为什么模型训练时就带 MTP 能让投机解码更"免费"
- [ ] 分析 DeepInfer DSpark 用置信度安排验证时机的意义：高并发下减少无效验证

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

- DeepSeek-V3 技术报告中的 Multi-Token Prediction 部分
- EAGLE-3 论文
- DeepInfer README 中的 DSpark 说明（见 [6.1](stage6/01-deepinfer-overview.md)）
