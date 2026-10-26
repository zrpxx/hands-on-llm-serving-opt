# 6.6 长上下文：稀疏注意力与 Context Parallelism

> 书外内容

> [!NOTE]
> 状态：未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 学习目标

- [ ] 估算 128K / 1M 上下文下 KV Cache 与注意力计算各自的规模
- [ ] 比较滑动窗口、原生稀疏注意力（NSA）、DeepSeek Sparse Attention（DSA）的选择方式
- [ ] 理解 context parallelism 怎样把一条超长序列切到多卡上

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

- Native Sparse Attention (NSA) 论文
- DeepSeek-V3.2 技术报告中的 DSA 部分
- DeepInfer 的 SWA bounded replay 与 balanced CP（见 [6.1](stage6/01-deepinfer-overview.md)）
