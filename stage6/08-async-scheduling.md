# 6.8 异步调度：把 CPU 开销藏起来

> 📎 书外内容

> [!NOTE]
> 本节状态：⬜ 未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 🎯 学习目标

- [ ] 用 profiler 量化小模型高并发下 CPU 调度、采样、detokenize 占 decode step 的比例
- [ ] 理解"提前一步调度"如何让 CPU 与 GPU 重叠
- [ ] 对比 SGLang overlap scheduler、vLLM async scheduling、DeepInfer 异步调度的做法

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

- SGLang 的 zero-overhead batch scheduler 相关博客
- vLLM async scheduling 相关文档
- DeepInfer 调度器源码（见 [6.1](stage6/01-deepinfer-overview.md)）
