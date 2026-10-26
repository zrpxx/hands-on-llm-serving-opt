# 6.2 FP4 时代：NVFP4 / MXFP4 与低精度通信

> 📎 书外内容

> [!NOTE]
> 本节状态：⬜ 未开始 ｜ 预计用时：__ h ｜ 完成日期：____

## 🎯 学习目标

- [ ] 用 2.5 的量化基础解释 NVFP4 的双层缩放（块级 + 张量级）为什么比单纯 INT4 精度好
- [ ] 对比 MXFP4 与 NVFP4 的块大小和缩放因子格式
- [ ] 理解"MoE dispatch 前把激活量化到 FP4，通信量比 FP16 少 4 倍"这类做法

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

- [vLLM：在 GB200 上推进 WideEP 与大规模 Serving（Part I）](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) —— NVFP4 GEMM、FP4 dispatch、RoPE+Quant 融合等具体做法
- OCP Microscaling (MX) Formats 规范 —— MXFP4/MXFP8 的定义
- NVIDIA 技术博客中关于 NVFP4 的介绍
