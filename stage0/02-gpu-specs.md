# 0.2 GPU 规格解读

> 对应书中第 5 章 The Role of Accelerator Chips in LLM Serving

## 学习目标

- 从 GPU 规格表中找出与推理相关的四项指标：算力、显存容量、显存带宽、互联带宽，并说明每一项影响什么。
- 识别规格表中"稀疏算力"与"稠密算力"的区别，避免估算时差一倍。
- 根据模型规模和延迟要求，初步判断应该选哪类 GPU、是否需要多卡。

## 1. 与推理相关的四项指标

| 指标 | 单位 | 决定什么 |
|---|---|---|
| 算力（Tensor Core FLOPS） | TFLOPS | 计算密集阶段（prefill、大批量）的速度上限 |
| 显存容量 | GB | 能否装下权重，以及剩余多少给 KV Cache（决定并发和上下文长度） |
| 显存带宽 | TB/s | 访存密集阶段（decode）的速度上限 |
| 互联带宽 | GB/s | 多卡切分模型时的通信开销 |

书中用厨房做比喻：算力是烤箱，显存带宽是送面团的速度，显存容量是冰箱。冰箱装不下就开不了张；烤箱再快，面团送不过来也只能空转。三者要和负载匹配。

## 2. 读规格表时的两个陷阱

### 2.1 稀疏与稠密

NVIDIA 从 Ampere 开始支持 2:4 结构化稀疏，规格表中 Hopper 系列通常列的是**稀疏**算力。书中表 5-1、5-4 给 H100 SXM 的 BF16 算力是 1979 TFLOPS，这是稀疏值；稠密矩阵乘的峰值是它的一半，约 989 TFLOPS。而同一张表里 A100 的 312 TFLOPS、L40S 的 362 TFLOPS 是稠密值。

LLM 推理绝大多数情况下用的是稠密矩阵乘，**估算时一律用稠密算力**。本笔记后面的所有计算都按稠密值。

### 2.2 精度

同一张卡在不同精度下的算力不同。FP8 的稠密算力约为 BF16 的两倍；不支持 FP8 张量计算的卡（如 A100、A800），FP8 只能用于减少存储与访存，计算时要先反量化。

## 3. 常见 GPU 对比

以下为稠密算力。价格随时间与地区变化较大，这里不列。

| GPU | 显存 | 显存带宽 | BF16 稠密算力 | FP8 张量计算 | 机内互联 |
|---|---|---|---|---|---|
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | ~989 TFLOPS | 支持 | NVLink 900 GB/s |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | ~989 TFLOPS | 支持 | NVLink 900 GB/s |
| A100 / A800 80GB SXM | 80 GB HBM2e | 2.04 TB/s | 312 TFLOPS | 不支持 | A100 600 GB/s；A800 400 GB/s |
| L40S | 48 GB GDDR6 | 0.864 TB/s | 362 TFLOPS | 支持 | 仅 PCIe |
| A10 | 24 GB GDDR6 | 0.6 TB/s | 125 TFLOPS | 不支持 | 仅 PCIe |

> [!NOTE]
> 书中表 5-4 给 A100 的带宽是 1.935 TB/s，那是 A100 80GB **PCIe** 版的数字；SXM 版为 2.039 TB/s。A800 是面向中国市场的 A100 变体，算力与显存相同，主要差别是 NVLink 带宽从 600 GB/s 降到 400 GB/s。

**同代不同型号的取舍。** 书中比较了 H100 SXM 与 H100 NVL：NVL 算力略低，但显存更大（94 GB）、带宽更高（3.9 TB/s）。对 decode 为主的负载，NVL 反而可能更快。不能只看算力一项。

## 4. 互联

### 4.1 机内

| 形态 | GPU 间带宽 | 说明 |
|---|---|---|
| PCIe | 128 GB/s（PCIe Gen5 x16 双向） | 最便宜，适合各卡独立跑小模型 |
| NVLink Bridge（如 H100 NVL） | 600 GB/s | 只能两两相连，跨对仍走 PCIe |
| NVLink + NVSwitch（SXM） | 900 GB/s | 8 卡全互联，每对 GPU 都能跑满带宽 |

没有 NVSwitch 时，SXM 卡的 900 GB/s 要平分给另外 7 张卡，每条点对点链路约 128 GB/s。

### 4.2 跨机

跨机通常用 InfiniBand 配合 GPUDirect RDMA。NDR 400G InfiniBand 单卡约 50 GB/s，比机内 NVLink 低一个数量级以上。这就是 0.1 中"优先单机多卡"的原因：大多数部署中，每个模型副本限制在一台机器的 1–8 张卡内，流量增长时横向加副本。

需要跨机的情况主要是超大 MoE 模型（例如 DeepSeek-V3/R1）的专家并行，以及 Prefill-Decode 分离。第 4、6 部分会讨论。

### 4.3 机柜级系统

GB200 NVL72 把 72 张 Blackwell GPU 和 36 颗 Grace CPU 放进一个 NVLink 域，GPU 之间都能以 NVLink 带宽通信。它主要服务于大规模 MoE 的分离式部署，目前只有少数大厂在用。

## 5. 功耗

功耗通常用 TDP（热设计功耗）表示，数据中心 GPU 从几百瓦到 700 W 以上。对云用户，功耗体现在价格里；对自建机房，供电和散热会限制每个机柜能放多少卡；对端侧设备，功耗是首要约束。

## 6. 选型示例（书中）

| 场景 | 选择 | 理由 |
|---|---|---|
| Llama-3-8B 微调模型，延迟要求不高 | A10 | 24 GB 足够装下 16 GB 权重，价格低、货源多 |
| DeepSeek-R1-Distill-Qwen-14B | L40S | 48 GB 留出较多 KV Cache 空间；支持 FP8；避免多卡开销 |
| DeepSeek-R1（671B MoE） | 8 × H200 + NVSwitch，FP8 | 单卡放不下，需要高带宽机内互联 |

## 7. 存储墙

书中引用的数据显示，过去 20 年硬件峰值算力的增速远快于显存带宽和互联带宽。这个差距被称为"存储墙"（memory wall）。硬件上的两种应对方向：一是增加片上 SRAM，把数据留在计算单元附近；二是把多张 GPU 和 CPU 用高带宽互联紧耦合成一个整体（如 NVL72）。软件上的应对方法，就是本笔记大部分章节的内容：减少数据搬运。

## 动手实验

**实验 1：查看本机 GPU 与拓扑。**

```bash
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
nvidia-smi topo -m      # 看 GPU 之间是 NV#（NVLink）还是 PIX/PHB/SYS（PCIe）
```

**实验 2：实测矩阵乘算力与显存带宽，和规格表对比。**

```python
# stage0_bench.py
import torch, time

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters

dt = torch.bfloat16
n = 8192
a, b = torch.randn(n, n, device="cuda", dtype=dt), torch.randn(n, n, device="cuda", dtype=dt)
t = bench(lambda: a @ b)
print(f"matmul {n}^3: {2 * n**3 / t / 1e12:.1f} TFLOPS")

x = torch.empty(2 * 1024**3 // 2, device="cuda", dtype=dt)   # 2 GiB
y = torch.empty_like(x)
t = bench(lambda: y.copy_(x))
print(f"copy: {2 * x.numel() * 2 / t / 1e12:.2f} TB/s")      # 读 + 写
```

把实测值填进下表。通常矩阵乘能达到峰值的 70–85%，拷贝带宽能达到 80–90%。

| 项 | 规格值 | 实测值 | 比例 |
|---|---|---|---|
| BF16 稠密算力 | | | |
| 显存带宽 | | | |

## 自测题

1. 规格表上 H100 SXM 的 BF16 算力写着 1979 TFLOPS，用它估算 prefill 时间会有什么问题？
2. 一个 decode 为主的聊天服务，在 H100 SXM 和 H100 NVL 之间怎么选？
3. 为什么 8 卡 SXM 机器加装 NVSwitch 后，任意两卡之间的带宽从约 128 GB/s 提升到 900 GB/s？
4. A800 和 A100 在单卡推理上有区别吗？多卡呢？

<details>
<summary>参考答案</summary>

1. 1979 是 2:4 稀疏下的数值，稠密矩阵乘的峰值约 989 TFLOPS。直接用会把 prefill 时间低估一半。
2. decode 阶段受显存带宽限制（见 0.4），NVL 的带宽（3.9 TB/s）高于 SXM（3.35 TB/s），显存也更大，单卡 decode 更有优势；但如果需要 2 卡以上张量并行，SXM 的 NVSwitch 全互联更好。
3. 没有 NVSwitch 时，每张卡的 NVLink 总带宽要分成 7 条点对点链路；NVSwitch 作为交换结构，让每张卡都能用满自己的全部 NVLink 带宽与任意其他卡通信。
4. 单卡算力、显存、带宽相同，没有区别。多卡张量并行时 A800 的 NVLink 带宽更低（400 vs 600 GB/s），all-reduce 通信更慢，卡数越多、批越大，影响越明显。

</details>

## 延伸阅读

- NVIDIA 各型号数据手册（注意区分 sparse 与 dense 栏目）
- Gholami et al., *AI and Memory Wall*, IEEE Micro 2024（存储墙问题的系统讨论）
