# 4.7 Multi-LoRA Serving

> 对应原书第 10 章 *Multi-LoRA Serving*。Punica、S-LoRA 的机制和 vLLM 的参数是书外内容。

企业使用 LLM 通常分两步：先用 RAG 把自己的知识接入模型，再用微调让模型的回答更准确、更贴合自己的数据和风格。微调最常用的方法是 **LoRA**：不改动原模型的权重，只为部分线性层训练两个很小的低秩矩阵。

一个平台上往往会有很多个这样的微调版本：每个客户一个、每个任务一个。如果每个版本都部署成独立的模型，GPU 的数量会随着版本数线性增长，而大部分版本的流量都很小。Multi-LoRA serving 的做法是：只部署一份基座模型，把很多个 LoRA 适配器一起放在显存里，不同的请求使用不同的适配器，但在同一个 batch 中一起计算。

## 1. LoRA 回顾

LoRA 给一个线性层 $W$（$d_{out} \times d_{in}$）加上一个低秩的修正：

$$
y = x W^\top + \frac{\alpha}{r} \cdot x A^\top B^\top
$$

$A$ 是 $r \times d_{in}$，$B$ 是 $d_{out} \times r$，秩 $r$ 通常是 8 到 64。训练时 $W$ 不变，只训练 $A$ 和 $B$。

适配器很小。下面的组件按各模型的实际维度计算一个适配器的大小，以及它带来的额外计算量：

<div class="llm-widget" data-widget="lora-cost"></div>

以 Qwen3-8B、秩 16、所有线性层为例，一个适配器约 83 MiB，8 GB 显存就能放下近百个；每个 token 增加的计算量不到 1%。

## 2. 难点在批处理

如果一个 batch 中所有请求都用同一个适配器，最简单的做法是把 $BA$ 加到 $W$ 上（合并权重），推理时没有任何额外开销。但 Multi-LoRA 的场景下，同一个 batch 中的请求用的是不同的适配器。基座部分 $xW^\top$ 所有请求共用，一次矩阵乘法即可；LoRA 部分每个请求要乘以自己的 $A$、$B$。

逐个请求计算会产生大量的小矩阵乘法，每个都是一次 kernel 调用，3.3 节说过，这类小 kernel 的开销主要在启动，而不在计算。`code/stage4/lora_batch.py` 在 CPU 上比较了几种算法：

```bash
python code/stage4/lora_batch.py
```

```text
batch 64，8 个适配器，秩 16
逐个请求与按适配器分组的最大差：2.38e-07；与批量矩阵乘法：0.00e+00
矩阵乘法调用次数：逐个请求 128 次；按适配器分组 16 次；批量矩阵乘法 2 次
CPU 耗时（仅供参考）：逐个 82.7 ms，分组 33.2 ms，批量 3.6 ms
LoRA 部分的计算量是基座矩阵乘法的 3.1%
```

原书提到的 **Punica** 专门为此设计了 kernel：SGMV（Segmented Gather Matrix-Vector multiplication）把 batch 中的请求按适配器分段，在一次 kernel 调用中完成所有段的计算；BGMV 则为每个请求收集它自己的适配器权重，适合 decode 时每个请求只有一个 token 的情况。论文报告，与当时最好的系统相比，同一组 GPU 上的吞吐提高了 12 倍。

## 3. 很多个适配器

适配器多到显存放不下时，就要像多模型服务（4.5 节）那样管理一个适配器的缓存：活跃的适配器放在显存中，不活跃的放在 CPU 内存或磁盘上，需要时再加载。原书也是这样描述的：所有活跃的适配器都已加载、随时可用，冷的适配器在需要时从 CPU 内存或磁盘加载。

适配器的加载比整个模型快得多（几十 MB 对几十 GB），但它和 KV Cache 争夺同一块显存。**S-LoRA** 提出用统一的分页方式管理两者：适配器的权重和 KV Cache 都以页为单位存放在同一个内存池中，减少碎片。论文中一张 GPU 可以服务几千个适配器，吞吐比当时的 vLLM 加 PEFT 的方案高约 4 倍。

## 4. 在 vLLM 中使用

```bash
vllm serve Qwen/Qwen3-8B --enable-lora \
  --lora-modules customer-a=/path/to/lora_a customer-b=/path/to/lora_b \
  --max-loras 8 --max-lora-rank 64 --max-cpu-loras 64
```

- `--max-loras`：同一个 batch 中最多同时使用几个不同的适配器，显存中为它们预留空间；
- `--max-lora-rank`：支持的最大秩，影响预留的空间和 kernel 的选择；
- `--max-cpu-loras`：CPU 内存中缓存的适配器数量。

客户端用适配器的名字作为模型名发送请求，就会使用对应的适配器：

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "customer-a", "messages": [{"role": "user", "content": "你好"}]}'
```

运行时动态加载新的适配器，需要设置环境变量 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`，然后调用 `/v1/load_lora_adapter` 接口。

## 5. 什么时候不该用

原书指出，Multi-LoRA 不是所有微调模型的默认选择：如果每个适配器自己的流量就大到需要多个副本，那么更好的做法是把适配器合并进基座模型，作为独立的模型分别部署。Multi-LoRA 适合的是**每个适配器的流量都不大**、单独部署无法用满一张 GPU 的情况。

原因在于 Multi-LoRA 的开销：LoRA 部分的计算虽然少，却是额外的 kernel；`--max-loras` 预留的空间减少了 KV Cache；不同适配器的请求也无法共享前缀缓存（缓存的键中包含了适配器）。流量大的时候，这些开销都不如合并权重来得划算。

## 6. 实验

- **CPU**：把 `lora_batch.py` 的 `n_adapters` 从 8 改成 64（每个请求几乎都用不同的适配器），三种算法的调用次数和耗时怎样变化？
- **组件**：在 `lora-cost` 中比较"所有线性层"和"只有注意力"两种设置的适配器大小。FFN 在参数中占了多少比例？
- **GPU**：用 Qwen3-8B 和 4 个随机初始化的 LoRA 适配器，分别以"所有请求使用同一个适配器"和"请求随机使用 4 个适配器"压测，与不加 LoRA 时比较吞吐。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "Qwen3-8B 加一个秩 16、作用于所有线性层的 LoRA 适配器，大约多大？", "options": ["约 8 MiB", "约 80 MiB", "约 800 MiB", "约 8 GiB"], "answer": 1, "explain": "每层 7 个线性层，每个增加 r × (d_in + d_out) 个参数，36 层合计约 4400 万参数，BF16 约 83 MiB。几 GB 的显存就能放下上百个适配器。"}, {"q": "Multi-LoRA 的主要技术难点是什么？", "options": ["适配器太大，放不下", "同一个 batch 中的请求使用不同的适配器，逐个计算会产生大量小矩阵乘法；需要 SGMV 这样的 kernel 一次处理多个适配器", "LoRA 的计算量太大", "适配器无法与 KV Cache 共存"], "answer": 1, "explain": "LoRA 部分的计算量只有基座的百分之几，但如果逐个请求调用，kernel 的启动开销会占主导。Punica 把按适配器分段的计算合并到一次 kernel 调用中。"}, {"q": "某个 LoRA 适配器的流量很大，需要好几个副本。原书建议怎样部署？", "options": ["继续用 Multi-LoRA，增加 max-loras", "把适配器合并进基座模型，作为独立的模型部署，用数据并行扩展", "改用全参数微调", "把它放到 CPU 上运行"], "answer": 1, "explain": "合并后没有任何额外的 kernel 和显存预留，前缀缓存也能正常共享。Multi-LoRA 适合每个适配器流量都不大的情况。"}]'></div>

## 延伸阅读

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022。
- Chen et al., [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547), MLSys 2024。
- Sheng et al., [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285), MLSys 2024。
- [vLLM 文档：LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora.html)。
