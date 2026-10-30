# 项目 4　Cluster Commander：模拟一个推理集群

> 前置：4.3、4.6 节，以及 1.2、1.4 节。代码在 [`code/project4`](https://github.com/zrpxx/hands-on-llm-serving-opt/tree/main/code/project4)。只依赖 Python 标准库和 pytest，全部在 CPU 上运行，每组实验几秒钟。GPU 上的验证实验需要 2 张以上的卡。

第 4 部分的很多结论，比如"缓存感知路由能降低 TTFT""PD 分离的配比很关键"，要在真实的集群上验证，需要好几张 GPU 和大量的调试。这个项目换一种方式：写一个推理集群的**离散事件模拟器**。每个实例按连续批处理和 chunked prefill 运行，有自己的前缀缓存，每一步的耗时用 Roofline 估算。在它上面实现几种路由策略和 PD 分离，几秒钟就能比较它们在不同负载下的表现。

模拟器不能代替真实的测量，但它能让你在动手部署之前，先对"什么配置值得试"建立直觉，也能帮助你理解真实系统中观察到的现象。

## 1. 代码结构

```text
code/project4/
├── starter/cluster/         要完成的代码，5 步
├── solution/cluster/        参考实现
├── run.py                   实验：routers、pd、ratio
└── tests/test_steps.py      分步测试
```

| 文件 | 内容 | 是否要写 |
|---|---|---|
| `costmodel.py` | 硬件、模型的参数；一步的耗时 = max(读权重, 计算) + 读 KV + 固定开销 | 已写好 |
| `workload.py` | 生成请求：多租户带共享前缀的负载、长输入的负载 | 已写好 |
| `instance.py` | 一个实例：连续批处理、chunked prefill、按前缀缓存跳过已算的部分；也可以只做 prefill 或只做 decode | 已写好 |
| `sim.py` | 事件循环和统计；`goodput` | 第 5 步 |
| `prefix_cache.py` | 按块的前缀缓存，引用计数加 LRU 淘汰 | 第 1 步 |
| `router.py` | 轮询、最少负载、一致性哈希、缓存感知 | 第 2、3 步 |
| `pd.py` | KV 传输时间、PD 分离的集群 | 第 4 步 |

建议先读一遍 `instance.py` 的 `start_step` 和 `finish_step`，它们与项目 1 的调度器是同一个思路，只是用估算的耗时代替了真实的前向计算。

## 2. 准备

```bash
cd code/project4
pip install -r requirements.txt                    # 只有 pytest
CLUSTER_IMPL=solution python -m pytest -q tests    # 参考实现，应全部通过，不到 1 秒
python -m pytest -q tests                          # 你的实现，现在全部失败
CLUSTER_IMPL=solution python run.py routers        # 看一眼实验的输出
```

## 3. 分步完成

| 步骤 | 内容 | 验证 |
|---|---|---|
| 第 1 步 | 前缀缓存 | `python -m pytest tests/test_steps.py -k step01` |
| 第 2 步 | 轮询与最少负载 | `-k step02` |
| 第 3 步 | 一致性哈希与缓存感知路由 | `-k step03`，然后 `python run.py routers` |
| 第 4 步 | PD 分离 | `-k step04`，然后 `python run.py pd` |
| 第 5 步 | goodput 与端到端 | `-k step05`，然后 `python run.py ratio` |

### 第 1 步：前缀缓存

`PrefixCache` 用两个结构管理缓存的块：`ref` 记录每个块的引用计数，`free` 是一个 `OrderedDict`，只放引用计数为 0 的块，按最近使用的顺序排列。这和 vLLM 的做法相同：淘汰时从 `free` 的开头取，是 O(1) 的；正在被请求使用的块不在 `free` 里，不会被淘汰。

要实现四个方法：`match`（从头连续匹配）、`acquire`（请求开始使用这些块）、`release`（请求结束）、`evict`（腾出空间）。

<details>
<summary>提示：match 中为什么只移动空闲的块</summary>

正在使用的块不在 `free` 里，它的"最近使用"时间在它被释放、放进 `free` 末尾的那一刻才有意义。空闲的块被命中，说明它马上又要被用，移到末尾以免被优先淘汰。
</details>

测试中的第二个用例值得手算一遍：容量为 4，`a`、`b` 空闲，`c`、`d` 在使用，访问 `a` 之后再插入 `e`，应该淘汰 `b`；再插入两个块时，只有 `a` 能淘汰，缓存会暂时超过容量。

### 第 2 步：轮询与最少负载

两个最简单的路由器。最少负载用 `queued_tokens()` 衡量：还没算的 prefill token 加上还要生成的 token。测试构造了一个"请求数最少、但工作量最大"的实例，检查你没有按请求个数选择。

### 第 3 步：一致性哈希与缓存感知

**一致性哈希。** 构造函数已经把每个实例的 64 个虚拟节点放到了哈希环上。`lookup` 要在环上找到第一个哈希值大于 key 的虚拟节点，越过末尾时绕回开头。

<details>
<summary>提示：用 bisect</summary>

`self.points` 是排好序的哈希值列表。`bisect.bisect(self.points, h)` 返回第一个大于 `h` 的位置，对环的长度取模就处理了绕回。
</details>

测试检查三件事：同一个 key 总是映射到同一个实例；2000 个 key 在 4 个实例上的分布大致均匀；增加第 5 个实例后，只有约五分之一的 key 换了位置。最后一点是一致性哈希相对于"哈希值对实例数取模"的优势：后者增加一个实例，几乎所有的 key 都要换位置，所有缓存都会失效。

**缓存感知。** 按 4.6 节描述的 SGLang 路由器的思路：负载明显失衡时按负载选；否则选命中最多的实例；哪里都命中不了多少时也按负载选。

完成之后运行：

```bash
python run.py routers
```

应该得到与 4.6 节第 3 节相近的结果：两种按前缀的路由命中率约 0.83，轮询约 0.58；缓存感知的 TTFT P99 最低。

### 第 4 步：PD 分离

`kv_transfer_time` 计算一个请求的 KV 从 prefill 实例传到 decode 实例需要多久。`on_prefill_done` 在 prefill 实例算完一个请求时被调用，要为它安排传输，并在传输完成时把它交给一个 decode 实例。

这里模拟了链路的竞争：每个 prefill 实例的出口链路同一时间只能传一个请求的 KV。逐层传输时，大部分数据在 prefill 计算期间已经传完，但如果链路还被前一个请求占着，就要等。

<details>
<summary>提示：结束时间的两种情况</summary>

逐层传输：`max(now + full / 层数, 链路空闲时间 + full)`。前一项是"链路空闲时，只剩最后一层要传"；后一项是"链路被占用，整个传输要排在前一个后面"。整体传输：`max(now, 链路空闲时间) + full`。
</details>

测试检查传输时间的计算，以及 2P2D 的集群能完成所有请求、每个请求最后都在 decode 实例上完成、ITL P99 不到聚合部署的一半。

### 第 5 步：goodput

`goodput(result, ttft_slo, tpot_slo)` 返回同时满足 TTFT 和 TPOT 目标的请求比例，没有完成的请求算作不达标。最后一个测试在 128 个租户的负载上比较缓存感知与轮询，要求命中率至少高 0.1。

## 4. 实验与报告

完成全部步骤后，做下面的实验，写一份简短的报告。

**实验一：路由策略。**

```bash
python run.py routers --tenants 16
python run.py routers --tenants 128
python run.py routers --tenants 512
```

租户数变化时，四种策略的命中率和 TTFT 怎样变化？缓存感知路由在什么条件下优势最大？用"每个实例能缓存多少个前缀"解释。

**实验二：PD 分离的配比。**

```bash
python run.py ratio
```

在每秒 2、4、6、8 个请求下，哪种配置的 goodput 最高？请求速率升高时，最优的 P:D 比例怎样变化？

**实验三：网络。**

```bash
python run.py pd --link 5 --no-layerwise
python run.py pd --link 5
```

解释"第 2 个 token"一列的变化。哪种配置对网络最敏感，为什么？

**实验四（自选）：** 修改代码回答一个你自己的问题。比如：

- 让 decode 实例使用 L40S（`costmodel.py` 中已定义），PD 分离还划算吗？
- 给 `CacheAware` 加一个参数，用 `num_requests()` 代替 `queued_tokens()`，结果有什么不同？
- 实现一个简单的自动扩缩容：每隔 10 秒检查各实例的排队 token 数，超过阈值就增加一个实例（新实例要等 30 秒才能开始服务，模拟冷启动）。

## 5. GPU 上的验证（可选）

有 2 张以上 GPU 时，选一个结论在真实系统上验证：

- **缓存感知路由**：启动两个 SGLang 实例，分别用 `sglang_router` 的 `round_robin` 和 `cache_aware` 策略，发送带共享前缀的多租户负载（可以用 `workload.py` 生成请求的长度和前缀，用 Qwen3-0.6B 或 Qwen3-8B），比较命中率和 TTFT。
- **PD 分离**：按 vLLM 或 SGLang 文档中的示例，在两张卡上部署 1P1D，与两个聚合实例对比 ITL P99。

把实测的结果与模拟器的预测放在一起，分析差异的来源：模拟器忽略了什么？

## 6. 完成标准

- `python -m pytest -q tests` 全部通过。
- 实验一到三的结果和分析，以及实验四的问题、改动和结论。
- 有 GPU 时：一组真实系统上的验证结果，以及与模拟器的对比。

## 7. 扩展练习

- **更准确的耗时模型。** 现在注意力只计入了读 KV 的时间。加上 prefill 时注意力的计算量（与长度的平方成正比），看长输入负载下 TTFT 的变化。
- **基于 SLO 的路由。** 请求带上自己的 TTFT 目标，路由器估算每个实例上的排队时间，选一个能满足目标、且负载最高的实例（把余量留给后面的请求）。
- **KV 卸载。** 给 `PrefixCache` 加一层 CPU 缓存：从显存淘汰的块进入 CPU 层，命中 CPU 层时按 4.4 节的带宽加上加载时间。

## 自测

<div class="llm-widget" data-widget="quiz" data-questions='[{"q": "前缀缓存为什么要把引用计数为 0 的块单独放在一个按最近使用排序的结构里？", "options": ["为了节省内存", "淘汰时直接从这个结构的开头取，是 O(1) 的；正在使用的块不在其中，不会被误淘汰", "为了支持多线程", "为了计算命中率"], "answer": 1, "explain": "如果每次淘汰都扫描所有块、跳过正在使用的，缓存满的时候每插入一块就是 O(容量)。项目中参考实现最初就是这样写的，模拟慢了几十倍。"}, {"q": "用「哈希值对实例数取模」做前缀路由，增加一个实例时会发生什么？一致性哈希怎样避免？", "options": ["没有影响", "几乎所有前缀都会换到另一个实例，所有缓存失效；一致性哈希只让约 1/N 的前缀换位置", "所有请求都会发到新实例", "新实例收不到任何请求"], "answer": 1, "explain": "取模的结果随 N 变化而整体改变。一致性哈希把实例和前缀都映射到一个环上，增加一个实例只影响环上它所在的那一段。"}, {"q": "3P1D 配置的 TTFT 很好，但第二个 token 的 P99 超过一秒。最可能的原因是：", "options": ["KV 传输的链路太慢", "只有一个 decode 实例，它的 KV 空间放不下那么多正在生成的请求，传过来的请求要排队等空间", "prefill 实例太多", "路由器选错了实例"], "answer": 1, "explain": "在 50 GB/s 的链路、逐层传输时，传输只暴露不到 1 毫秒，瓶颈在 decode 侧的容量。PD 分离的配比要同时满足两边的需求。"}]'></div>
