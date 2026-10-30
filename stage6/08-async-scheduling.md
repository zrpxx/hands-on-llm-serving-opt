# 6.8 异步调度

> 书外内容。原书第 10 章介绍了 vLLM V1 把预处理放到单独进程的做法（5.4 节），3.6 节用性能剖析观察过 CPU 开销。本节讨论更进一步的异步调度。模拟实验和组件是书外内容。

3.6 节和项目 1 都观察到：模型越小、GPU 越快，decode 的每一步中 CPU 的开销占比越高。SGLang 的博客引用过一个估计：没有优化的推理引擎，可能有一半的时间花在 CPU 上。CUDA Graph 消除了 kernel 发射的开销（3.6 节），但每一步还有一段 CPU 工作无法放进 CUDA Graph：

- **调度**：选出这一步参与的请求，分配 KV 块，前缀缓存的匹配；
- **准备输入**：构造 `input_ids`、`positions`、block table 等张量，拷到 GPU；
- **后处理**：把采样结果拷回 CPU，检查停止条件，增量 detokenize，发给客户端。

同步执行时，这三段与 GPU 的前向依次进行，GPU 在 CPU 工作时空闲。

## 1. 提前一步调度

异步调度的思路很简单：GPU 在计算第 $N$ 步时，CPU 已经在准备第 $N+1$ 步，并处理第 $N-1$ 步的结果。这样 CPU 的工作就藏在了 GPU 计算的背后。

<div class="llm-widget" data-widget="async-timeline"></div>

稳定之后，每一步的耗时从"CPU + GPU"变成了二者中较大的那个。`code/stage6/async_sched.py` 用一个线程模拟 GPU（`sleep` 时释放 GIL，与真实的 GPU 一样可以和 CPU 并行），验证了这一点：

```bash
python code/stage6/async_sched.py
```

```text
每步：调度 2.5 ms + GPU 8.0 ms + 后处理 2.0 ms；64 个请求，batch 上限 32

方式      总耗时   GPU 利用率     每步   有效 token   多算的 token
同步      3.98 s        62%   13.2 ms        7253              0
异步      2.54 s        97%    8.4 ms        7253             64
```

## 2. 难点：下一步依赖这一步的结果

问题在于，第 $N+1$ 步的输入 token 正是第 $N$ 步采样出来的。CPU 准备第 $N+1$ 步时，第 $N$ 步还没算完。解决办法是把调度拆成两部分：

- **CPU 只决定结构**：哪些请求参与、各有几个 token、KV 写到哪些块。这些在采样之前就能确定。
- **token 的值留在 GPU 上**：第 $N$ 步采样的结果直接写进第 $N+1$ 步的输入张量，不经过 CPU。SGLang 称之为"未来 token"（future token）：CPU 用占位符代替还不知道的 token，GPU 在执行时再替换成真实的值。

由此带来两个代价：

- **多算一步**。第 $N$ 步结束的请求（生成了结束符或达到长度上限），调度器要到处理第 $N$ 步的结果时才知道，而第 $N+1$ 步已经把它排进去了。这个请求会多算一个 token，结果被丢弃。上面的模拟中，64 个请求各多算了一步。请求越长，这部分的占比越小。
- **与其他功能的组合更复杂**。投机解码中，接受多少个草稿决定了下一步的输入长度；结构化输出中，下一步的合法 token 掩码取决于这一步选出的 token；惩罚项依赖于已经生成的 token。每一种都要专门处理，才能不在 CPU 和 GPU 之间同步。

## 3. 各框架的实现

| 框架 | 做法 | 默认 |
|---|---|---|
| SGLang | 零开销的批调度器（2024 年 12 月的 v0.4）：调度器领先一个 batch，用未来 token 解决依赖 | 开启，`--disable-overlap-schedule` 关闭 |
| TensorRT-LLM | overlap scheduler：先发射第 $N+1$ 步的 GPU 计算，再处理第 $N$ 步的停止判断和响应 | 开启 |
| vLLM | 异步调度：调度和准备输入与 GPU 执行重叠；从 v0.14 起默认开启，`--no-async-scheduling` 关闭 | 开启 |
| DeepInfer | 异步的零开销批调度器 | 开启 |

vLLM 2026 年的 Model Runner V2 进一步把输入的准备搬到了 GPU 上：请求的状态大部分常驻显存，用 Triton kernel 直接在 GPU 上构造 `input_ids`、`positions` 等张量，采样结果用单独的 CUDA 流异步拷回 CPU。在 GB200 上跑 Qwen3-0.6B（小模型，CPU 开销占比大），输出吞吐从每秒约 1.6 万提高到 2.5 万个 token。写作时它仍需通过 `VLLM_USE_V2_MODEL_RUNNER=1` 开启。

## 4. 异步也藏不住的情况

异步调度的上限是 GPU 的计算时间。如果 CPU 每步的工作比 GPU 还长，每一步就由 CPU 决定（在组件中把 GPU 前向调到 2 ms 试试）。这在以下情况中很常见：小模型、很大的 batch、多卡 TP（每张卡上的计算变少了，CPU 的工作却没有变少）、大规模的 DP（每步还要在各副本之间同步）。这时要减少 CPU 工作本身：

- 把输入的准备搬到 GPU 上（Model Runner V2）；
- 用更高效的数据结构，避免每步重建 Python 对象；
- 把 detokenize 和网络发送放到另一个进程（vLLM V1 的进程分离，5.4 节）；
- 减少多卡之间的同步。vLLM 在 DeepSeek-V4.1-Flash 的部署中，把每步 DP 同步用的 CPU all-reduce 从 10.8 ms 降到 1.5 ms，每步耗时从 16.1 ms 降到 13.2 ms。

## 5. 实验

- **CPU**：运行 `async_sched.py --gpu-ms 2`，异步还有多少收益？再运行 `--requests 64 --max-batch 64`，多算的 token 占比是多少？
- **CPU**：在项目 1 的 MiniServe 中，用 `time.perf_counter` 测出一个 decode 步中调度、准备输入、前向、后处理各自的耗时。哪些部分可以与前向重叠？
- **GPU**：用 vLLM 部署 Qwen3-0.6B，在并发 64 下比较开关 `--no-async-scheduling` 的 TPOT 和吞吐；再用 Nsight Systems 观察两种情况下 GPU 时间线上的空隙（3.6 节）。

## 自测

<!--QUIZ-->

## 延伸阅读

- SGLang 团队，[SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4)，2024。
- [TensorRT-LLM 文档：Overlap Scheduler](https://nvidia.github.io/TensorRT-LLM/latest/features/overlap-scheduler.html)。
- vLLM 团队，[Model Runner V2: A Modular and Faster Core for vLLM](https://vllm.ai/blog/2026-03-24-mrv2)，2026。
