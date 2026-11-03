# LLM Serving 实战教程

<div class="home-hero">
  <div class="home-hero-copy">
    <h1 class="home-title">LLM Serving<br />实战教程</h1>
    <p class="home-lede">以《Hands-On LLM Serving and Optimization》为主线，从一次前向计算讲到一个能上线的推理服务。先算清显存与带宽这两笔账，再亲手写出连续批处理和分页 KV Cache，最后对照 2026 年的大规模生产系统。</p>
    <div class="home-actions">
      <a class="is-primary" data-resume href="#/stage0/01-serving-landscape">从 0.1 开始</a>
      <a href="#/guide/roadmap">路线与学习方法</a>
      <a href="#/stage1/project1-walkthrough">项目 1 分步指南</a>
    </div>
    <p class="home-scope">51 节正文，48 个交互组件，204 道自测题，4 个代码项目和 1 个毕业项目。只有 CPU 也能从头学到尾。</p>
  </div>
  <div class="llm-widget" data-widget="kv-pool"></div>
</div>

<div class="home-pillars">
  <div class="home-pillar">
    <div class="home-pillar-head">
      <span class="home-pillar-tag"><em>I.</em> 显存账本</span>
      <span class="home-pillar-eq">$m_{\text{KV}} = 2 L n_{kv} d_h B$</span>
    </div>
    <p>权重之外，KV Cache 随生成不断增长，决定了一张卡能同时服务多少请求。全书沿这条式子展开：<strong>量化</strong>压缩每个元素的字节数 $B$，<strong>GQA 与 MLA</strong> 压缩头数和维度，<strong>PagedAttention</strong> 消除预留造成的碎片，<strong>前缀缓存与分层卸载</strong>复用算过的 KV。</p>
  </div>
  <div class="home-pillar">
    <div class="home-pillar-head">
      <span class="home-pillar-tag"><em>II.</em> 算术强度与 Roofline</span>
      <span class="home-pillar-eq">$I = \text{FLOPs} / \text{Bytes}$</span>
    </div>
    <p>prefill 一次处理整段输入，受算力限制；decode 每步只出 1 个 token，却要读完全部权重和 KV，受带宽限制。<strong>连续批处理</strong>分摊读权重的开销，<strong>FlashAttention 与算子融合</strong>减少显存往返，<strong>投机解码</strong>一次验证多个 token，<strong>PD 分离与 AFD</strong> 把不同的瓶颈拆到不同的卡上。</p>
  </div>
</div>

<h2 class="home-section-title">每一节怎样学</h2>

<div class="home-facts">
  <div class="home-fact">
    <div class="home-fact-num">01</div>
    <h4>先手算</h4>
    <p>每个公式都代入 Qwen3-8B、DeepSeek-V3 和 A100、H100 的真实参数算一遍。原书有误的数字，在对应位置单独勘误。</p>
  </div>
  <div class="home-fact">
    <div class="home-fact-num">02</div>
    <h4>用组件验证</h4>
    <p>拖动参数或单步追踪：显存账本、Roofline 拐点、批处理时间线、分页块分配、GPTQ 误差补偿、集群路由，结果随手可见。</p>
  </div>
  <div class="home-fact">
    <div class="home-fact-num">03</div>
    <h4>做自测</h4>
    <p>每节 3～4 道题，错误选项针对常见误解设计。答题记录和阅读进度保存在本地浏览器，下次从上次停下的地方继续。</p>
  </div>
  <div class="home-fact">
    <div class="home-fact-num">04</div>
    <h4>写代码</h4>
    <p>每部分以一个代码项目收尾。核心代码和单元测试都能在 CPU 上完成，每一步有独立的测试和分级提示。</p>
  </div>
</div>

<h2 class="home-section-title">课程地图</h2>

<div class="llm-widget" data-widget="course-map"></div>

<h2 class="home-section-title">实战项目</h2>
<p class="home-section-note">先在 CPU 上跑通全部单元测试，弄清数据结构和边界条件；有 GPU 时再运行压测脚本，核对性能曲线。</p>

<div class="home-projects">
  <div class="home-proj-row home-proj-head">
    <span>项目</span>
    <span>要实现的内容</span>
    <span>代码与运行环境</span>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">项目 1</span>
      <a href="#/stage1/boss-miniserve">MiniServe 推理引擎</a>
      <a class="home-proj-sublink" href="#/stage1/project1-walkthrough">11 步分步指南</a>
    </div>
    <div class="home-proj-desc">约一千行 PyTorch 加 FastAPI 的完整推理服务：块分配器、连续批处理、chunked prefill、重算式抢占、展平输入与分页 GQA 注意力。贪心解码的输出与 Hugging Face 逐 token 一致。</div>
    <div class="home-proj-meta"><code>code/project1</code><span>CPU 测试，GPU 压测</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">项目 2</span>
      <a href="#/stage2/boss-quant-arena">Quant Arena 量化竞技场</a>
    </div>
    <div class="home-proj-desc">从零实现整数量化与 FP8、GPTQ 的逐列误差补偿、AWQ 的缩放搜索、SmoothQuant 与 KV 分轴量化，在一个结构与 Qwen3 相同的小模型上比较困惑度。</div>
    <div class="home-proj-meta"><code>code/project2</code><span>CPU</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">项目 3</span>
      <a href="#/stage3/boss-kernel-forge">Kernel Forge 算子与投机解码</a>
    </div>
    <div class="home-proj-desc">用 Triton 写融合 RMSNorm、SwiGLU、在线 softmax、带因果掩码和 GQA 的 FlashAttention 前向、分页 decode 注意力；再实现投机解码的拒绝采样与 n-gram 起草。</div>
    <div class="home-proj-meta"><code>code/project3</code><span>CPU 解释器，GPU 测速</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">项目 4</span>
      <a href="#/stage4/boss-cluster-commander">Cluster Commander 集群模拟</a>
    </div>
    <div class="home-proj-desc">一个推理集群的离散事件模拟器：引用计数加 LRU 的前缀缓存、四种路由策略、PD 分离与逐层 KV 传输，按 SLO 计算 goodput。</div>
    <div class="home-proj-meta"><code>code/project4</code><span>CPU，仅标准库</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">毕业项目</span>
      <a href="#/final/boss-graduation">为一个业务设计推理服务</a>
    </div>
    <div class="home-proj-desc">在知识库问答、长文档分析、Agent 编程助手中选一个场景，经过手算、模拟器搜索和实测验证，给出满足 TTFT、TPOT 目标且成本最低的部署方案。</div>
    <div class="home-proj-meta"><code>code/graduation</code><span>CPU 规划，GPU 验证</span></div>
  </div>
</div>

<div class="home-bottom-grid">
  <div class="home-bottom-col">
    <h2 class="home-section-title">开始之前</h2>
    <p>会写 Python 的类和函数，知道张量的形状和矩阵乘法就够了。异步编程、GPU 体系结构、分布式通信，用到时从头讲起。硬件按需准备，详见<a href="#/guide/lab-setup">实验环境</a>：</p>
    <ul class="home-spec-list">
      <li><strong>只有 CPU</strong>：全部交互组件、估算与模拟脚本，以及项目 1～4 的开发和测试。</li>
      <li><strong>一张 24～80 GB 的显卡</strong>：vLLM、SGLang 对照实验，真实模型的量化评测，Triton 算子的性能测试。按小时租用云 GPU 即可。</li>
      <li><strong>两个 Triton</strong>：原书中的 Triton 多指 NVIDIA Triton Inference Server，一个服务框架；第 3 部分写算子用的是 OpenAI 的 Triton 语言，两者无关。</li>
      <li><strong>书外内容</strong>：标注「书外」的章节和段落，依据论文、框架源码和官方文档整理。</li>
    </ul>
  </div>
  <div class="home-bottom-col">
    <h2 class="home-section-title">在本地运行</h2>
    <p>站点的前端依赖（Docsify、KaTeX、Mermaid、Prism）都在仓库里，不访问外部 CDN。</p>
    <pre><code class="lang-bash">git clone https://github.com/zrpxx/hands-on-llm-serving-opt.git
cd hands-on-llm-serving-opt
python3 -m http.server 3000    # 打开 http://localhost:3000
# 跑一个估算脚本和项目 1 参考实现的测试
python3 code/stage0/estimate_memory.py code/stage0/configs/qwen3-8b.json
python3 code/project1/tools/make_tiny_model.py --out /tmp/tiny
MINISERVE_IMPL=solution python3 -m pytest -q code/project1/tests</code></pre>
  </div>
</div>
