# LLM Serving 实战教程

<div class="home-hero">
  <div class="home-hero-copy">
    <div class="home-eyebrow">PART 00–06 · 50 节正文 · 33 个交互推演 · 5 个实战项目</div>
    <h1 class="home-title">LLM Serving<br />实战教程</h1>
    <p class="home-lede">以《Hands-On LLM Serving and Optimization》为主线，从一次前向计算讲到一个能上线的推理服务：先拆解自回归生成、显存账本与 Roofline 瓶颈，再手写连续批处理与分页 KV Cache，最后对照 2026 年的大规模生产系统。</p>
    <div class="home-actions">
      <a class="is-primary" data-resume href="#/stage0/01-serving-landscape">从 0.1 开始</a>
      <a href="#/guide/roadmap">阅读路线与方法</a>
      <a href="#/stage1/project1-walkthrough">项目 1 分步指南</a>
    </div>
  </div>
  <div class="llm-widget" data-widget="kv-pool"></div>
</div>

<div class="home-pillars">
  <div class="home-pillar">
    <div class="home-pillar-head">
      <span class="home-pillar-tag"><em>I.</em> 显存容量账本</span>
      <span class="home-pillar-eq">$m_{\text{KV}} = 2 L n_{kv} d_h B$</span>
    </div>
    <p>权重与随生成不断增长的 KV Cache 占满显存，直接决定单卡并发上限。全书沿着这条式子展开：用<strong>权重量化与 KV 量化</strong>压缩字节数 $B$，用 <strong>GQA 与 MLA</strong> 压缩头数与维度，用 <strong>PagedAttention</strong> 消除预留碎片，用<strong>前缀缓存与分层卸载</strong>复用历史计算。</p>
  </div>
  <div class="home-pillar">
    <div class="home-pillar-head">
      <span class="home-pillar-tag"><em>II.</em> 算术强度与 Roofline</span>
      <span class="home-pillar-eq">$I = \text{FLOPs} / \text{Bytes}$</span>
    </div>
    <p>prefill 算术强度高、受算力限制；decode 每步只处理 1 个 token 却要读完整权重与 KV，受显存带宽限制。围绕这一矛盾：用<strong>连续批处理</strong>分摊权重读取，用 <strong>FlashAttention 与算子融合</strong>减少显存往返，用<strong>投机解码</strong>一次前向验证多个 token，用 <strong>PD 分离与 AFD</strong> 隔离异构瓶颈。</p>
  </div>
</div>

<h2 class="home-section-title">学习体例</h2>

<div class="home-facts">
  <div class="home-fact">
    <div class="home-fact-num">01</div>
    <h4>先手算，再看公式</h4>
    <p>每个公式都代入 Qwen3-8B、DeepSeek-V3 与 A100、H100 的真实参数手算一遍；原书中存在疏漏的地方，在相应位置单独勘误。</p>
  </div>
  <div class="home-fact">
    <div class="home-fact-num">02</div>
    <h4>用组件验证直觉</h4>
    <p>正文内嵌 33 个交互组件，支持调参或单步追踪：显存账本、Roofline 拐点、批处理时间线、分页块分配、GPTQ 误差补偿与集群路由。</p>
  </div>
  <div class="home-fact">
    <div class="home-fact-num">03</div>
    <h4>每节一组诊断自测</h4>
    <p>共 204 道自测题，干扰项针对常见概念误区设计。作答记录与章节完成进度自动保存在本地浏览器，随时接着上次的位置继续。</p>
  </div>
  <div class="home-fact">
    <div class="home-fact-num">04</div>
    <h4>用代码项目收尾</h4>
    <p>五个实战项目贯穿全书。所有项目的核心代码开发与单元测试均可在纯 CPU 上完成，每一步配有独立测试与逐级提示。</p>
  </div>
</div>

<h2 class="home-section-title">课程地图</h2>

<div class="llm-widget" data-widget="course-map"></div>

<h2 class="home-section-title">实战项目</h2>
<p class="home-section-note">先在笔记本 CPU 上跑通全部单元测试，理解数据结构与边界条件；有 GPU 时再运行压测脚本核对性能曲线。</p>

<div class="home-projects">
  <div class="home-proj-row home-proj-head">
    <span>项目</span>
    <span>核心实现内容</span>
    <span>代码与验证</span>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">PROJECT 01</span>
      <a href="#/stage1/boss-miniserve">MiniServe 推理引擎</a>
      <a class="home-proj-sublink" href="#/stage1/project1-walkthrough">11 步分步指南 →</a>
    </div>
    <div class="home-proj-desc">用约一千行 PyTorch + FastAPI 实现完整推理服务：物理块分配器、连续批处理、Chunked Prefill、重算式抢占、展平输入与分页 GQA 注意力，贪心输出与 Hugging Face 逐 token 一致。</div>
    <div class="home-proj-meta"><code>code/project1</code><span>CPU 单测 · GPU 压测</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">PROJECT 02</span>
      <a href="#/stage2/boss-quant-arena">Quant Arena 量化竞技场</a>
    </div>
    <div class="home-proj-desc">从零实现对称/非对称整数量化、FP8 E4M3/E5M2、GPTQ 逐列 Cholesky 误差补偿、AWQ 网格搜索、SmoothQuant 激活平滑与 K/V 分轴量化，在真实结构小模型上评测困惑度。</div>
    <div class="home-proj-meta"><code>code/project2</code><span>CPU 单测与实验</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">PROJECT 03</span>
      <a href="#/stage3/boss-kernel-forge">Kernel Forge 算子与投机解码</a>
    </div>
    <div class="home-proj-desc">用 Triton 编写融合残差 RMSNorm、SwiGLU、在线 Softmax、带因果掩码与 GQA 的 FlashAttention 前向及分页 Decode 注意力；实现投机解码拒绝采样与 n-gram 起草。</div>
    <div class="home-proj-meta"><code>code/project3</code><span>CPU 解释器 · GPU 测速</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">PROJECT 04</span>
      <a href="#/stage4/boss-cluster-commander">Cluster Commander 集群模拟</a>
    </div>
    <div class="home-proj-desc">构建多实例离散事件模拟器：引用计数 + LRU 前缀缓存、轮询/最少负载/一致性哈希/缓存感知路由、Prefill-Decode 分离与逐层 KV 传输竞争，计算 SLO 约束下的 Goodput。</div>
    <div class="home-proj-meta"><code>code/project4</code><span>纯 Python 标准库</span></div>
  </div>
  <div class="home-proj-row">
    <div class="home-proj-name">
      <span class="home-proj-idx">GRADUATION</span>
      <a href="#/final/boss-graduation">毕业设计：容量规划与方案评审</a>
    </div>
    <div class="home-proj-desc">面向多租户知识库问答、长文档分析与多轮 Agent 三类真实业务场景，综合手算下限、集群模拟器搜索与实测验证，给出满足 TTFT/TPOT 目标且每百万 token 成本最低的架构方案。</div>
    <div class="home-proj-meta"><code>code/graduation</code><span>端到端容量规划</span></div>
  </div>
</div>

<div class="home-bottom-grid">
  <div class="home-bottom-col">
    <h2 class="home-section-title">前置要求与约定</h2>
    <p>会写 Python 类和函数，了解张量形状与矩阵乘法即可；异步编程、GPU 体系结构与分布式通信在用到时从头讲起。硬件可按需准备（详见<a href="#/guide/lab-setup">实验环境</a>）：</p>
    <ul class="home-spec-list">
      <li><strong>只有 CPU</strong>：全部 33 个交互组件、估算与模拟脚本，以及项目 1～4 的代码开发与正确性测试。</li>
      <li><strong>单张 24 GB～80 GB 显卡</strong>：运行 vLLM / SGLang 对照实验、真实模型量化评测与 Triton 算子性能剖析（云平台按小时租用即可）。</li>
      <li><strong>术语区分</strong>：原书中的 "Triton" 多数指 NVIDIA Triton Inference Server（服务框架）；第 3 部分用来写算子的是 OpenAI 的 Triton 语言，两者无关。标注「书外」的章节由论文与源码补充整理。</li>
    </ul>
  </div>
  <div class="home-bottom-col">
    <h2 class="home-section-title">在本地运行</h2>
    <p>站点的全部前端依赖（Docsify、KaTeX、Mermaid、Prism）均托管在仓库内，无需访问外部 CDN：</p>
    <pre><code class="lang-bash">git clone https://github.com/zrpxx/hands-on-llm-serving-opt.git
cd hands-on-llm-serving-opt
python3 -m http.server 3000      # 访问 http://localhost:3000
# 验证 CPU 估算脚本与项目 1 参考实现测试
python3 code/stage0/estimate_memory.py code/stage0/configs/qwen3-8b.json
python3 code/project1/tools/make_tiny_model.py --out /tmp/tiny
MINISERVE_IMPL=solution pytest -q code/project1/tests</code></pre>
  </div>
</div>
