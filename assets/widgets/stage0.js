/**
 * Stage 0 Interactive Widgets:
 * 1. serving-arch-selector (0.1 模型服务概览)
 * 2. gpu-specs-explorer (0.2 GPU 规格解读)
 * 3. kv-memory-calculator (0.3 显存估算：权重与 KV Cache)
 * 4. roofline-explorer (0.4 算术强度与 Roofline 分析)
 * 5. metrics-goodput-sim (0.5 性能指标：TTFT、TPOT、吞吐与 Goodput)
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  // =========================================================================
  // 1. 0.1 Serving Architecture & Routing Selector
  // =========================================================================
  W.register('serving-arch-selector', function (container) {
    const scenarios = [
      {
        id: 'chatbot-prod',
        label: '场景 A：高并发在线对话助手（如企业客服）',
        traffic: '高并发（50~200 QPS），请求长度差异大（100 ~ 16K token）',
        models: '1 个主力模型（如 Qwen3-32B）',
        privacy: '云端或专有云部署，要求 TTFT < 1s',
        arch: '单模型专用服务（Single-Model Serving）+ 多副本横向扩展',
        routing: '前缀缓存感知路由（Prefix/KV-aware Routing）+ 动态负载均衡',
        whyArch: '流量集中在一个主力模型上，独占 GPU 容器可避免多模型争抢显存，且能把剩余显存全部留给 KV Cache 支撑高并发。',
        whyRoute: '不能用简单轮询（Round-Robin）。因为 16K 长文档请求和 50 字短闲聊耗时相差百倍；同时多轮对话包含相同历史前缀，路由到同一实例可直接省去重复 Prefill。',
        diagram: ['客户端流量', 'KV/负载感知网关', 'vLLM 副本 1 (独占 GPU)', 'vLLM 副本 2 (独占 GPU)'],
      },
      {
        id: 'multi-custom',
        label: '场景 B：SaaS 平台托管 500 个客户低频模型',
        traffic: '单个模型流量极低（平均几分钟 1 次），但模型总数多',
        models: '500 个不同微调权重（每个日均调用 < 100 次）',
        privacy: '多租户共享基础设施，控制空置成本是第一要务',
        arch: '多模型共享服务（Multi-Model Serving）或 Multi-LoRA 服务',
        routing: '模型亲和路由（Model-Map Router）+ LRU 按需加载/卸载',
        whyArch: '如果按单模型形态给每个模型配一张卡，需要 500 张 GPU，99% 时间在空转烧钱。多模型共享 GPU 内存池（或共享同一基座动态挂载 LoRA 适配器）可将成本降低一个数量级。',
        whyRoute: '网关必须维护「模型 ID → 已加载该模型的 GPU 实例」映射表；若目标模型未加载，则挑显存最空闲的实例触发按需加载（或 LRU 淘汰旧模型）。',
        diagram: ['多租户请求', '模型路由表 & 装箱调度', '共享 GPU 实例（按需加载 / Multi-LoRA）'],
      },
      {
        id: 'edge-offline',
        label: '场景 C：手机 / 车载 / 笔记本本地隐私助手',
        traffic: '单用户（并发 = 1），断网可用，零云端推理费用',
        models: '1 个轻量级量化模型（如 Qwen3-4B-INT4）',
        privacy: '数据绝不出端设备，功耗与内存严格受限',
        arch: '端侧本地服务（On-Device Serving，如 llama.cpp / Core ML / LiteRT）',
        routing: '本地进程直连（无需网络负载均衡）',
        whyArch: '完全省去服务器成本与网络传输延迟，保护用户隐私；重点在于 INT4 低比特量化减少内存占用与适配端侧芯片（Apple Silicon / NPU）。',
        whyRoute: '并发永远是 1，不需要复杂的 Continuous Batching 调度器，优化重心是单流 Decode 带宽利用率和低功耗。',
        diagram: ['用户本地 App', '进程内 C++/Metal 运行时 (llama.cpp)', '统一内存 (RAM/VRAM) INT4 权重'],
      },
      {
        id: 'agent-platform',
        label: '场景 D：多部门共享的复杂 Agent 工作流平台',
        traffic: '单次用户提问触发 5~20 次内部模型调用（改写→检索→重排→生成→审查）',
        models: 'Embedding 模型 + Reranker + 小模型路由 + 70B 推理大模型协同',
        privacy: '多业务线共用集群，需按部门隔离配额与优先级',
        arch: '多模型编排服务平台（Serving Platform，如 Ray Serve / Kubernetes + 网关）',
        routing: 'DAG 图编排调度 + 业务配额隔离 + 模型级缓存感知路由',
        whyArch: '不仅要管单个模型，还要把多个异构模型串成一条有向无环图（DAG）流水线，并按部门划分资源组（Resource Group），防止某个离线任务冲垮线上客服。',
        whyRoute: '顶层按业务配额与 DAG 拓扑分发，底层针对生成式大模型采用 KV 缓存感知路由，针对 Embedding 采用动态批处理。',
        diagram: ['Agent 复合请求', '平台网关 & 配额隔离', 'DAG 编排引擎 (Ray)', 'Embedding 池 + LLM 生成池'],
      },
    ];

    let currentIdx = 0;

    function render() {
      const s = scenarios[currentIdx];
      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <h4 class="lw-title">服务形态与路由策略决策组件</h4>
            </div>
          </div>
          <p class="lw-subtitle">点击切换下方四个典型业务场景，观察约束条件如何决定服务架构与负载均衡策略：</p>
          <div class="lw-tabs">
            ${scenarios
              .map(
                (item, idx) =>
                  `<button type="button" class="lw-tab ${idx === currentIdx ? 'active' : ''}" data-idx="${idx}">${item.label}</button>`
              )
              .join('')}
          </div>
          <div class="lw-stats">
            <div class="lw-stat info">
              <div class="lw-stat-label">流量与并发特征</div>
              <div class="lw-stat-value" style="font-size:13px">${s.traffic}</div>
            </div>
            <div class="lw-stat info">
              <div class="lw-stat-label">模型规模与数量</div>
              <div class="lw-stat-value" style="font-size:13px">${s.models}</div>
            </div>
            <div class="lw-stat info">
              <div class="lw-stat-label">部署与成本约束</div>
              <div class="lw-stat-value" style="font-size:13px">${s.privacy}</div>
            </div>
          </div>
          <div style="background:var(--paper-2);border:1px solid var(--rule);border-radius:8px;padding:14px;margin:12px 0;">
            <div style="font-size:12px;color:var(--ink-3);margin-bottom:8px;font-weight:600;">推荐请求链路架构：</div>
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
              ${s.diagram
                .map(
                  (node, i) => `
                  <div style="background:var(--prefill-soft);color:var(--prefill-ink);border:1px solid var(--prefill);padding:5px 12px;border-radius:4px;font-size:12.5px;font-weight:600;">${node}</div>
                  ${i < s.diagram.length - 1 ? '<span style="color:var(--ink-3);font-weight:700;">→</span>' : ''}
                `
                )
                .join('')}
            </div>
          </div>
          <div class="lw-stats">
            <div class="lw-stat ok">
              <div class="lw-stat-label">推荐服务架构</div>
              <div class="lw-stat-value" style="font-size:14px;color:var(--ok);">${s.arch}</div>
              <div class="lw-stat-sub" style="margin-top:6px;line-height:1.5;">${s.whyArch}</div>
            </div>
            <div class="lw-stat warn">
              <div class="lw-stat-label">推荐路由与负载均衡策略</div>
              <div class="lw-stat-value" style="font-size:14px;color:var(--decode-ink);">${s.routing}</div>
              <div class="lw-stat-sub" style="margin-top:6px;line-height:1.5;">${s.whyRoute}</div>
            </div>
          </div>
        </div>
      `;

      container.querySelectorAll('.lw-tab').forEach((btn) => {
        btn.addEventListener('click', () => {
          currentIdx = Number(btn.getAttribute('data-idx'));
          render();
        });
      });
    }

    render();
  });

  // =========================================================================
  // 2. 0.2 GPU Specs & Interconnect Explorer
  // =========================================================================
  W.register('gpu-specs-explorer', function (container) {
    const gpus = [
      { id: 'h200', name: 'NVIDIA H200 SXM', vram: 141, bw: 4.8, bf16Dense: 989, fp8Dense: 1979, interconnect: 'NVLink + NVSwitch (900 GB/s)', linkBw: 900, arch: 'Hopper (SM90)' },
      { id: 'h100', name: 'NVIDIA H100 SXM', vram: 80, bw: 3.35, bf16Dense: 989, fp8Dense: 1979, interconnect: 'NVLink + NVSwitch (900 GB/s)', linkBw: 900, arch: 'Hopper (SM90)' },
      { id: 'h100nvl', name: 'NVIDIA H100 NVL', vram: 94, bw: 3.9, bf16Dense: 835, fp8Dense: 1671, interconnect: '双卡 NVLink Bridge (600 GB/s)', linkBw: 600, arch: 'Hopper (SM90)' },
      { id: 'a100', name: 'NVIDIA A100 80GB SXM', vram: 80, bw: 2.04, bf16Dense: 312, fp8Dense: 0, interconnect: 'NVLink + NVSwitch (600 GB/s)', linkBw: 600, arch: 'Ampere (SM80)' },
      { id: 'a800', name: 'NVIDIA A800 80GB SXM', vram: 80, bw: 2.04, bf16Dense: 312, fp8Dense: 0, interconnect: 'NVLink + NVSwitch (400 GB/s)', linkBw: 400, arch: 'Ampere (SM80)' },
      { id: 'l40s', name: 'NVIDIA L40S', vram: 48, bw: 0.864, bf16Dense: 362, fp8Dense: 733, interconnect: '仅 PCIe Gen4 (64 GB/s)', linkBw: 64, arch: 'Ada Lovelace (SM89)' },
      { id: 'a10', name: 'NVIDIA A10', vram: 24, bw: 0.6, bf16Dense: 125, fp8Dense: 0, interconnect: '仅 PCIe Gen4 (64 GB/s)', linkBw: 64, arch: 'Ampere (SM86)' },
    ];

    const state = {
      gpuA: 'h100',
      gpuB: 'a100',
      showSparseTrap: false,
      precision: 'bf16',
    };

    function getGpu(id) {
      return gpus.find((g) => g.id === id) || gpus[0];
    }

    function effectiveTflops(g) {
      const base = state.precision === 'fp8' ? (g.fp8Dense || g.bf16Dense) : g.bf16Dense;
      return state.showSparseTrap ? base * 2 : base;
    }

    function render() {
      const a = getGpu(state.gpuA);
      const b = getGpu(state.gpuB);
      const tflopsA = effectiveTflops(a);
      const tflopsB = effectiveTflops(b);
      const ridgeA = Math.round((a.bf16Dense * 1000) / (a.bw * 1000));
      const ridgeB = Math.round((b.bf16Dense * 1000) / (b.bw * 1000));

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <h4 class="lw-title">GPU 规格对比器与“稀疏算力陷阱”演示</h4>
            </div>
          </div>
          <div class="lw-controls">
            <div class="lw-control">
              <label>主选显卡 A</label>
              <select id="lw-gpu-a">
                ${gpus.map((g) => `<option value="${g.id}" ${g.id === state.gpuA ? 'selected' : ''}>${g.name}</option>`).join('')}
              </select>
            </div>
            <div class="lw-control">
              <label>对比显卡 B</label>
              <select id="lw-gpu-b">
                ${gpus.map((g) => `<option value="${g.id}" ${g.id === state.gpuB ? 'selected' : ''}>${g.name}</option>`).join('')}
              </select>
            </div>
            <div class="lw-control">
              <label>计算精度</label>
              <select id="lw-gpu-prec">
                <option value="bf16" ${state.precision === 'bf16' ? 'selected' : ''}>BF16 / FP16（主流基线）</option>
                <option value="fp8" ${state.precision === 'fp8' ? 'selected' : ''}>FP8 Tensor Core（Hopper/Ada）</option>
              </select>
            </div>
            <div class="lw-control">
              <label>算力口径切换</label>
              <button type="button" id="lw-sparse-toggle" class="lw-btn ${state.showSparseTrap ? 'active' : ''}">
                ${state.showSparseTrap ? '当前显示：宣传册 2:4 稀疏算力（虚高 2 倍）' : '当前显示：真实稠密（Dense）算力'}
              </button>
            </div>
          </div>

          ${
            state.showSparseTrap
              ? `<div class="lw-callout" style="border-left-color:var(--err);">
                  <strong>踩坑预警：</strong> NVIDIA 宣传页常标注 <strong>With Sparsity（2:4 结构化稀疏）</strong> 算力（如 H100 BF16 标为 1979 TFLOPS）。但绝大多数开源大模型权重都是<strong>稠密（Dense）</strong>的，实际可用峰值只有一半（989 TFLOPS）。用稀疏值估算 Prefill 时间会直接算快一倍。
                </div>`
              : ''
          }

          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:14px;">
            ${[
              { g: a, tf: tflopsA, ridge: ridgeA, color: W.color('prefill') },
              { g: b, tf: tflopsB, ridge: ridgeB, color: W.color('decode') },
            ]
              .map(
                ({ g, tf, ridge, color }) => `
              <div style="border:1px solid var(--rule);border-top:3px solid ${color};border-radius:8px;padding:14px;background:var(--paper-2);">
                <div style="font-weight:700;font-size:15px;margin-bottom:4px;">${g.name}</div>
                <div style="font-size:12px;color:var(--ink-3);margin-bottom:10px;">架构：${g.arch} ｜ 互联：${g.interconnect}</div>
                <div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;">
                    <span>显存容量（决定能不能装下 & 并发数）</span>
                    <strong>${g.vram} GB</strong>
                  </div>
                  <div style="height:8px;background:var(--rule);border-radius:4px;margin-top:3px;">
                    <div style="height:100%;width:${Math.min(100, (g.vram / 141) * 100)}%;background:${color};border-radius:4px;"></div>
                  </div>
                </div>
                <div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;">
                    <span>显存带宽（决定 Decode 吐字速度上限）</span>
                    <strong>${g.bw} TB/s</strong>
                  </div>
                  <div style="height:8px;background:var(--rule);border-radius:4px;margin-top:3px;">
                    <div style="height:100%;width:${Math.min(100, (g.bw / 4.8) * 100)}%;background:${color};border-radius:4px;"></div>
                  </div>
                </div>
                <div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;">
                    <span>${state.precision.toUpperCase()} 算力（决定 Prefill 速度上限）</span>
                    <strong>${state.precision === 'fp8' && g.fp8Dense === 0 ? '不支持原生 FP8 计算 (回退 BF16: ' + tf + ')' : tf + ' TFLOPS'}</strong>
                  </div>
                  <div style="height:8px;background:var(--rule);border-radius:4px;margin-top:3px;">
                    <div style="height:100%;width:${Math.min(100, (tf / 2000) * 100)}%;background:${color};border-radius:4px;"></div>
                  </div>
                </div>
                <div style="font-size:12px;color:var(--ink-2);margin-top:10px;padding-top:8px;border-top:1px dashed var(--rule-strong);">
                  16B 权重单流 Decode 理论最快一步：<strong>${((16.4 / (g.bw * 1000)) * 1000).toFixed(1)} ms</strong>（约 ${Math.round((g.bw * 1000) / 16.4)} tok/s）<br/>
                  BF16 算术强度拐点 $I^*$：<strong>${ridge} FLOPs/Byte</strong>
                </div>
              </div>
            `
              )
              .join('')}
          </div>
        </div>
      `;

      container.querySelector('#lw-gpu-a').addEventListener('change', (e) => {
        state.gpuA = e.target.value;
        render();
      });
      container.querySelector('#lw-gpu-b').addEventListener('change', (e) => {
        state.gpuB = e.target.value;
        render();
      });
      container.querySelector('#lw-gpu-prec').addEventListener('change', (e) => {
        state.precision = e.target.value;
        render();
      });
      container.querySelector('#lw-sparse-toggle').addEventListener('click', () => {
        state.showSparseTrap = !state.showSparseTrap;
        render();
      });
    }

    render();
  });

  // =========================================================================
  // 3. 0.3 KV Cache & Memory Budget Calculator
  // =========================================================================
  W.register('kv-memory-calculator', function (container) {
    const presets = {
      'qwen3-8b': { name: 'Qwen3-8B (GQA)', paramsB: 8.19, layers: 36, qHeads: 32, kvHeads: 8, headDim: 128 },
      'llama2-7b': { name: 'Llama-2-7B (MHA)', paramsB: 6.74, layers: 32, qHeads: 32, kvHeads: 32, headDim: 128 },
      'qwen3-0.6b': { name: 'Qwen3-0.6B (GQA)', paramsB: 0.6, layers: 28, qHeads: 16, kvHeads: 8, headDim: 128 },
      'qwen3-32b': { name: 'Qwen3-32B (GQA)', paramsB: 32.8, layers: 64, qHeads: 64, kvHeads: 8, headDim: 128 },
    };

    const state = {
      preset: 'qwen3-8b',
      paramsB: 8.19,
      layers: 36,
      qHeads: 32,
      kvHeads: 8,
      headDim: 128,
      weightBytes: 2, // BF16=2, FP8=1, INT4=0.5
      kvBytes: 2,     // BF16=2, FP8=1
      gpuMem: 80,
      util: 0.9,
      overhead: 2.0,
      seqLen: 4096,
      tp: 1,
    };

    function applyPreset(key) {
      const p = presets[key];
      if (!p) return;
      state.preset = key;
      state.paramsB = p.paramsB;
      state.layers = p.layers;
      state.qHeads = p.qHeads;
      state.kvHeads = p.kvHeads;
      state.headDim = p.headDim;
    }

    function render() {
      const weightGB = (state.paramsB * state.weightBytes) / state.tp;
      // KV per token per GPU in bytes
      const kvPerTokenBytes = (2 * state.layers * Math.max(1, Math.floor(state.kvHeads / state.tp)) * state.headDim * state.kvBytes);
      const kvPerTokenKiB = kvPerTokenBytes / 1024;
      const usableMemGB = state.gpuMem * state.util;
      const kvBudgetGB = Math.max(0, usableMemGB - weightGB - state.overhead);
      const maxTokens = kvBudgetGB > 0 ? Math.floor((kvBudgetGB * 1e9) / kvPerTokenBytes) : 0;
      const maxConcurrency = maxTokens / state.seqLen;
      const unusedGB = Math.max(0, state.gpuMem - usableMemGB);
      const isOOM = weightGB + state.overhead > usableMemGB;

      const wPct = Math.min(100, (weightGB / state.gpuMem) * 100);
      const oPct = Math.min(100 - wPct, (state.overhead / state.gpuMem) * 100);
      const kvPct = Math.min(100 - wPct - oPct, (kvBudgetGB / state.gpuMem) * 100);
      const uPct = Math.max(0, 100 - wPct - oPct - kvPct);

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <h4 class="lw-title">显存账本与并发容量计算器（模拟 vLLM 启动测算）</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${Object.entries(presets)
              .map(
                ([k, v]) =>
                  `<button type="button" class="lw-tab ${state.preset === k ? 'active' : ''}" data-preset="${k}">${v.name}</button>`
              )
              .join('')}
          </div>
          <div class="lw-controls">
            <div class="lw-control">
              <label>单卡显存容量 <span class="lw-val">${state.gpuMem} GB</span></label>
              <select id="lw-mem-gpu">
                <option value="24" ${state.gpuMem === 24 ? 'selected' : ''}>24 GB (RTX 4090 / A10)</option>
                <option value="48" ${state.gpuMem === 48 ? 'selected' : ''}>48 GB (L40S / A40)</option>
                <option value="80" ${state.gpuMem === 80 ? 'selected' : ''}>80 GB (A100 / H100)</option>
                <option value="141" ${state.gpuMem === 141 ? 'selected' : ''}>141 GB (H200)</option>
              </select>
            </div>
            <div class="lw-control">
              <label>权重精度 ($b_w$) <span class="lw-val">${state.weightBytes} B/参数</span></label>
              <select id="lw-mem-wb">
                <option value="2" ${state.weightBytes === 2 ? 'selected' : ''}>BF16 / FP16 (2 Bytes)</option>
                <option value="1" ${state.weightBytes === 1 ? 'selected' : ''}>FP8 / INT8 (1 Byte)</option>
                <option value="0.5" ${state.weightBytes === 0.5 ? 'selected' : ''}>INT4 / AWQ / GPTQ (0.5 Byte)</option>
              </select>
            </div>
            <div class="lw-control">
              <label>KV Cache 精度 ($b_{kv}$) <span class="lw-val">${state.kvBytes} B/元素</span></label>
              <select id="lw-mem-kvb">
                <option value="2" ${state.kvBytes === 2 ? 'selected' : ''}>BF16 / FP16 (2 Bytes)</option>
                <option value="1" ${state.kvBytes === 1 ? 'selected' : ''}>FP8 KV Cache (1 Byte)</option>
              </select>
            </div>
            <div class="lw-control">
              <label>KV 头数 ($n_{kv}$) <span class="lw-val">${state.kvHeads} 头 (Q头=${state.qHeads})</span></label>
              <input type="range" id="lw-mem-kvh" min="1" max="64" step="1" value="${state.kvHeads}" />
            </div>
            <div class="lw-control">
              <label>单请求上下文长度 ($S_{\\max}$) <span class="lw-val">${state.seqLen} tokens</span></label>
              <input type="range" id="lw-mem-seq" min="512" max="32768" step="512" value="${state.seqLen}" />
            </div>
            <div class="lw-control">
              <label>张量并行卡数 (TP) <span class="lw-val">${state.tp} 卡</span></label>
              <select id="lw-mem-tp">
                <option value="1" ${state.tp === 1 ? 'selected' : ''}>TP = 1（单卡）</option>
                <option value="2" ${state.tp === 2 ? 'selected' : ''}>TP = 2（双卡切分）</option>
                <option value="4" ${state.tp === 4 ? 'selected' : ''}>TP = 4（四卡切分）</option>
                <option value="8" ${state.tp === 8 ? 'selected' : ''}>TP = 8（八卡切分）</option>
              </select>
            </div>
          </div>

          <div style="font-size:12.5px;font-weight:600;color:var(--ink-2);">单卡显存分配比例（总计 ${state.gpuMem} GB，utilization = ${state.util}）：</div>
          <div class="lw-stack-bar">
            <div class="lw-stack-seg" style="width:${wPct}%;background:var(--ink-2);" title="模型权重">${wPct > 8 ? weightGB.toFixed(1) + 'G 权重' : ''}</div>
            <div class="lw-stack-seg" style="width:${oPct}%;background:var(--decode);" title="激活与框架开销">${oPct > 6 ? state.overhead.toFixed(1) + 'G 激活' : ''}</div>
            <div class="lw-stack-seg" style="width:${kvPct}%;background:var(--prefill);" title="可用 KV Cache 池">${kvPct > 10 ? kvBudgetGB.toFixed(1) + 'G KV 池' : ''}</div>
            <div class="lw-stack-seg" style="width:${uPct}%;background:var(--rule-strong);color:var(--ink-2);" title="安全预留">${uPct > 8 ? unusedGB.toFixed(1) + 'G 预留' : ''}</div>
          </div>
          <div class="lw-legend">
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--ink-2)"></i>模型权重 (${weightGB.toFixed(1)} GB)</span>
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--decode)"></i>激活与 CUDA Graph 开销 (${state.overhead.toFixed(1)} GB)</span>
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--prefill)"></i>可用 KV Cache 预算 (${kvBudgetGB.toFixed(1)} GB)</span>
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--rule-strong)"></i>未分配预留 (${unusedGB.toFixed(1)} GB)</span>
          </div>

          <div class="lw-stats">
            <div class="lw-stat info">
              <div class="lw-stat-label">单 Token KV 大小（每卡）</div>
              <div class="lw-stat-value">${kvPerTokenKiB.toFixed(1)} KiB</div>
              <div class="lw-stat-sub">2 × ${state.layers}层 × ${Math.max(1, Math.floor(state.kvHeads / state.tp))}头 × ${state.headDim}维 × ${state.kvBytes}B</div>
            </div>
            <div class="lw-stat ${isOOM ? 'err' : 'ok'}">
              <div class="lw-stat-label">KV Cache 总容量</div>
              <div class="lw-stat-value">${isOOM ? 'OOM 显存不足' : W.formatNum(maxTokens, 0) + ' tok'}</div>
              <div class="lw-stat-sub">可用预算 ${kvBudgetGB.toFixed(1)} GB</div>
            </div>
            <div class="lw-stat ${maxConcurrency < 4 ? 'warn' : 'ok'}">
              <div class="lw-stat-label">满长度最大并发请求数 ($B_{\\max}$)</div>
              <div class="lw-stat-value">${isOOM ? '0' : maxConcurrency.toFixed(1)} 个</div>
              <div class="lw-stat-sub">每个请求用满 ${state.seqLen} tokens 时</div>
            </div>
          </div>

          <div class="lw-terminal">
            <div>INFO: Loading model weights took <span class="hl-blue">${weightGB.toFixed(2)} GiB</span> (tp=${state.tp})</div>
            ${
              isOOM
                ? `<div class="hl-red">ERROR: CUDA out of memory! Weights (${weightGB.toFixed(1)} GiB) + Overhead (${state.overhead.toFixed(1)} GiB) exceeds usable GPU memory (${usableMemGB.toFixed(1)} GiB). Try INT4 quantization or increase TP!</div>`
                : `<div>INFO: GPU KV cache size: <span class="hl-green">${W.formatNum(maxTokens, 0)} tokens</span> (${kvPerTokenKiB.toFixed(0)} KiB / token)</div>
                   <div>INFO: Maximum concurrency for <span class="hl-amber">${state.seqLen}</span> tokens per request: <span class="hl-green">${maxConcurrency.toFixed(2)}x</span></div>`
            }
          </div>
        </div>
      `;

      container.querySelectorAll('[data-preset]').forEach((btn) => {
        btn.addEventListener('click', () => {
          applyPreset(btn.getAttribute('data-preset'));
          render();
        });
      });
      container.querySelector('#lw-mem-gpu').addEventListener('change', (e) => {
        state.gpuMem = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-mem-wb').addEventListener('change', (e) => {
        state.weightBytes = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-mem-kvb').addEventListener('change', (e) => {
        state.kvBytes = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-mem-kvh').addEventListener('input', (e) => {
        state.kvHeads = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-mem-seq').addEventListener('input', (e) => {
        state.seqLen = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-mem-tp').addEventListener('change', (e) => {
        state.tp = Number(e.target.value);
        render();
      });
    }

    render();
  });

  // =========================================================================
  // 4. 0.4 Interactive Roofline Explorer
  // =========================================================================
  W.register('roofline-explorer', function (container) {
    const gpus = {
      a100: { name: 'A100 80GB SXM', peakTflops: 312, bwTbs: 2.04 },
      h100: { name: 'H100 80GB SXM', peakTflops: 989, bwTbs: 3.35 },
      l40s: { name: 'L40S 48GB', peakTflops: 362, bwTbs: 0.864 },
      a10: { name: 'A10 24GB', peakTflops: 125, bwTbs: 0.6 },
    };

    const state = {
      gpu: 'a100',
      seqLen: 1024, // Prefill prompt length s
      batchSize: 8, // Decode batch size b
      hidden: 4096,
    };

    function render() {
      const g = gpus[state.gpu];
      const ridge = g.peakTflops / g.bwTbs; // FLOPs/Byte
      const h = state.hidden;
      const s = state.seqLen;
      const b = state.batchSize;

      // Arithmetic intensity of linear layer [M, h] x [h, h] in BF16: I = M*h / (2*M + h)
      const prefillM = s;
      const prefillI = (prefillM * h) / (2 * prefillM + h);
      const prefillPerf = Math.min(g.peakTflops, prefillI * g.bwTbs);

      const decodeM = b;
      const decodeI = (decodeM * h) / (2 * decodeM + h);
      const decodePerf = Math.min(g.peakTflops, decodeI * g.bwTbs);

      // Estimate Qwen3-8B (8.19B params, 16.4GB weights, 144KiB/tok KV)
      const weightGB = 16.4;
      const kvBatchGB = (b * s * 144 * 1024) / 1e9;
      const decodeStepMs = ((weightGB + kvBatchGB) / (g.bwTbs * 1000)) * 1000;
      const decodeThroughput = (b / decodeStepMs) * 1000;
      const prefillMs = ((2 * 8.19e9 * s) / (g.peakTflops * 0.6 * 1e12)) * 1000;

      // SVG log-log coordinates: X from I=0.5 to 4096, Y from 1 to 1200 TFLOPS
      const minLogX = Math.log2(0.5);
      const maxLogX = Math.log2(4096);
      const minLogY = Math.log2(1);
      const maxLogY = Math.log2(1400);

      const W_SVG = 620;
      const H_SVG = 240;
      const PAD_L = 54;
      const PAD_R = 24;
      const PAD_T = 22;
      const PAD_B = 34;

      function toX(val) {
        const clamped = Math.max(0.5, Math.min(4096, val));
        const ratio = (Math.log2(clamped) - minLogX) / (maxLogX - minLogX);
        return PAD_L + ratio * (W_SVG - PAD_L - PAD_R);
      }

      function toY(val) {
        const clamped = Math.max(1, Math.min(1400, val));
        const ratio = (Math.log2(clamped) - minLogY) / (maxLogY - minLogY);
        return H_SVG - PAD_B - ratio * (H_SVG - PAD_T - PAD_B);
      }

      const xStart = toX(0.5);
      const yStart = toY(0.5 * g.bwTbs);
      const xRidge = toX(ridge);
      const yPeak = toY(g.peakTflops);
      const xEnd = toX(4096);

      const pxPrefill = toX(prefillI);
      const pyPrefill = toY(prefillPerf);
      const pxDecode = toX(decodeI);
      const pyDecode = toY(decodePerf);

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <h4 class="lw-title">Roofline 算术强度与 Prefill / Decode 落点模拟器</h4>
            </div>
          </div>
          <div class="lw-controls">
            <div class="lw-control">
              <label>选择 GPU 硬件</label>
              <select id="lw-rf-gpu">
                ${Object.entries(gpus)
                  .map(([k, v]) => `<option value="${k}" ${state.gpu === k ? 'selected' : ''}>${v.name} (拐点 I*=${Math.round(v.peakTflops / v.bwTbs)})</option>`)
                  .join('')}
              </select>
            </div>
            <div class="lw-control">
              <label>Decode 批大小 ($b$) <span class="lw-val">${state.batchSize} 请求并发</span></label>
              <input type="range" id="lw-rf-batch" min="1" max="512" step="1" value="${state.batchSize}" />
            </div>
            <div class="lw-control">
              <label>Prefill 序列长度 ($s$) <span class="lw-val">${state.seqLen} tokens</span></label>
              <input type="range" id="lw-rf-seq" min="16" max="4096" step="16" value="${state.seqLen}" />
            </div>
          </div>

          <div style="background:var(--paper-2);border:1px solid var(--rule);border-radius:8px;padding:10px;text-align:center;">
            <svg viewBox="0 0 ${W_SVG} ${H_SVG}" style="width:100%;max-width:${W_SVG}px;height:auto;">
              <!-- 坐标轴 -->
              <line x1="${PAD_L}" y1="${H_SVG - PAD_B}" x2="${W_SVG - PAD_R}" y2="${H_SVG - PAD_B}" stroke="${W.color('ink-3')}" stroke-width="1.5"/>
              <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${H_SVG - PAD_B}" stroke="${W.color('ink-3')}" stroke-width="1.5"/>

              <!-- 拐点垂直虚线 -->
              <line x1="${xRidge}" y1="${yPeak}" x2="${xRidge}" y2="${H_SVG - PAD_B}" stroke="${W.color('ink-3')}" stroke-dasharray="4 4" stroke-width="1.5"/>
              <text x="${xRidge}" y="${H_SVG - PAD_B + 14}" font-size="10" fill="var(--decode-ink)" text-anchor="middle">拐点 I*=${Math.round(ridge)}</text>

              <!-- Roofline 折线 -->
              <polyline points="${xStart},${yStart} ${xRidge},${yPeak} ${xEnd},${yPeak}" fill="none" stroke="var(--ink)" stroke-width="3"/>

              <!-- 区域标注 -->
              <text x="${PAD_L + 14}" y="${PAD_T + 16}" font-size="11" fill="var(--ink-3)">↖ 斜线区：受显存带宽限制 (Memory-bound)</text>
              <text x="${W_SVG - PAD_R - 10}" y="${yPeak - 8}" font-size="11" fill="var(--ink)" text-anchor="end">平顶区：受峰值算力限制 (${g.peakTflops} TFLOPS)</text>

              <!-- Decode 落点 -->
              <circle cx="${pxDecode}" cy="${pyDecode}" r="7" fill="${W.color('decode')}" stroke="${W.color('paper')}" stroke-width="2"/>
              <text x="${pxDecode + 10}" y="${pyDecode + 4}" font-size="11" font-weight="bold" fill="var(--prefill-ink)">Decode (b=${b}, I=${decodeI.toFixed(1)})</text>

              <!-- Prefill 落点 -->
              <circle cx="${pxPrefill}" cy="${pyPrefill}" r="7" fill="${W.color('prefill')}" stroke="${W.color('paper')}" stroke-width="2"/>
              <text x="${pxPrefill - 10}" y="${pyPrefill - 10}" font-size="11" font-weight="bold" fill="var(--ok)" text-anchor="end">Prefill (s=${s}, I=${prefillI.toFixed(0)})</text>

              <!-- 轴标题 -->
              <text x="${(PAD_L + W_SVG - PAD_R) / 2}" y="${H_SVG - 4}" font-size="11" fill="var(--ink-2)" text-anchor="middle">算术强度 I (FLOPs / Byte，对数刻度)</text>
              <text x="14" y="${H_SVG / 2}" font-size="11" fill="var(--ink-2)" text-anchor="middle" transform="rotate(-90 14 ${H_SVG / 2})">可达算力 (TFLOPS)</text>
            </svg>
          </div>

          <div class="lw-stats">
            <div class="lw-stat ${decodeI < ridge ? 'info' : 'ok'}">
              <div class="lw-stat-label">Decode 阶段状态 (b=${b})</div>
              <div class="lw-stat-value" style="font-size:15px">${decodeI < ridge ? '访存带宽受限' : '算力受限'}</div>
              <div class="lw-stat-sub">可达算力 ${decodePerf.toFixed(1)} TFLOPS（利用率 ${((decodePerf / g.peakTflops) * 100).toFixed(1)}%）</div>
            </div>
            <div class="lw-stat ${prefillI < ridge ? 'warn' : 'ok'}">
              <div class="lw-stat-label">Prefill 阶段状态 (s=${s})</div>
              <div class="lw-stat-value" style="font-size:15px">${prefillI < ridge ? '访存带宽受限' : '算力受限'}</div>
              <div class="lw-stat-sub">可达算力 ${prefillPerf.toFixed(0)} TFLOPS（利用率 ${((prefillPerf / g.peakTflops) * 100).toFixed(0)}%）</div>
            </div>
            <div class="lw-stat ok">
              <div class="lw-stat-label">Qwen3-8B 单步 Decode 下限</div>
              <div class="lw-stat-value">${decodeStepMs.toFixed(1)} ms</div>
              <div class="lw-stat-sub">总输出吞吐上限 ≈ ${Math.round(decodeThroughput)} tok/s</div>
            </div>
            <div class="lw-stat ok">
              <div class="lw-stat-label">Qwen3-8B Prefill 耗时估算</div>
              <div class="lw-stat-value">${prefillMs.toFixed(0)} ms</div>
              <div class="lw-stat-sub">按 60% MFU 估算 TTFT 下限</div>
            </div>
          </div>
        </div>
      `;

      container.querySelector('#lw-rf-gpu').addEventListener('change', (e) => {
        state.gpu = e.target.value;
        render();
      });
      container.querySelector('#lw-rf-batch').addEventListener('input', (e) => {
        state.batchSize = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-rf-seq').addEventListener('input', (e) => {
        state.seqLen = Number(e.target.value);
        render();
      });
    }

    render();
  });

  // =========================================================================
  // 5. 0.5 Metrics & Goodput Simulator
  // =========================================================================
  W.register('metrics-goodput-sim', function (container) {
    const state = {
      qps: 6,           // Request arrival rate (req/s)
      sloTtft: 1000,    // ms
      sloTpot: 50,      // ms
      outputLen: 128,   // tokens
    };

    function render() {
      const qps = state.qps;
      // Simulate realistic queueing & batching behavior on a single 80GB GPU running 8B model
      const saturationQps = 12.0;
      const loadRatio = qps / saturationQps;

      // Active batch size grows with QPS, then explodes near/above saturation
      const activeBatch = Math.min(128, Math.round(qps * 2.8 * (1 + Math.pow(Math.max(0, loadRatio - 0.6), 2) * 4)));
      // TTFT P99 = base prefill (180ms) + queueing delay + interference
      const ttftP50 = Math.round(180 + Math.pow(loadRatio, 2) * 320);
      const ttftP99 = Math.round(ttftP50 * (1.35 + Math.max(0, loadRatio - 0.7) * 2.8));

      // TPOT P50 & P99 = base step (9ms) + batch growth + prefill interference spikes
      const tpotP50 = Math.round(9 + activeBatch * 0.38);
      const itlP99 = Math.round(tpotP50 * 1.4 + (qps > 4 ? 28 : 8));
      const tpotP99 = Math.round(tpotP50 * 1.25 + Math.max(0, loadRatio - 0.65) * 45);

      const e2eP50 = ((ttftP50 + tpotP50 * (state.outputLen - 1)) / 1000).toFixed(2);
      const rawThroughput = Math.min(saturationQps, qps) * state.outputLen;

      // Fraction of requests meeting both SLOs
      const ttftPass = ttftP99 <= state.sloTtft ? 1.0 : Math.max(0, 1 - (ttftP99 - state.sloTtft) / state.sloTtft);
      const tpotPass = tpotP99 <= state.sloTpot ? 1.0 : Math.max(0, 1 - (tpotP99 - state.sloTpot) / state.sloTpot);
      const passRate = Math.max(0, Math.min(1, ttftPass * tpotPass));
      const goodputReq = Math.min(saturationQps, qps) * passRate;

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <h4 class="lw-title">请求时间线、吞吐饱和与 Goodput（有效吞吐）模拟器</h4>
            </div>
          </div>
          <div class="lw-controls">
            <div class="lw-control">
              <label>线上请求到达率 (QPS) <span class="lw-val">${state.qps} req/s</span></label>
              <input type="range" id="lw-met-qps" min="1" max="18" step="1" value="${state.qps}" />
            </div>
            <div class="lw-control">
              <label>TTFT P99 SLO 上限 <span class="lw-val">${state.sloTtft} ms</span></label>
              <input type="range" id="lw-met-ttft" min="300" max="3000" step="100" value="${state.sloTtft}" />
            </div>
            <div class="lw-control">
              <label>TPOT P99 SLO 上限 <span class="lw-val">${state.sloTpot} ms/tok</span></label>
              <input type="range" id="lw-met-tpot" min="20" max="100" step="5" value="${state.sloTpot}" />
            </div>
          </div>

          <div style="font-size:12.5px;font-weight:600;color:var(--ink-2);margin-bottom:6px;">
            单个请求的端到端时间轴分解（E2E P50 ≈ ${e2eP50} 秒，共生成 ${state.outputLen} tokens）：
          </div>
          <div class="lw-stack-bar" style="height:32px;">
            <div class="lw-stack-seg" style="width:${Math.min(70, Math.max(15, (ttftP50 / (ttftP50 + tpotP50 * state.outputLen)) * 100))}%;background:var(--prefill);">
              TTFT (排队+Prefill): ${ttftP50} ms
            </div>
            <div class="lw-stack-seg" style="flex:1;background:var(--decode);">
              Decode 阶段 (${state.outputLen - 1} 步 × TPOT ${tpotP50} ms/tok, ITL P99 尖峰 ${itlP99} ms)
            </div>
          </div>

          <div class="lw-stats">
            <div class="lw-stat info">
              <div class="lw-stat-label">原始输出吞吐 (Raw TPS)</div>
              <div class="lw-stat-value">${Math.round(rawThroughput)} tok/s</div>
              <div class="lw-stat-sub">并发活跃批大小 ≈ ${activeBatch}</div>
            </div>
            <div class="lw-stat ${ttftP99 <= state.sloTtft ? 'ok' : 'err'}">
              <div class="lw-stat-label">TTFT (P50 / P99)</div>
              <div class="lw-stat-value">${ttftP50} / ${ttftP99} ms</div>
              <div class="lw-stat-sub">SLO 要求 P99 ≤ ${state.sloTtft} ms</div>
            </div>
            <div class="lw-stat ${tpotP99 <= state.sloTpot ? 'ok' : 'err'}">
              <div class="lw-stat-label">TPOT (P50 / P99)</div>
              <div class="lw-stat-value">${tpotP50} / ${tpotP99} ms</div>
              <div class="lw-stat-sub">SLO 要求 P99 ≤ ${state.sloTpot} ms</div>
            </div>
            <div class="lw-stat ${passRate > 0.85 ? 'ok' : passRate > 0.4 ? 'warn' : 'err'}">
              <div class="lw-stat-label">有效吞吐 (Goodput)</div>
              <div class="lw-stat-value">${goodputReq.toFixed(1)} req/s</div>
              <div class="lw-stat-sub">SLO 达标率 ${(passRate * 100).toFixed(0)}%</div>
            </div>
          </div>

          <div class="lw-callout">
            <strong>观察结论：</strong>
            ${
              loadRatio < 0.6
                ? '当前 QPS 处于轻载区，TTFT 和 TPOT 都远低于 SLO 紅线，全部吞吐都是有效吞吐（Goodput = 100% QPS），还可以继续提高并发降低单 Token 成本。'
                : passRate > 0.7
                ? '当前处于黄金工作点（Sweet Spot）附近。GPU 批处理效率高，且绝大多数请求仍满足 SLO。'
                : '警报：虽然原始输出吞吐（Raw TPS）看起来很高，但因为排队和批大小过大，TTFT/TPOT 已经击穿 SLO 红线，有效吞吐（Goodput）反而暴跌。这正是线上压测不能只看最高吞吐的原因。'
            }
          </div>
        </div>
      `;

      container.querySelector('#lw-met-qps').addEventListener('input', (e) => {
        state.qps = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-met-ttft').addEventListener('input', (e) => {
        state.sloTtft = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-met-tpot').addEventListener('input', (e) => {
        state.sloTpot = Number(e.target.value);
        render();
      });
    }

    render();
  });
})();
