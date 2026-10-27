/**
 * Stage 1 Interactive Widgets:
 * 1. transformer-tensor-inspector (1.1 Decoder-only Transformer 与注意力)
 * 2. sampling-autoregressive-lab  (1.2 自回归生成的执行过程)
 * 3. kv-cache-animator            (1.3 KV Cache 与 Prefill / Decode)
 * 4. async-streaming-flow         (1.4 从零实现在线服务：批处理与流式输出)
 * 5. batching-timeline-sim        (1.5 Continuous Batching 与 Chunked Prefill)
 * 6. paged-attention-sim          (1.6 vLLM 架构与调度器)
 * 7. prefix-cache-matcher         (1.7 Prefix Caching 与 RadixAttention)
 * 8. framework-advisor            (1.8 推理框架对比)
 * 9. miniserve-batch-sandbox      (项目 1 MiniServe)
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  // =========================================================================
  // 1. 1.1 Transformer Tensor Shape & FLOPs Inspector
  // =========================================================================
  W.register('transformer-tensor-inspector', function (container) {
    const state = {
      seqLen: 1024,
      stepIdx: 2,
    };

    const steps = [
      { name: '① Token Embedding', inShape: (T) => `[${T}] (Token IDs)`, outShape: (T) => `[${T}, 4096]`, desc: '查词表矩阵 [151936, 4096]，把每个整数 ID 映射成 4096 维浮点数向量。' },
      { name: '② QKV 线性投影', inShape: (T) => `[${T}, 4096]`, outShape: (T) => `Q:[${T}, 32, 128], K/V:[${T}, 8, 128]`, desc: 'Qwen3-8B 采用 GQA：32 个 Query 头（总维 4096），但只有 8 个 K/V 头（总维 1024），每 4 个 Q 头共享 1 组 K/V。' },
      { name: '③ RoPE + 因果自注意力', inShape: (T) => `Q:[${T}, 32, 128], K/V:[${T}, 8, 128]`, outShape: (T) => `[${T}, 4096]`, desc: '先按位置旋转 Q、K（RoPE），再做因果掩码下的打分 softmax(QK^T / √128)V，最后经 W_O 投影回 4096 维。注意：计算量随序列长度二次方或线性增长！' },
      { name: '④ SwiGLU 前馈网络 (MLP)', inShape: (T) => `[${T}, 4096]`, outShape: (T) => `[${T}, 4096]`, desc: '先用 Gate 和 Up 两个矩阵升维到 [T, 12288]，逐元素相乘后再经 Down 矩阵降回 [T, 4096]。占全模型近 78% 的权重参数！' },
      { name: '⑤ LM Head (仅取最后 1 行)', inShape: (T) => `[${T}, 4096] → 切片 [1, 4096]`, outShape: () => `[1, 151936] (Logits)`, desc: '生成下一个词只依赖最后一个位置的隐藏状态！只对最后 1 行做词表投影，省去 (T-1) 行的无用计算与数 GB 显存。' },
    ];

    function render() {
      const T = state.seqLen;
      const cur = steps[state.stepIdx];
      // Single token decode FLOPs comparison on Qwen3-8B (36 layers)
      const linearGflops = 16.4; // 2 * 8.19B
      const attnGflops = (4 * 32 * 128 * T * 36) / 1e9;
      const totalGflops = linearGflops + attnGflops;
      const attnPct = ((attnGflops / totalGflops) * 100).toFixed(1);

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Inspector</span>
              <h4 class="lw-title">Qwen3-8B 张量形状追踪器与长序列 Attention 开销透视</h4>
            </div>
          </div>
          <div class="lw-controls">
            <div class="lw-control">
              <label>当前上下文序列长度 ($T$) <span class="lw-val">${T} tokens</span></label>
              <input type="range" id="lw-tf-seq" min="128" max="32768" step="128" value="${T}" />
            </div>
          </div>
          <div class="lw-tabs">
            ${steps.map((s, i) => `<button type="button" class="lw-tab ${i === state.stepIdx ? 'active' : ''}" data-idx="${i}">${s.name}</button>`).join('')}
          </div>
          <div class="lw-stats">
            <div class="lw-stat info">
              <div class="lw-stat-label">当前模块输入张量形状</div>
              <div class="lw-stat-value" style="font-size:14px">${cur.inShape(T)}</div>
            </div>
            <div class="lw-stat ok">
              <div class="lw-stat-label">当前模块输出张量形状</div>
              <div class="lw-stat-value" style="font-size:14px">${cur.outShape(T)}</div>
            </div>
          </div>
          <div class="lw-callout">${cur.desc}</div>
          <div style="margin-top:14px;font-size:12.5px;font-weight:600;color:#334155;">
            Decode 单步计算量构成（上下文 $T = ${T}$ 时，线性层固定 ${linearGflops} GFLOPs vs 注意力 ${attnGflops.toFixed(1)} GFLOPs，注意力占 ${attnPct}%）：
          </div>
          <div class="lw-stack-bar">
            <div class="lw-stack-seg" style="width:${100 - attnPct}%;background:#3b82f6;">线性层 (MLP+QKV+O): ${linearGflops} GFLOPs</div>
            <div class="lw-stack-seg" style="width:${attnPct}%;background:#f59e0b;">${attnPct > 12 ? 'Attention: ' + attnGflops.toFixed(1) + ' GFLOPs' : ''}</div>
          </div>
        </div>
      `;

      container.querySelector('#lw-tf-seq').addEventListener('input', (e) => {
        state.seqLen = Number(e.target.value);
        render();
      });
      container.querySelectorAll('[data-idx]').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.stepIdx = Number(btn.getAttribute('data-idx'));
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 2. 1.2 Sampling & Autoregressive Step Lab
  // =========================================================================
  W.register('sampling-autoregressive-lab', function (container) {
    const rawCandidates = [
      { token: '北京', logit: 4.2 },
      { token: '上海', logit: 3.1 },
      { token: '一座', logit: 2.4 },
      { token: '中国', logit: 1.8 },
      { token: '美丽', logit: 1.1 },
      { token: '火锅', logit: -0.5 },
    ];
    const state = {
      temp: 1.0,
      topK: 6,
      topP: 1.0,
    };

    function render() {
      const t = Math.max(0.05, state.temp);
      const scaled = rawCandidates.map((c) => ({ ...c, sLogit: c.logit / t }));
      const maxL = Math.max(...scaled.map((c) => c.sLogit));
      const exps = scaled.map((c) => ({ ...c, exp: Math.exp(c.sLogit - maxL) }));
      const sumExp = exps.reduce((a, b) => a + b.exp, 0);
      const probs = exps.map((c) => ({ ...c, prob: c.exp / sumExp }));

      let cum = 0;
      const filtered = probs.map((c, idx) => {
        const passK = idx < state.topK;
        const passP = cum < state.topP || idx === 0;
        cum += c.prob;
        const kept = state.temp === 0 ? idx === 0 : passK && passP;
        return { ...c, kept };
      });
      const keptSum = filtered.reduce((a, b) => a + (b.kept ? b.prob : 0), 0) || 1;

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Lab</span>
              <h4 class="lw-title">解码采样实验室：Temperature、Top-K 与 Top-P 核采样</h4>
            </div>
          </div>
          <p class="lw-subtitle">假设当前输入是 <code>"中国的首都是"</code>，模型 LM Head 输出了以下 6 个候选词的原始分数（Logits）。拖动滑块观察概率分布如何被重塑与截断：</p>
          <div class="lw-controls">
            <div class="lw-control">
              <label>温度 Temperature ($\\tau$) <span class="lw-val">${state.temp === 0 ? '0 (退化为贪心 Greedy)' : state.temp.toFixed(2)}</span></label>
              <input type="range" id="lw-sp-temp" min="0" max="2.0" step="0.1" value="${state.temp}" />
            </div>
            <div class="lw-control">
              <label>Top-K 截断 ($k$) <span class="lw-val">保留前 ${state.topK} 个</span></label>
              <input type="range" id="lw-sp-k" min="1" max="6" step="1" value="${state.topK}" />
            </div>
            <div class="lw-control">
              <label>Top-P 核采样 ($p$) <span class="lw-val">${state.topP.toFixed(2)}</span></label>
              <input type="range" id="lw-sp-p" min="0.5" max="1.0" step="0.05" value="${state.topP}" />
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            ${filtered
              .map((c) => {
                const finalP = c.kept ? (state.temp === 0 ? 1.0 : c.prob / keptSum) : 0;
                return `
                  <div style="display:flex;align-items:center;gap:10px;opacity:${c.kept ? 1 : 0.4};">
                    <div style="width:95px;font-family:monospace;font-size:12.5px;font-weight:700;">"${c.token}" <span style="color:#64748b;font-weight:400;">(${c.logit})</span></div>
                    <div style="flex:1;height:20px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                      <div style="height:100%;width:${(finalP * 100).toFixed(1)}%;background:${c.kept ? '#10b981' : '#94a3b8'};transition:width 0.2s;"></div>
                    </div>
                    <div style="width:130px;font-family:monospace;font-size:12px;text-align:right;">
                      ${c.kept ? `<strong>${(finalP * 100).toFixed(1)}%</strong> (原始 ${(c.prob * 100).toFixed(1)}%)` : '<span style="color:#ef4444;">已截断淘汰</span>'}
                    </div>
                  </div>
                `;
              })
              .join('')}
          </div>
        </div>
      `;

      container.querySelector('#lw-sp-temp').addEventListener('input', (e) => {
        state.temp = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-sp-k').addEventListener('input', (e) => {
        state.topK = Number(e.target.value);
        render();
      });
      container.querySelector('#lw-sp-p').addEventListener('input', (e) => {
        state.topP = Number(e.target.value);
        render();
      });
    }
    render();
  });

  // =========================================================================
  // 3. 1.3 KV Cache Prefill vs Decode Step Animator
  // =========================================================================
  W.register('kv-cache-animator', function (container) {
    const tokens = ['[Prompt] AI', '[Prompt] 推理', '[Prompt] 很', '[Gen 1] 快', '[Gen 2] 且', '[Gen 3] 省'];
    const state = { step: 0 }; // 0 = Prefill(3 tokens), 1 = Decode 1, 2 = Decode 2, 3 = Decode 3

    function render() {
      const curLen = 3 + state.step;
      const activeTokens = tokens.slice(0, curLen);
      const noCacheFlops = [3, 3 + 4, 3 + 4 + 5, 3 + 4 + 5 + 6][state.step];
      const withCacheFlops = [3, 3 + 1, 3 + 2, 3 + 3][state.step];

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Step-by-Step Animator</span>
              <h4 class="lw-title">KV Cache 矩阵复用步进动画：为什么只缓存 K、V 而不缓存 Q？</h4>
            </div>
            <div class="lw-btn-row" style="margin:0;">
              <button type="button" class="lw-btn" id="lw-kv-prev" ${state.step === 0 ? 'disabled' : ''}>◀ 上一步</button>
              <button type="button" class="lw-btn primary" id="lw-kv-next" ${state.step === 3 ? 'disabled' : ''}>下一步生成 ▶</button>
              <button type="button" class="lw-btn" id="lw-kv-reset">重置</button>
            </div>
          </div>
          <div class="lw-tabs">
            ${['第 0 步：Prefill (一次并行算 3 个 Prompt 词)', '第 1 步：Decode 生成 "快"', '第 2 步：Decode 生成 "且"', '第 3 步：Decode 生成 "省"']
              .map((label, idx) => `<button type="button" class="lw-tab ${state.step === idx ? 'active' : ''}" data-step="${idx}">${label}</button>`)
              .join('')}
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;">
            <div style="border:1px solid #fecaca;background:#fef2f2;border-radius:8px;padding:12px;">
              <div style="font-weight:700;font-size:13px;color:#b91c1c;margin-bottom:8px;">❌ 无 KV Cache（每步把全部 ${curLen} 个词重新送入模型）</div>
              ${activeTokens
                .map(
                  (t) => `<div style="background:#fca5a5;color:#7f1d1d;padding:5px 10px;border-radius:4px;margin-bottom:4px;font-size:12px;font-weight:600;">
                  🔄 重新计算 Q, K, V：${t}
                </div>`
                )
                .join('')}
              <div style="font-size:12px;color:#991b1b;margin-top:8px;">本步计算行数：<strong>${curLen} 行</strong> ｜ 累计已算：<strong>${noCacheFlops} 行</strong></div>
            </div>
            <div style="border:1px solid #a7f3d0;background:#ecfdf5;border-radius:8px;padding:12px;">
              <div style="font-weight:700;font-size:13px;color:#047857;margin-bottom:8px;">✅ 开启 KV Cache（只送最新词，直接读取历史 K、V）</div>
              ${activeTokens
                .map((t, idx) => {
                  const isNew = state.step === 0 || idx === curLen - 1;
                  return `<div style="background:${isNew ? '#fde68a' : '#6ee7b7'};color:#064e3b;padding:5px 10px;border-radius:4px;margin-bottom:4px;font-size:12px;font-weight:600;">
                    ${isNew ? '⚡️ 本步新算 Q, K, V 并写入缓存：' : '📦 从显存直接复用 K, V（Q 已丢弃）：'}${t}
                  </div>`;
                })
                .join('')}
              <div style="font-size:12px;color:#065f46;margin-top:8px;">本步仅算：<strong>${state.step === 0 ? 3 : 1} 行</strong> ｜ 累计仅算：<strong>${withCacheFlops} 行</strong></div>
            </div>
          </div>
        </div>
      `;

      container.querySelector('#lw-kv-prev').addEventListener('click', () => {
        if (state.step > 0) { state.step--; render(); }
      });
      container.querySelector('#lw-kv-next').addEventListener('click', () => {
        if (state.step < 3) { state.step++; render(); }
      });
      container.querySelector('#lw-kv-reset').addEventListener('click', () => {
        state.step = 0; render();
      });
      container.querySelectorAll('[data-step]').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.step = Number(btn.getAttribute('data-step'));
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 4. 1.4 Async Streaming Architecture Flow State Machine
  // =========================================================================
  W.register('async-streaming-flow', function (container) {
    const stages = [
      { title: '① 客户端发起并发 SSE 请求', detail: '用户 A 和用户 B 同时请求 /v1/completions(stream=true)。FastAPI 为每个请求创建一个专属的 asyncio.Queue，并将请求封装成 (seq, queue) 放入线程安全的 inbox 队列。', activeBox: 0 },
      { title: '② 后台引擎线程从 inbox 取任务', detail: '独立的后台线程 AsyncEngine._loop 被唤醒，把 inbox 里的新请求全部取出并注册进 Scheduler.waiting 队列，同时记住 seq_id → asyncio.Queue 的映射。', activeBox: 1 },
      { title: '③ 执行一步 engine.step() 合批推理', detail: 'Scheduler 将 A 和 B 组成一个 Batch，GPU 一次前向传播同时算出 A 的新词 "Hello" 和 B 的新词 "你好"。', activeBox: 2 },
      { title: '④ 跨线程精准投递回各自的协程队列', detail: '后台线程调用 loop.call_soon_threadsafe(q.put_nowait, out)，把 "Hello" 塞进 A 的 asyncio.Queue，把 "你好" 塞进 B 的 asyncio.Queue，两边互不串台、立刻通过 SSE 推给浏览器！', activeBox: 3 },
    ];
    let step = 0;

    function render() {
      const s = stages[step];
      const boxes = ['FastAPI 路由协程\n(创建专属 asyncio.Queue)', '线程安全 inbox 队列\n(跨线程桥接)', '后台线程 LLMEngine.step()\n(GPU 合批前向)', 'call_soon_threadsafe\n(逐 Token SSE 推送)'];

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Walkthrough</span>
              <h4 class="lw-title">异步流式批处理状态机：Web 协程与 GPU 后台线程如何协作？</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${stages.map((item, idx) => `<button type="button" class="lw-tab ${idx === step ? 'active' : ''}" data-s="${idx}">${item.title}</button>`).join('')}
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:12px 0;">
            ${boxes
              .map(
                (b, idx) => `
              <div style="padding:12px;border-radius:8px;border:2px solid ${idx === s.activeBox ? '#10b981' : '#cbd5e1'};background:${idx === s.activeBox ? 'rgba(16,185,129,0.1)' : '#f8fafc'};text-align:center;font-size:12.5px;font-weight:600;white-space:pre-line;">
                ${b}
              </div>
            `
              )
              .join('')}
          </div>
          <div class="lw-callout">${s.detail}</div>
        </div>
      `;

      container.querySelectorAll('[data-s]').forEach((btn) => {
        btn.addEventListener('click', () => {
          step = Number(btn.getAttribute('data-s'));
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 5. 1.5 Continuous Batching & Chunked Prefill Timeline Simulator
  // =========================================================================
  W.register('batching-timeline-sim', function (container) {
    const state = { mode: 'chunked' };
    const modes = {
      static: {
        name: '① 静态批处理 (Static Batching)',
        ttft: '1690 ms',
        itlMax: '9.2 ms',
        util: '42% (大量空泡等待)',
        desc: '请求 1 虽然第 3 秒就生成完了，但必须干等请求 3（第 10 秒结束）才能释放整批；期间新来的请求 4 只能在门外干等 7 秒！',
        rows: [
          { label: '槽位 1', segs: [{ w: 30, c: '#10b981', t: 'Req1 (3s)' }, { w: 70, c: '#fecaca', t: '❌ 空闲等待 Req3 结束 (7s)' }] },
          { label: '槽位 2', segs: [{ w: 50, c: '#3b82f6', t: 'Req2 (5s)' }, { w: 50, c: '#fecaca', t: '❌ 空闲等待 (5s)' }] },
          { label: '槽位 3', segs: [{ w: 100, c: '#8b5cf6', t: 'Req3 长回答 (10s)' }] },
        ],
      },
      continuous: {
        name: '② 连续批处理 · 不切块 (Continuous, No Chunking)',
        ttft: '264 ms',
        itlMax: '62.7 ms (⚠️ 被长 Prefill 阻塞尖峰)',
        util: '86%',
        desc: 'Req1 一结束，Req4 立刻无缝补进槽位 1！但因为 Req4 的 Prompt 很长，它在做完整 Prefill 的那一步，把正在 Decode 的 Req2、Req3 硬生生卡停了 62.7 ms！',
        rows: [
          { label: '槽位 1', segs: [{ w: 30, c: '#10b981', t: 'Req1 (3s)' }, { w: 25, c: '#f59e0b', t: '⚡️ Req4 长 Prefill' }, { w: 45, c: '#10b981', t: 'Req4 Decode' }] },
          { label: '槽位 2', segs: [{ w: 30, c: '#3b82f6', t: 'Req2' }, { w: 25, c: '#fca5a5', t: '⏸ 被 Req4 卡顿' }, { w: 20, c: '#3b82f6', t: 'Req2 完' }, { w: 25, c: '#06b6d4', t: 'Req5' }] },
          { label: '槽位 3', segs: [{ w: 30, c: '#8b5cf6', t: 'Req3' }, { w: 25, c: '#fca5a5', t: '⏸ 被 Req4 卡顿' }, { w: 45, c: '#8b5cf6', t: 'Req3 继续 Decode' }] },
        ],
      },
      chunked: {
        name: '③ 连续批处理 + Chunked Prefill (推荐)',
        ttft: '150 ms',
        itlMax: '22.7 ms (✅ 丝滑无尖峰)',
        util: '94% (算力与带宽双丰收)',
        desc: '将 Req4 的长 Prompt 切成 3 个小块（受 max_num_batched_tokens 控制），每一块捎带脚和 Req2、Req3 的单步 Decode 一起算，既消除了空泡，又消灭了卡顿尖峰！',
        rows: [
          { label: '槽位 1', segs: [{ w: 30, c: '#10b981', t: 'Req1 (3s)' }, { w: 12, c: '#f59e0b', t: 'P块1' }, { w: 12, c: '#f59e0b', t: 'P块2' }, { w: 12, c: '#f59e0b', t: 'P块3' }, { w: 34, c: '#10b981', t: 'Req4 Decode' }] },
          { label: '槽位 2', segs: [{ w: 50, c: '#3b82f6', t: 'Req2 全程丝滑 Decode (5s)' }, { w: 50, c: '#06b6d4', t: 'Req5 立即接替 (5s)' }] },
          { label: '槽位 3', segs: [{ w: 100, c: '#8b5cf6', t: 'Req3 全程丝滑 Decode，无长卡顿 (10s)' }] },
        ],
      },
    };

    function render() {
      const m = modes[state.mode];
      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Timeline</span>
              <h4 class="lw-title">批处理演进甘特图：从静态批处理到 Chunked Prefill</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${Object.entries(modes).map(([k, v]) => `<button type="button" class="lw-tab ${state.mode === k ? 'active' : ''}" data-m="${k}">${v.name}</button>`).join('')}
          </div>
          <div style="display:flex;flex-direction:column;gap:8px;margin:14px 0;">
            ${m.rows
              .map(
                (r) => `
              <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:60px;font-size:12px;font-weight:700;color:#475569;">${r.label}</div>
                <div class="lw-stack-bar" style="flex:1;margin:0;">
                  ${r.segs.map((sg) => `<div class="lw-stack-seg" style="width:${sg.w}%;background:${sg.c};color:#0f172a;">${sg.t}</div>`).join('')}
                </div>
              </div>
            `
              )
              .join('')}
          </div>
          <div class="lw-stats">
            <div class="lw-stat info"><div class="lw-stat-label">排队与首字 TTFT P50</div><div class="lw-stat-value">${m.ttft}</div></div>
            <div class="lw-stat ${state.mode === 'continuous' ? 'warn' : 'ok'}"><div class="lw-stat-label">最大单步间隔 (ITL Max)</div><div class="lw-stat-value" style="font-size:14px">${m.itlMax}</div></div>
            <div class="lw-stat ok"><div class="lw-stat-label">GPU 有效利用率</div><div class="lw-stat-value" style="font-size:14px">${m.util}</div></div>
          </div>
          <div class="lw-callout">${m.desc}</div>
        </div>
      `;

      container.querySelectorAll('[data-m]').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.mode = btn.getAttribute('data-m');
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 6. 1.6 PagedAttention Block Allocator & Preemption Simulator
  // =========================================================================
  W.register('paged-attention-sim', function (container) {
    // 8 physical blocks, block_size = 4 tokens
    const state = {
      step: 0,
    };

    const snapshots = [
      {
        label: '状态 1：Req A (6 tok) 与 Req B (5 tok) 运行中',
        note: '每块容量 4 tokens。Req A 用了 2 个物理块（块 #1 满 4 个 + 块 #4 用 2 个，内部碎片仅 2 槽位）；Req B 用了 2 个物理块（块 #2 满 + 块 #6 用 1 个）。物理块完全不需要连续！',
        seqs: [
          { name: 'Req A (6 tokens)', color: '#10b981', bt: [1, 4], usedInLast: 2 },
          { name: 'Req B (5 tokens)', color: '#3b82f6', bt: [2, 6], usedInLast: 1 },
        ],
      },
      {
        label: '状态 2：Req C (9 tok) 到达，按需分配 3 个物理块',
        note: 'Req C 需要 ⌈9/4⌉ = 3 个块，分配器从空闲池弹出 #0、#3、#5。此时 8 个物理块已用 7 个，仅剩 #7 空闲。传统连续预留若按 max_len=16 分配，这 3 个请求早就在第 1 步 OOM 了！',
        seqs: [
          { name: 'Req A (6 tokens)', color: '#10b981', bt: [1, 4], usedInLast: 2 },
          { name: 'Req B (5 tokens)', color: '#3b82f6', bt: [2, 6], usedInLast: 1 },
          { name: 'Req C (9 tokens)', color: '#8b5cf6', bt: [0, 3, 5], usedInLast: 1 },
        ],
      },
      {
        label: '状态 3：继续生成导致物理块耗尽 → 触发队尾抢占 (Preempt Req C)',
        note: '当 Req A 和 Req B 继续增长需要新块而空闲池为空时，调度器暂停队尾的 Req C，释放它的物理块 #0、#3、#5，将其 num_computed_tokens 清零放回 waiting 队列，优先保住排在前面的 Req A 和 Req B 完成生成！',
        seqs: [
          { name: 'Req A (9 tokens, 扩块 #7)', color: '#10b981', bt: [1, 4, 7], usedInLast: 1 },
          { name: 'Req B (9 tokens, 扩块 #0)', color: '#3b82f6', bt: [2, 6, 0], usedInLast: 1 },
          { name: 'Req C (⚠️ 已抢占放回 Waiting, 释放块)', color: '#94a3b8', bt: [], usedInLast: 0 },
        ],
      },
    ];

    function render() {
      const snap = snapshots[state.step];
      const blockOwner = Array(8).fill(null);
      snap.seqs.forEach((s) => {
        s.bt.forEach((bId, idx) => {
          const isLast = idx === s.bt.length - 1;
          blockOwner[bId] = { name: s.name.split(' ')[1], color: s.color, filled: isLast ? s.usedInLast : 4 };
        });
      });

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Allocator</span>
              <h4 class="lw-title">PagedAttention 物理块池（Block Pool）与重算抢占模拟器</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${snapshots.map((s, i) => `<button type="button" class="lw-tab ${i === state.step ? 'active' : ''}" data-st="${i}">${s.label}</button>`).join('')}
          </div>
          <div style="font-size:12.5px;font-weight:600;margin-bottom:6px;">各请求的 Block Table（逻辑块号 → 物理块号）：</div>
          <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;">
            ${snap.seqs
              .map(
                (s) => `
              <div style="border:1.5px solid ${s.color};border-radius:6px;padding:8px 12px;background:#fff;font-size:12px;">
                <div style="font-weight:700;color:${s.color};margin-bottom:4px;">${s.name}</div>
                <div style="font-family:monospace;">block_table: [${s.bt.map((b, i) => `L${i}→#${b}`).join(', ') || '已释放'}]</div>
              </div>
            `
              )
              .join('')}
          </div>
          <div style="font-size:12.5px;font-weight:600;margin-bottom:6px;">GPU 物理 KV 块池（共 8 块，每块 block_size = 4 槽位）：</div>
          <div class="lw-grid-blocks">
            ${blockOwner
              .map(
                (owner, bId) => `
              <div class="lw-block" style="border-color:${owner ? owner.color : '#cbd5e1'};background:${owner ? 'rgba(16,185,129,0.04)' : '#f8fafc'};">
                <div class="lw-block-title">物理块 #${bId}</div>
                <div class="lw-slots">
                  ${[0, 1, 2, 3]
                    .map((slot) => `<span class="lw-slot" style="background:${owner && slot < owner.filled ? owner.color : '#e2e8f0'}"></span>`)
                    .join('')}
                </div>
                <div style="font-size:10px;margin-top:4px;color:#64748b;">${owner ? `归属 Req ${owner.name}` : '空闲 (Free)'}</div>
              </div>
            `
              )
              .join('')}
          </div>
          <div class="lw-callout">${snap.note}</div>
        </div>
      `;

      container.querySelectorAll('[data-st]').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.step = Number(btn.getAttribute('data-st'));
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 7. 1.7 Prefix Cache Matcher (Radix vs Block Hash)
  // =========================================================================
  W.register('prefix-cache-matcher', function (container) {
    const cases = [
      {
        title: '场景 1：标准结构（静态 System + 静态长文档在前，动态问题在后）',
        cached: ['<sys>', '你是助手', '<doc>', '财报第一段', '财报第二段', '财报第三段', '<usr>', '去年利润多少'],
        incoming: ['<sys>', '你是助手', '<doc>', '财报第一段', '财报第二段', '财报第三段', '<usr>', '今年营收多少'],
        desc: '前 7 个词完全相同！RadixAttention（按 Token 树匹配）命中 7/8（87.5%）；vLLM 块哈希（设每块 2 词）前 3 个完整块（6 词）全部命中，仅最后 1 块因未填满或不同需重算！',
      },
      {
        title: '场景 2：反面教材（在 Prompt 最开头塞入动态时间戳！）',
        cached: ['[12:00:01]', '<sys>', '你是助手', '<doc>', '财报第一段', '财报第二段', '财报第三段', '问题A'],
        incoming: ['[12:00:05]', '<sys>', '你是助手', '<doc>', '财报第一段', '财报第二段', '财报第三段', '问题A'],
        desc: '⚠️ 灾难发生！仅仅因为第 0 个词的时间戳变了 4 秒，由于因果注意力中后面所有词的 K/V 都依赖前面的位置与内容，从第 0 个词起链式哈希全部失效，命中率瞬間变成 0%！',
      },
    ];
    let idx = 0;

    function render() {
      const c = cases[idx];
      let matchLen = 0;
      while (matchLen < c.cached.length && matchLen < c.incoming.length && c.cached[matchLen] === c.incoming[matchLen]) {
        matchLen++;
      }
      const blockHitLen = Math.floor(matchLen / 2) * 2;

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Matcher</span>
              <h4 class="lw-title">Prefix Caching 前缀匹配沙盒：为什么动态内容必须放最后？</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${cases.map((item, i) => `<button type="button" class="lw-tab ${i === idx ? 'active' : ''}" data-c="${i}">${item.title}</button>`).join('')}
          </div>
          <div style="font-size:12px;color:#475569;margin-bottom:6px;">已缓存的请求 1 Token 序列：</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            ${c.cached.map((t) => `<span style="padding:4px 8px;background:#e2e8f0;border-radius:4px;font-family:monospace;font-size:12px;">${W.escapeHtml(t)}</span>`).join('')}
          </div>
          <div style="font-size:12px;color:#475569;margin-bottom:6px;">新到达的请求 2 Token 序列（绿色=命中复用 KV，红色=必须重新 Prefill）：</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            ${c.incoming
              .map((t, i) => {
                const hit = i < matchLen;
                return `<span style="padding:4px 8px;background:${hit ? '#10b981' : '#ef4444'};color:#fff;border-radius:4px;font-family:monospace;font-size:12px;font-weight:600;">${W.escapeHtml(t)}</span>`;
              })
              .join('')}
          </div>
          <div class="lw-stats">
            <div class="lw-stat ${matchLen > 0 ? 'ok' : 'err'}">
              <div class="lw-stat-label">RadixAttention (Token 级树匹配)</div>
              <div class="lw-stat-value">${matchLen} / ${c.incoming.length} 命中</div>
            </div>
            <div class="lw-stat ${blockHitLen > 0 ? 'ok' : 'err'}">
              <div class="lw-stat-label">vLLM 块哈希 (设 Block Size = 2)</div>
              <div class="lw-stat-value">${blockHitLen} / ${c.incoming.length} 命中</div>
            </div>
          </div>
          <div class="lw-callout">${c.desc}</div>
        </div>
      `;

      container.querySelectorAll('[data-c]').forEach((btn) => {
        btn.addEventListener('click', () => {
          idx = Number(btn.getAttribute('data-c'));
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 8. 1.8 Framework Comparison Advisor
  // =========================================================================
  W.register('framework-advisor', function (container) {
    const options = [
      {
        name: '场景 A：通用高并发在线 API，追求最快上线与最广模型覆盖',
        rec: 'vLLM',
        cmd: 'vllm serve Qwen/Qwen3-8B --max-model-len 8192',
        reason: '社区生态最大，新模型发布当天通常即支持；默认开启 V1 引擎、Chunked Prefill 与块级 Prefix Caching，开箱即用且可维护性极佳。',
      },
      {
        name: '场景 B：多步 Agent 工作流、高频共享前缀或严格 JSON 结构化输出',
        rec: 'SGLang',
        cmd: 'python -m sglang.launch_server --model-path Qwen/Qwen3-8B --context-length 8192',
        reason: 'RadixAttention 基数树对多轮 Agent 分支前缀复用极强，内置基于有限状态机（FSM）的零开销正则/JSON 约束解码，支持多硬件后端。',
      },
      {
        name: '场景 C：全栈 NVIDIA H100/B200 集群，追求极致压榨单 Token 成本',
        rec: 'TensorRT-LLM',
        cmd: 'trtllm-serve Qwen/Qwen3-8B --tp_size 1',
        reason: '深度绑定 NVIDIA 底层 CUDA / TensorRT 编译器与 FP8/NVFP4 算子，在固定 NVIDIA 硬件上往往能压榨出最高吞吐，适合有专职推理工程团队的深入调优。',
      },
      {
        name: '场景 D：MacBook / CPU 服务器 / 边缘端本地低并发私有部署',
        rec: 'llama.cpp (或 Ollama)',
        cmd: './llama-server -m qwen3-8b-q4_k_m.gguf -c 4096 --port 8000',
        reason: '纯 C/C++ 无重型依赖，支持 Apple Metal、CPU AVX 与 GGUF 2~8 bit 量化，单用户低并发下延迟极低、内存占用极小。',
      },
    ];
    let cur = 0;

    function render() {
      const o = options[cur];
      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Advisor</span>
              <h4 class="lw-title">推理框架选型决策器（vLLM vs SGLang vs TRT-LLM vs llama.cpp）</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${options.map((item, i) => `<button type="button" class="lw-tab ${i === cur ? 'active' : ''}" data-f="${i}">${item.name}</button>`).join('')}
          </div>
          <div class="lw-stats">
            <div class="lw-stat ok">
              <div class="lw-stat-label">首选推荐框架</div>
              <div class="lw-stat-value" style="color:#047857;">${o.rec}</div>
              <div class="lw-stat-sub" style="margin-top:6px;line-height:1.5;">${o.reason}</div>
            </div>
          </div>
          <div class="lw-terminal">
            <div># 推荐启动命令示例：</div>
            <div class="hl-green">$ ${W.escapeHtml(o.cmd)}</div>
          </div>
        </div>
      `;

      container.querySelectorAll('[data-f]').forEach((btn) => {
        btn.addEventListener('click', () => {
          cur = Number(btn.getAttribute('data-f'));
          render();
        });
      });
    }
    render();
  });

  // =========================================================================
  // 9. Project 1 MiniServe Flattened Batch & slot_mapping Sandbox
  // =========================================================================
  W.register('miniserve-batch-sandbox', function (container) {
    const cases = [
      {
        title: '混合步示例：Seq 1 正在做第 2 块 Chunked Prefill (算 2 词) + Seq 2 正在 Decode (算 1 词)',
        desc: 'MiniServe 的 _prepare_inputs 把两个请求本步要算的新词拍平成一维张量（共 3 行）送入模型，并用 slot_mapping 告诉每行的新 K/V 写到物理池的哪个格子里，用 sample_rows 挑出算完当前全部词的行去采样下一个 Token！',
        inputIds: '[105, 208, 991]   # 前 2 个属于 Seq 1 的 Prefill 块，最后 1 个属于 Seq 2 的 Decode 词',
        positions: '[2, 3, 5]         # Seq 1 之前已算过 2 个词(pos 0,1)，本步是 pos 2,3；Seq 2 本步是 pos 5',
        slotMapping: '[10, 11, 29]      # slot = block_table[pos // 4] * 4 + (pos % 4)',
        sampleRows: '[1, 2]            # 扁平张量的第 1 行（Seq 1 的末尾词）和第 2 行（Seq 2 的末尾词）进入 LM Head',
      },
      {
        title: '未完成切块示例：Seq 1 很长（共 6 词），本步只算前 3 词（未算完 Prompt）',
        desc: '注意！因为 Seq 1 的 Prompt 一共 6 个词，本步只算了前 3 个词（num_computed_tokens 从 0 变成 3，还没追平 num_tokens=6），所以这一步结束时 Seq 1 不能采样新词，sample_rows 里不包含 Seq 1！',
        inputIds: '[101, 102, 103, 888] # 前 3 个属于 Seq 1 的第一块，最后 1 个属于 Seq 2 的 Decode',
        positions: '[0, 1, 2, 4]         # Seq 1 位置 0,1,2；Seq 2 位置 4',
        slotMapping: '[4, 5, 6, 16]        # 各自映射到物理块槽位',
        sampleRows: '[3]                  # 只有第 3 行（Seq 2）完成了全部已有词，参与 LM Head 采样！',
      },
    ];
    let cIdx = 0;

    function render() {
      const c = cases[cIdx];
      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Code Walkthrough Sandbox</span>
              <h4 class="lw-title">MiniServe 核心难点突破：<code>_prepare_inputs</code> 扁平化张量拼接沙盒</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${cases.map((item, i) => `<button type="button" class="lw-tab ${i === cIdx ? 'active' : ''}" data-idx="${i}">${item.title}</button>`).join('')}
          </div>
          <div class="lw-terminal">
            <div><span class="hl-blue">input_ids   </span> = ${c.inputIds}</div>
            <div><span class="hl-green">positions   </span> = ${c.positions}</div>
            <div><span class="hl-amber">slot_mapping</span> = ${c.slotMapping}</div>
            <div><span class="hl-red">sample_rows </span> = ${c.sampleRows}</div>
          </div>
          <div class="lw-callout">${c.desc}</div>
        </div>
      `;

      container.querySelectorAll('.lw-tab').forEach((btn, i) => {
        btn.addEventListener('click', () => {
          cIdx = i;
          render();
        });
      });
    }
    render();
  });
})();
