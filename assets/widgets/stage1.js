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
      { name: 'Token Embedding', inShape: (T) => `[${T}] (Token IDs)`, outShape: (T) => `[${T}, 4096]`, desc: '查词表矩阵 [151936, 4096]，把每个整数 ID 映射成 4096 维浮点数向量。' },
      { name: 'QKV 线性投影', inShape: (T) => `[${T}, 4096]`, outShape: (T) => `Q:[${T}, 32, 128], K/V:[${T}, 8, 128]`, desc: 'Qwen3-8B 采用 GQA：32 个 Query 头（总维 4096），但只有 8 个 K/V 头（总维 1024），每 4 个 Q 头共享 1 组 K/V。' },
      { name: 'RoPE + 因果自注意力', inShape: (T) => `Q:[${T}, 32, 128], K/V:[${T}, 8, 128]`, outShape: (T) => `[${T}, 4096]`, desc: '先按位置旋转 Q、K（RoPE），再做因果掩码下的打分 softmax(QK^T / √128)V，最后经 W_O 投影回 4096 维。注意：计算量随序列长度二次方或线性增长。' },
      { name: 'SwiGLU 前馈网络 (MLP)', inShape: (T) => `[${T}, 4096]`, outShape: (T) => `[${T}, 4096]`, desc: '先用 Gate 和 Up 两个矩阵升维到 [T, 12288]，逐元素相乘后再经 Down 矩阵降回 [T, 4096]。占全模型近 78% 的权重参数。' },
      { name: 'LM Head (仅取最后 1 行)', inShape: (T) => `[${T}, 4096] → 切片 [1, 4096]`, outShape: () => `[1, 151936] (Logits)`, desc: '生成下一个词只依赖最后一个位置的隐藏状态。只对最后 1 行做词表投影，省去 (T-1) 行的无用计算与数 GB 显存。' },
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
          <div style="margin-top:14px;font-size:12.5px;font-weight:600;color:var(--ink-2);">
            Decode 单步计算量构成（上下文 $T = ${T}$ 时，线性层固定 ${linearGflops} GFLOPs vs 注意力 ${attnGflops.toFixed(1)} GFLOPs，注意力占 ${attnPct}%）：
          </div>
          <div class="lw-stack-bar">
            <div class="lw-stack-seg" style="width:${100 - attnPct}%;background:var(--ink-2);">线性层 (MLP+QKV+O): ${linearGflops} GFLOPs</div>
            <div class="lw-stack-seg" style="width:${attnPct}%;background:var(--decode);">${attnPct > 12 ? 'Attention: ' + attnGflops.toFixed(1) + ' GFLOPs' : ''}</div>
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
                    <div style="width:95px;font-family:monospace;font-size:12.5px;font-weight:700;">"${c.token}" <span style="color:var(--ink-3);font-weight:400;">(${c.logit})</span></div>
                    <div style="flex:1;height:20px;background:var(--rule);border-radius:4px;overflow:hidden;">
                      <div style="height:100%;width:${(finalP * 100).toFixed(1)}%;background:${c.kept ? 'var(--prefill)' : 'var(--rule-strong)'};transition:width 0.2s;"></div>
                    </div>
                    <div style="width:130px;font-family:monospace;font-size:12px;text-align:right;">
                      ${c.kept ? `<strong>${(finalP * 100).toFixed(1)}%</strong> (原始 ${(c.prob * 100).toFixed(1)}%)` : '<span style="color:var(--err);">已截断淘汰</span>'}
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
            <div style="border:1px solid var(--err-soft);background:var(--err-soft);border-radius:8px;padding:12px;">
              <div style="font-weight:700;font-size:13px;color:var(--err);margin-bottom:8px;">无 KV Cache（每步把全部 ${curLen} 个词重新送入模型）</div>
              ${activeTokens
                .map(
                  (t) => `<div style="background:var(--err-soft);color:var(--err);padding:5px 10px;border-radius:4px;margin-bottom:4px;font-size:12px;font-weight:600;">
                  重新计算 Q, K, V：${t}
                </div>`
                )
                .join('')}
              <div style="font-size:12px;color:var(--err);margin-top:8px;">本步计算行数：<strong>${curLen} 行</strong> ｜ 累计已算：<strong>${noCacheFlops} 行</strong></div>
            </div>
            <div style="border:1px solid var(--ok-soft);background:var(--ok-soft);border-radius:8px;padding:12px;">
              <div style="font-weight:700;font-size:13px;color:var(--ok);margin-bottom:8px;">开启 KV Cache（只送最新词，直接读取历史 K、V）</div>
              ${activeTokens
                .map((t, idx) => {
                  const isNew = state.step === 0 || idx === curLen - 1;
                  return `<div style="background:${isNew ? 'var(--decode-soft)' : 'var(--ok-soft)'};color:var(--ok);padding:5px 10px;border-radius:4px;margin-bottom:4px;font-size:12px;font-weight:600;">
                    ${isNew ? '本步新算 Q, K, V 并写入缓存：' : '从显存直接复用 K, V（Q 已丢弃）：'}${t}
                  </div>`;
                })
                .join('')}
              <div style="font-size:12px;color:var(--ok);margin-top:8px;">本步仅算：<strong>${state.step === 0 ? 3 : 1} 行</strong> ｜ 累计仅算：<strong>${withCacheFlops} 行</strong></div>
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
      { title: '客户端发起并发 SSE 请求', detail: '用户 A 和用户 B 同时请求 /v1/completions(stream=true)。FastAPI 为每个请求创建一个专属的 asyncio.Queue，并将请求封装成 (seq, queue) 放入线程安全的 inbox 队列。', activeBox: 0 },
      { title: '后台引擎线程从 inbox 取任务', detail: '独立的后台线程 AsyncEngine._loop 被唤醒，把 inbox 里的新请求全部取出并注册进 Scheduler.waiting 队列，同时记住 seq_id → asyncio.Queue 的映射。', activeBox: 1 },
      { title: '执行一步 engine.step() 合批推理', detail: 'Scheduler 将 A 和 B 组成一个 Batch，GPU 一次前向传播同时算出 A 的新词 "Hello" 和 B 的新词 "你好"。', activeBox: 2 },
      { title: '跨线程精准投递回各自的协程队列', detail: '后台线程调用 loop.call_soon_threadsafe(q.put_nowait, out)，把 "Hello" 塞进 A 的 asyncio.Queue，把 "你好" 塞进 B 的 asyncio.Queue，两边互不串台、立刻通过 SSE 推给浏览器。', activeBox: 3 },
    ];
    let step = 0;

    function render() {
      const s = stages[step];
      const boxes = ['FastAPI 路由协程\n(创建专属 asyncio.Queue)', '线程安全 inbox 队列\n(跨线程桥接)', '后台线程 LLMEngine.step()\n(GPU 合批前向)', 'call_soon_threadsafe\n(逐 Token SSE 推送)'];

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
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
              <div style="padding:12px;border-radius:8px;border:1px solid ${idx === s.activeBox ? 'var(--prefill)' : 'var(--rule)'};background:${idx === s.activeBox ? 'var(--prefill-soft)' : 'var(--paper)'};text-align:center;font-size:12.5px;font-weight:600;white-space:pre-line;">
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
    // 用与 MiniServe 相同的规则逐步模拟 5 个请求。代价模型：每一步固定 1 个时间单位（读一遍权重），
    // 每个 prefill token 额外 1/8 个单位（prefill 受算力限制，耗时随 token 数增长）。
    const MAX_SEQS = 3;
    const PREFILL_COST = 1 / 8;
    const BASE = [
      { id: 'R1', arrive: 0, prompt: 6, out: 4 },
      { id: 'R2', arrive: 0, prompt: 8, out: 10 },
      { id: 'R3', arrive: 0, prompt: 5, out: 14 },
      { id: 'R4', arrive: 3, prompt: 32, out: 6, long: true },
      { id: 'R5', arrive: 6, prompt: 6, out: 6 },
    ];
    const POLICIES = {
      static: '静态批处理',
      nochunk: '连续批处理，不切块',
      chunked: '连续批处理 + 切块',
    };
    const state = { policy: 'nochunk', budget: 12, longPrompt: 32 };

    function simulate(policy) {
      const reqs = BASE.map((r) => ({
        ...r, prompt: r.long ? state.longPrompt : r.prompt,
        queued: false, running: false, done: false, computed: 0, generated: 0, tokenTimes: [], segs: [],
      }));
      const waiting = [];
      const running = [];
      const total = (r) => r.prompt + r.generated;
      let t = 0;
      for (let guard = 0; guard < 2000 && reqs.some((r) => !r.done); guard++) {
        reqs.forEach((r) => { if (!r.queued && r.arrive <= t + 1e-9) { r.queued = true; waiting.push(r); } });
        if (!waiting.length && !running.length) {
          t = Math.min(...reqs.filter((r) => !r.queued).map((r) => r.arrive));
          continue;
        }
        const sched = [];
        const admit = (budget, partial) => {
          while (waiting.length && running.length < MAX_SEQS && budget > 0) {
            const r = waiting[0];
            const need = total(r) - r.computed;
            if (!partial && need > budget) break;
            waiting.shift(); running.push(r); r.running = true;
            const n = Math.min(need, budget);
            sched.push([r, n]); budget -= n;
          }
          return budget;
        };
        if (policy === 'static') {
          if (!running.length) admit(Infinity, false);
          else running.forEach((r) => sched.push([r, 1]));
        } else if (policy === 'nochunk') {
          if (waiting.length && running.length < MAX_SEQS) admit(Infinity, false);
          else running.forEach((r) => sched.push([r, 1]));
        } else {
          let budget = state.budget;
          for (const r of running) {
            if (budget <= 0) break;
            const n = Math.min(total(r) - r.computed, budget);
            sched.push([r, n]); budget -= n;
          }
          admit(budget, true);
        }
        const prefillTokens = sched.reduce((acc, [r, n]) => acc + (r.computed < r.prompt ? n : 0), 0);
        const dur = 1 + prefillTokens * PREFILL_COST;
        const inStep = new Map(sched);
        reqs.forEach((r) => {
          if (!r.queued || r.done) return;
          let kind = 'wait';
          if (inStep.has(r)) kind = r.computed < r.prompt ? 'prefill' : 'decode';
          else if (r.running) kind = 'stall';
          const last = r.segs[r.segs.length - 1];
          if (last && last.kind === kind && Math.abs(last.end - t) < 1e-9) last.end = t + dur;
          else r.segs.push({ kind, start: t, end: t + dur });
        });
        t += dur;
        sched.forEach(([r, n]) => {
          r.computed += n;
          if (r.computed === total(r)) {
            r.generated += 1;
            r.tokenTimes.push(t);
            if (r.generated >= r.out) {
              r.done = true; r.running = false;
              running.splice(running.indexOf(r), 1);
            }
          }
        });
      }
      const ttft = reqs.map((r) => r.tokenTimes[0] - r.arrive);
      const itl = reqs.map((r) => r.tokenTimes.slice(1).reduce((m, x, i) => Math.max(m, x - r.tokenTimes[i]), 0));
      const outTokens = reqs.reduce((a, r) => a + r.out, 0);
      return { reqs, makespan: t, ttft, itl, throughput: outTokens / t };
    }

    function render() {
      const sims = Object.fromEntries(Object.keys(POLICIES).map((k) => [k, simulate(k)]));
      const sim = sims[state.policy];
      const scale = Math.max(...Object.values(sims).map((x) => x.makespan));
      const pct = (v) => ((v / scale) * 100).toFixed(2) + '%';
      const f = (v) => v.toFixed(1);
      const meanTTFT = sim.ttft.reduce((a, b) => a + b, 0) / sim.ttft.length;
      const maxITL = Math.max(...sim.itl);
      const worst = sim.reqs[sim.itl.indexOf(maxITL)];
      const r4 = sim.reqs.findIndex((r) => r.long);

      let note;
      if (state.policy === 'static') {
        note = `R4 在第 ${BASE[3].arrive} 个时间单位到达，但要等前一批三个请求全部结束才能开始，TTFT 为 ${f(sim.ttft[r4])}。R1 早早结束后，它的位置一直空着。`;
      } else if (state.policy === 'nochunk') {
        note = `R4 的 ${state.longPrompt} 个 prompt token 在一步内算完，这一步耗时 ${f(1 + state.longPrompt * PREFILL_COST)} 个单位，正在 decode 的请求在这一步停顿（红色），最大 ITL 达到 ${f(maxITL)}（${worst.id}）。`;
      } else {
        note = `每步最多处理 ${state.budget} 个 token，R4 的 prompt 被切成 ${Math.ceil(state.longPrompt / state.budget)} 块左右，与其他请求的 decode 同步进行。最大 ITL 降到 ${f(maxITL)}，代价是 R4 的 TTFT 为 ${f(sim.ttft[r4])}。调小预算，ITL 更平稳、TTFT 更长。`;
      }

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header"><h4 class="lw-title">三种批处理策略的逐步模拟</h4></div>
          <p class="lw-subtitle">5 个请求，最多同时运行 3 个。R4 的 prompt 很长。横轴是时间：每一步固定 1 个单位，每个 prefill token 再加 1/8 个单位。</p>
          <div class="lw-tabs">
            ${Object.entries(POLICIES).map(([k, v]) => `<button type="button" class="lw-tab ${state.policy === k ? 'active' : ''}" data-p="${k}">${v}</button>`).join('')}
          </div>
          <div class="lw-controls">
            <div class="lw-control">
              <label>每步 token 预算（仅切块模式） <span class="lw-val">${state.budget}</span></label>
              <input type="range" min="4" max="32" step="1" value="${state.budget}" data-k="budget" ${state.policy === 'chunked' ? '' : 'disabled'} />
            </div>
            <div class="lw-control">
              <label>R4 的 prompt 长度 <span class="lw-val">${state.longPrompt}</span></label>
              <input type="range" min="8" max="64" step="4" value="${state.longPrompt}" data-k="longPrompt" />
            </div>
          </div>
          <div class="bt-rows">
            ${sim.reqs.map((r, i) => `
              <div class="bt-row">
                <div class="bt-label">${r.id}<span>${r.prompt} → ${r.out}</span></div>
                <div class="bt-track">
                  <i class="bt-arrive" style="left:${pct(r.arrive)}"></i>
                  ${r.segs.map((sg) => `<i class="bt-seg is-${sg.kind}" style="left:${pct(sg.start)};width:${pct(sg.end - sg.start)}"></i>`).join('')}
                </div>
                <div class="bt-num">${f(sim.ttft[i])}<span>/</span>${f(sim.itl[i])}</div>
              </div>`).join('')}
            <div class="bt-row bt-head"><div></div><div class="bt-axis"><span>0</span><span>${f(scale)}</span></div><div class="bt-num">TTFT / 最大 ITL</div></div>
          </div>
          <div class="lw-legend">
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--prefill)"></i>prefill</span>
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--decode)"></i>decode</span>
            <span class="lw-legend-item"><i class="lw-dot bt-dot-wait"></i>排队</span>
            <span class="lw-legend-item"><i class="lw-dot" style="background:var(--err-soft);outline:1px solid var(--err)"></i>已在运行，但这一步没被调度</span>
          </div>
          <div class="lw-stats">
            <div class="lw-stat info"><div class="lw-stat-label">平均 TTFT</div><div class="lw-stat-value">${f(meanTTFT)}</div></div>
            <div class="lw-stat ${maxITL > 2.5 ? 'warn' : 'ok'}"><div class="lw-stat-label">最大 ITL</div><div class="lw-stat-value">${f(maxITL)}</div></div>
            <div class="lw-stat"><div class="lw-stat-label">全部完成用时</div><div class="lw-stat-value">${f(sim.makespan)}</div></div>
            <div class="lw-stat"><div class="lw-stat-label">吞吐（token / 单位时间）</div><div class="lw-stat-value">${sim.throughput.toFixed(2)}</div></div>
          </div>
          <div class="lw-callout">${note}</div>
        </div>`;

      container.querySelectorAll('[data-p]').forEach((btn) => btn.addEventListener('click', () => { state.policy = btn.dataset.p; render(); }));
      container.querySelectorAll('input[data-k]').forEach((inp) => inp.addEventListener('input', () => { state[inp.dataset.k] = Number(inp.value); render(); }));
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
        note: '每块容量 4 tokens。Req A 用了 2 个物理块（块 #1 满 4 个 + 块 #4 用 2 个，内部碎片仅 2 槽位）；Req B 用了 2 个物理块（块 #2 满 + 块 #6 用 1 个）。物理块完全不需要连续。',
        seqs: [
          { name: 'Req A (6 tokens)', color: 'var(--series-1)', bt: [1, 4], usedInLast: 2 },
          { name: 'Req B (5 tokens)', color: 'var(--series-2)', bt: [2, 6], usedInLast: 1 },
        ],
      },
      {
        label: '状态 2：Req C (9 tok) 到达，按需分配 3 个物理块',
        note: 'Req C 需要 ⌈9/4⌉ = 3 个块，分配器从空闲池弹出 #0、#3、#5。此时 8 个物理块已用 7 个，仅剩 #7 空闲。传统连续预留若按 max_len=16 分配，这 3 个请求早就在第 1 步 OOM 了。',
        seqs: [
          { name: 'Req A (6 tokens)', color: 'var(--series-1)', bt: [1, 4], usedInLast: 2 },
          { name: 'Req B (5 tokens)', color: 'var(--series-2)', bt: [2, 6], usedInLast: 1 },
          { name: 'Req C (9 tokens)', color: 'var(--series-3)', bt: [0, 3, 5], usedInLast: 1 },
        ],
      },
      {
        label: '状态 3：继续生成导致物理块耗尽 → 触发队尾抢占 (Preempt Req C)',
        note: '当 Req A 和 Req B 继续增长需要新块而空闲池为空时，调度器暂停队尾的 Req C，释放它的物理块 #0、#3、#5，将其 num_computed_tokens 清零放回 waiting 队列，优先保住排在前面的 Req A 和 Req B 完成生成。',
        seqs: [
          { name: 'Req A (9 tokens, 扩块 #7)', color: 'var(--series-1)', bt: [1, 4, 7], usedInLast: 1 },
          { name: 'Req B (9 tokens, 扩块 #0)', color: 'var(--series-2)', bt: [2, 6, 0], usedInLast: 1 },
          { name: 'Req C (已抢占放回 Waiting, 释放块)', color: 'var(--ink-3)', bt: [], usedInLast: 0 },
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
              <div class="lw-block" style="border-color:${owner ? owner.color : 'var(--rule-strong)'};background:${owner ? 'var(--prefill-soft)' : 'var(--paper-2)'};">
                <div class="lw-block-title">物理块 #${bId}</div>
                <div class="lw-slots">
                  ${[0, 1, 2, 3]
                    .map((slot) => `<span class="lw-slot" style="background:${owner && slot < owner.filled ? owner.color : 'var(--rule)'}"></span>`)
                    .join('')}
                </div>
                <div style="font-size:10px;margin-top:4px;color:var(--ink-3);">${owner ? `归属 Req ${owner.name}` : '空闲 (Free)'}</div>
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
        desc: '前 7 个词完全相同。RadixAttention（按 Token 树匹配）命中 7/8（87.5%）；vLLM 块哈希（设每块 2 词）前 3 个完整块（6 词）全部命中，仅最后 1 块因未填满或不同需重算。',
      },
      {
        title: '场景 2：反面教材（在 Prompt 最开头塞入动态时间戳。）',
        cached: ['[12:00:01]', '<sys>', '你是助手', '<doc>', '财报第一段', '财报第二段', '财报第三段', '问题A'],
        incoming: ['[12:00:05]', '<sys>', '你是助手', '<doc>', '财报第一段', '财报第二段', '财报第三段', '问题A'],
        desc: '灾难发生。仅仅因为第 0 个词的时间戳变了 4 秒，由于因果注意力中后面所有词的 K/V 都依赖前面的位置与内容，从第 0 个词起链式哈希全部失效，命中率瞬間变成 0%。',
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
              <h4 class="lw-title">Prefix Caching 前缀匹配组件：为什么动态内容必须放最后？</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${cases.map((item, i) => `<button type="button" class="lw-tab ${i === idx ? 'active' : ''}" data-c="${i}">${item.title}</button>`).join('')}
          </div>
          <div style="font-size:12px;color:var(--ink-2);margin-bottom:6px;">已缓存的请求 1 Token 序列：</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            ${c.cached.map((t) => `<span style="padding:4px 8px;background:var(--rule);border-radius:4px;font-family:monospace;font-size:12px;">${W.escapeHtml(t)}</span>`).join('')}
          </div>
          <div style="font-size:12px;color:var(--ink-2);margin-bottom:6px;">新到达的请求 2 Token 序列（绿色=命中复用 KV，红色=必须重新 Prefill）：</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            ${c.incoming
              .map((t, i) => {
                const hit = i < matchLen;
                return `<span style="padding:4px 8px;background:${hit ? 'var(--ok)' : 'var(--err)'};color:#fff;border-radius:4px;font-family:monospace;font-size:12px;font-weight:600;">${W.escapeHtml(t)}</span>`;
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
              <h4 class="lw-title">推理框架选型决策器（vLLM vs SGLang vs TRT-LLM vs llama.cpp）</h4>
            </div>
          </div>
          <div class="lw-tabs">
            ${options.map((item, i) => `<button type="button" class="lw-tab ${i === cur ? 'active' : ''}" data-f="${i}">${item.name}</button>`).join('')}
          </div>
          <div class="lw-stats">
            <div class="lw-stat ok">
              <div class="lw-stat-label">首选推荐框架</div>
              <div class="lw-stat-value" style="color:var(--ok);">${o.rec}</div>
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
        desc: 'MiniServe 的 _prepare_inputs 把两个请求本步要算的新词拍平成一维张量（共 3 行）送入模型，并用 slot_mapping 告诉每行的新 K/V 写到物理池的哪个格子里，用 sample_rows 挑出算完当前全部词的行去采样下一个 Token。',
        inputIds: '[105, 208, 991]   # 前 2 个属于 Seq 1 的 Prefill 块，最后 1 个属于 Seq 2 的 Decode 词',
        positions: '[2, 3, 5]         # Seq 1 之前已算过 2 个词(pos 0,1)，本步是 pos 2,3；Seq 2 本步是 pos 5',
        slotMapping: '[10, 11, 29]      # slot = block_table[pos // 4] * 4 + (pos % 4)',
        sampleRows: '[1, 2]            # 扁平张量的第 1 行（Seq 1 的末尾词）和第 2 行（Seq 2 的末尾词）进入 LM Head',
      },
      {
        title: '未完成切块示例：Seq 1 很长（共 6 词），本步只算前 3 词（未算完 Prompt）',
        desc: '注意。因为 Seq 1 的 Prompt 一共 6 个词，本步只算了前 3 个词（num_computed_tokens 从 0 变成 3，还没追平 num_tokens=6），所以这一步结束时 Seq 1 不能采样新词，sample_rows 里不包含 Seq 1。',
        inputIds: '[101, 102, 103, 888] # 前 3 个属于 Seq 1 的第一块，最后 1 个属于 Seq 2 的 Decode',
        positions: '[0, 1, 2, 4]         # Seq 1 位置 0,1,2；Seq 2 位置 4',
        slotMapping: '[4, 5, 6, 16]        # 各自映射到物理块槽位',
        sampleRows: '[3]                  # 只有第 3 行（Seq 2）完成了全部已有词，参与 LM Head 采样。',
      },
    ];
    let cIdx = 0;

    function render() {
      const c = cases[cIdx];
      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <h4 class="lw-title">MiniServe 核心难点突破：<code>_prepare_inputs</code> 扁平化张量拼接组件</h4>
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
