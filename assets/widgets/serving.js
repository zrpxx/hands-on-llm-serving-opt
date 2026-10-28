/**
 * 第 1 部分的交互组件
 * - engine-loop-steps    1.1 请求在事件循环与引擎线程之间的流动
 * - batching-timeline    1.2 三种批处理策略的逐步模拟
 * - paged-kv             1.3 分页 KV Cache 的分配、释放与抢占
 * - prefix-match         1.4 块哈希前缀缓存的命中
 * - prepare-inputs       项目 1：把一批序列展平成模型输入
 *
 * 写法约定同 basics.js：控件只渲染一次，交互时只更新结果区域。
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  // ------------------------------------------------------------------ 1.1 引擎循环
  W.register('engine-loop-steps', function (container) {
    // 三个请求：A 输出 4 个 token，B 输出 3 个，C 本应输出 6 个，但客户端在收到 2 个之后断开
    const STEPS = [
      { inbox: ['add A'], run: [['A', 'prefill']], out: [['A', 1]], note: 'A 到达。引擎原本阻塞在 inbox.get() 上，被唤醒后取出命令，A 做 prefill，得到第 1 个 token。' },
      { inbox: ['add C'], run: [['A', 'decode'], ['C', 'prefill']], out: [['A', 2], ['C', 1]], note: 'C 在上一步进行期间到达，这一步开始前被取出。A 的 decode 和 C 的 prefill 在同一次前向计算里完成。' },
      { inbox: ['add B'], run: [['A', 'decode'], ['C', 'decode'], ['B', 'prefill']], out: [['A', 3], ['C', 2], ['B', 1]], note: 'B 加入。三个序列同批。这一步结束后，C 的用户关掉了页面：协程被取消，finally 向 inbox 放入 abort C。' },
      { inbox: ['abort C'], run: [['A', 'decode'], ['B', 'decode']], out: [['A', 4, true], ['B', 2]], note: '引擎取出 abort C，调度器把 C 移出运行队列并释放它的 KV 块。A 生成了第 4 个 token，结束，它的队列从 streams 中删除。' },
      { inbox: [], run: [['B', 'decode']], out: [['B', 3, true]], note: '只剩 B。它生成第 3 个 token 后结束。' },
      { inbox: [], run: [], out: [], note: '没有未完成的序列。引擎线程阻塞在 inbox.get() 上，不占用 CPU，直到下一个请求到来。' },
    ];
    const TOTAL = { A: 4, B: 3, C: 6 };
    let k = 0;

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">引擎循环逐步演示</h4>
          <div class="lw-btn-row" style="margin:0">
            <button type="button" class="lw-btn" data-prev>上一步</button>
            <button type="button" class="lw-btn primary" data-next>下一步</button>
          </div>
        </div>
        <div data-out></div>
      </div>`;
    const out = container.querySelector('[data-out]');

    const chip = (text, kind) => {
      const s = {
        cmd: 'background:var(--paper-2);border-color:var(--rule-strong);color:var(--ink)',
        prefill: 'background:var(--prefill-soft);border-color:var(--prefill);color:var(--prefill-ink)',
        decode: 'background:var(--decode-soft);border-color:var(--decode);color:var(--decode-ink)',
        abort: 'background:var(--err-soft);border-color:var(--err);color:var(--ink)',
      }[kind];
      return `<span style="display:inline-block;border:1px solid;border-radius:4px;padding:2px 8px;margin:2px 4px 2px 0;font:600 0.8rem var(--font-mono);${s}">${text}</span>`;
    };

    function update() {
      const s = STEPS[k];
      const got = { A: 0, B: 0, C: 0 };
      const ended = {};
      for (let i = 0; i <= k; i++) STEPS[i].out.forEach(([id, n, fin]) => { got[id] = n; if (fin) ended[id] = '已完成'; });
      if (k >= 2) ended.C = '已断开';
      const client = (id) => {
        const n = got[id];
        const boxes = Array.from({ length: TOTAL[id] }, (_, i) => `<span style="display:inline-block;width:14px;height:14px;margin-right:3px;border-radius:2px;background:${i < n ? W.color(id === 'C' ? 'ink-3' : 'prefill') : W.color('paper-3')}"></span>`).join('');
        return `<div style="display:flex;align-items:center;gap:10px;font-size:0.84rem;margin:4px 0"><span style="width:4.5em;font-weight:600">请求 ${id}</span>${boxes}<span style="color:var(--ink-3)">${ended[id] || (n ? '接收中' : '未开始')}</span></div>`;
      };
      out.innerHTML = `
        <div class="lw-subtitle" style="margin:0 0 10px">第 ${k + 1} 轮循环</div>
        <div style="display:grid;grid-template-columns:9em 1fr;gap:8px 12px;font-size:0.86rem;align-items:center">
          <div style="color:var(--ink-2)">从 inbox 取出</div><div>${s.inbox.length ? s.inbox.map((c) => chip(c, c.startsWith('abort') ? 'abort' : 'cmd')).join('') : '<span style="color:var(--ink-3)">无</span>'}</div>
          <div style="color:var(--ink-2)">参加这一步的序列</div><div>${s.run.length ? s.run.map(([id, ph]) => chip(`${id} ${ph}`, ph)).join('') : '<span style="color:var(--ink-3)">无，引擎空闲</span>'}</div>
          <div style="color:var(--ink-2)">投递到队列</div><div>${s.out.length ? s.out.map(([id, n, fin]) => chip(`${id} 第 ${n} 个${fin ? '，结束' : ''}`, 'cmd')).join('') : '<span style="color:var(--ink-3)">无</span>'}</div>
        </div>
        <div style="margin:14px 0 4px;padding-top:10px;border-top:1px solid var(--rule)">${['A', 'B', 'C'].map(client).join('')}</div>
        <div class="lw-callout">${s.note}</div>`;
      container.querySelector('[data-prev]').disabled = k === 0;
      container.querySelector('[data-next]').disabled = k === STEPS.length - 1;
    }

    container.querySelector('[data-prev]').addEventListener('click', () => { if (k > 0) { k--; update(); } });
    container.querySelector('[data-next]').addEventListener('click', () => { if (k < STEPS.length - 1) { k++; update(); } });
    update();
  });

  // ------------------------------------------------------------------ 1.2 批处理时间线
  W.register('batching-timeline', function (container) {
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

    function update() {
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

      container.querySelectorAll('[data-p]').forEach((b) => b.classList.toggle('active', b.dataset.p === state.policy));
      container.querySelector('[data-v=budget]').textContent = state.budget;
      container.querySelector('[data-v=longPrompt]').textContent = state.longPrompt;
      container.querySelector('input[data-k=budget]').disabled = state.policy !== 'chunked';
      container.querySelector('[data-out]').innerHTML = `
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
`;
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">三种批处理策略的逐步模拟</h4></div>
        <p class="lw-subtitle">5 个请求，最多同时运行 3 个，R4 的 prompt 很长。横轴是时间：每一步固定 1 个单位，每个 prefill token 再加 1/8 个单位。</p>
        <div class="lw-tabs">
          ${Object.entries(POLICIES).map(([k, v]) => `<button type="button" class="lw-tab" data-p="${k}">${v}</button>`).join('')}
        </div>
        <div class="lw-controls">
          <div class="lw-control">
            <label>每步 token 预算（仅切块时有效）<span class="lw-val" data-v="budget"></span></label>
            <input type="range" min="4" max="32" step="1" value="${state.budget}" data-k="budget" />
          </div>
          <div class="lw-control">
            <label>R4 的 prompt 长度<span class="lw-val" data-v="longPrompt"></span></label>
            <input type="range" min="8" max="64" step="4" value="${state.longPrompt}" data-k="longPrompt" />
          </div>
        </div>
        <div data-out></div>
      </div>`;

    container.querySelectorAll('[data-p]').forEach((btn) => btn.addEventListener('click', () => { state.policy = btn.dataset.p; update(); }));
    container.querySelectorAll('input[data-k]').forEach((inp) => inp.addEventListener('input', () => { state[inp.dataset.k] = Number(inp.value); update(); }));
    update();
  });


  // ------------------------------------------------------------------ 1.3 分页 KV
  // 调度规则与 MiniServe 的 _schedule_continuous 相同，参数与 1.3 节 trace 实验一致。
  W.register('paged-kv', function (container) {
    const B = 4, NUM_BLOCKS = 8, BUDGET = 16, MAX_TOKENS = 10;
    const SPEC = [
      { id: 'A', prompt: 6, arrive: 1, c: 'prefill' },
      { id: 'B', prompt: 5, arrive: 1, c: 'decode' },
      { id: 'C', prompt: 9, arrive: 3, c: 'series-2' },
    ];

    function run(maxStep) {
      const free = Array.from({ length: NUM_BLOCKS }, (_, i) => i);
      const seqs = SPEC.map((s) => ({ ...s, num: s.prompt, computed: 0, gen: 0, bt: [], status: 'new', preempts: 0 }));
      const waiting = [], running = [];
      let log = [];
      for (let step = 1; step <= maxStep; step++) {
        log = [];
        seqs.forEach((s) => { if (s.status === 'new' && s.arrive === step) { s.status = 'waiting'; waiting.push(s); log.push(`${s.id} 到达`); } });
        const need = (s, n) => Math.max(0, Math.ceil((s.computed + n) / B) - s.bt.length);
        const sched = [];
        let budget = BUDGET, preempted = 0;
        for (let i = 0; i < running.length && budget > 0;) {
          const s = running[i];
          const n = Math.min(s.num - s.computed, budget);
          let gone = false;
          while (need(s, n) > free.length) {
            const v = running[running.length - 1];
            free.push(...v.bt); v.bt = []; v.computed = 0; v.status = 'waiting'; v.preempts++;
            running.pop(); waiting.unshift(v); preempted++;
            log.push(`空闲块不够，抢占最晚加入的 ${v.id}，释放它的全部块`);
            if (v === s) { gone = true; break; }
          }
          if (gone) break;
          for (let k = need(s, n); k > 0; k--) s.bt.push(free.shift());
          sched.push([s, n]); budget -= n; i++;
        }
        if (!preempted) {
          while (waiting.length && budget > 0) {
            const s = waiting[0];
            const n = Math.min(s.num - s.computed, budget);
            if (need(s, n) > free.length) { log.push(`${s.id} 需要 ${need(s, n)} 个块，只剩 ${free.length} 个，继续等待`); break; }
            waiting.shift(); running.push(s); s.status = 'running';
            for (let k = need(s, n); k > 0; k--) s.bt.push(free.shift());
            sched.push([s, n]); budget -= n;
            log.push(s.preempts ? `${s.id} 重新加入，把已有的 ${s.num} 个 token 全部重新 prefill` : `接收 ${s.id}`);
          }
        }
        sched.forEach(([s, n]) => {
          s.computed += n;
          s.lastN = n;
          if (s.computed === s.num) {
            s.gen++; s.num++;
            if (s.gen >= MAX_TOKENS) {
              free.push(...s.bt); s.doneBt = s.bt; s.bt = []; s.status = 'finished';
              running.splice(running.indexOf(s), 1);
              log.push(`${s.id} 生成满 ${MAX_TOKENS} 个 token，结束，释放全部块`);
            }
          }
        });
        seqs.forEach((s) => { s.active = sched.some(([x]) => x === s); });
      }
      return { seqs, free, log, done: seqs.every((s) => s.status === 'finished') };
    }

    let step = 1;
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">分页 KV Cache：分配、释放与抢占</h4>
          <div class="lw-btn-row" style="margin:0">
            <button type="button" class="lw-btn" data-prev>上一步</button>
            <button type="button" class="lw-btn primary" data-next>下一步</button>
          </div>
        </div>
        <p class="lw-subtitle">8 个物理块，每块 4 个 token。A、B 的 prompt 长度为 6 和 5，在第 1 步到达；C 的 prompt 长度为 9，在第 3 步到达。每个序列生成 10 个 token。每步 token 预算 16。</p>
        <div data-out></div>
      </div>`;
    const out = container.querySelector('[data-out]');

    function update() {
      const r = run(step);
      const owner = {};
      r.seqs.forEach((s) => s.bt.forEach((b, i) => { owner[b] = { s, filled: Math.max(0, Math.min(B, s.computed - i * B)) }; }));
      const grid = Array.from({ length: NUM_BLOCKS }, (_, b) => {
        const o = owner[b];
        const slots = Array.from({ length: B }, (_, k) => {
          const on = o && k < o.filled;
          const bg = o ? (on ? W.color(o.s.c) : 'transparent') : 'transparent';
          return `<span title="槽位 ${b * B + k}" style="flex:1;height:20px;border-radius:2px;background:${bg};border:1px solid ${o ? W.color(o.s.c) : W.color('rule-strong')};opacity:${o && !on ? 0.55 : 1}"></span>`;
        }).join('');
        return `<div style="border:1px solid var(--rule);border-radius:4px;padding:5px 6px;background:var(--paper)">
          <div style="display:flex;justify-content:space-between;font:600 0.72rem var(--font-mono);color:var(--ink-3);margin-bottom:4px"><span>块 ${b}</span><span>${o ? o.s.id : '空闲'}</span></div>
          <div style="display:flex;gap:3px">${slots}</div></div>`;
      }).join('');
      const rows = r.seqs.map((s) => {
        const st = { new: '未到达', waiting: s.preempts ? '等待（被抢占过）' : '等待', running: '运行', finished: '已结束' }[s.status];
        const pos = s.computed - 1;
        const slot = s.bt.length && pos >= 0 ? `位置 ${pos} → 块表[${Math.floor(pos / B)}] = ${s.bt[Math.floor(pos / B)]}，槽位 ${s.bt[Math.floor(pos / B)]} × ${B} + ${pos % B} = ${s.bt[Math.floor(pos / B)] * B + (pos % B)}` : '';
        return `<tr><td><i class="lw-dot" style="background:${W.color(s.c)};margin-right:6px"></i>${s.id}</td><td>${st}</td><td style="font-family:var(--font-mono)">${s.computed} / ${s.num}</td>
          <td style="font-family:var(--font-mono)">[${s.bt.join(', ')}]</td><td style="font-size:0.78rem;color:var(--ink-2)">${s.active ? slot : ''}</td></tr>`;
      }).join('');
      out.innerHTML = `
        <div class="lw-subtitle" style="margin:0 0 10px">第 ${step} 步结束后，空闲块 ${r.free.length} / ${NUM_BLOCKS}</div>
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px">${grid}</div>
        <table style="margin-top:12px;font-size:0.84rem">
          <thead><tr><th>序列</th><th>状态</th><th>已算 / 总数</th><th>块表</th><th>本步最后写入的位置</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="lw-callout">${r.log.length ? r.log.join('。') + '。' : '所有运行中的序列各 decode 一个 token。'}</div>`;
      container.querySelector('[data-prev]').disabled = step === 1;
      container.querySelector('[data-next]').disabled = r.done;
    }

    container.querySelector('[data-prev]').addEventListener('click', () => { if (step > 1) { step--; update(); } });
    container.querySelector('[data-next]').addEventListener('click', () => { step++; update(); });
    update();
  });

  // ------------------------------------------------------------------ 1.4 前缀缓存
  W.register('prefix-match', function (container) {
    const B = 4;
    const PRESETS = [
      { name: '共享系统提示词', a: '你是客服助手请礼貌回答。用户：退货流程是什么？', b: '你是客服助手请礼貌回答。用户：几天能到货？' },
      { name: '开头写了时间', a: '时间10:01。你是客服助手请礼貌回答。用户：退货流程是什么？', b: '时间10:02。你是客服助手请礼貌回答。用户：退货流程是什么？' },
      { name: '多轮对话', a: '系统：简洁回答。用户：介绍北京。助手：北京是中国首都。', b: '系统：简洁回答。用户：介绍北京。助手：北京是中国首都。用户：人口多少？' },
      { name: '去掉了开头', a: '你好，今天天气怎么样？', b: '今天天气怎么样？' },
    ];
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">块哈希前缀缓存</h4></div>
        <p class="lw-subtitle">请求 1 先运行，它的完整块被缓存。然后请求 2 到达，按块依次查找。每个字符算作一个 token，块大小为 4。</p>
        <div class="lw-tabs">${PRESETS.map((p, i) => `<button type="button" class="lw-tab" data-i="${i}">${p.name}</button>`).join('')}</div>
        <div class="lw-controls" style="grid-template-columns:1fr">
          <div class="lw-control"><label>请求 1</label><input type="text" data-a /></div>
          <div class="lw-control"><label>请求 2</label><input type="text" data-b /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function blocks(text) {
      const chars = Array.from(text);
      const out = [];
      for (let i = 0; i < chars.length; i += B) out.push(chars.slice(i, i + B));
      return out;
    }

    function box(chars, kind, label) {
      const s = {
        hit: 'border-color:var(--ok);background:var(--ok-soft)',
        miss: 'border-color:var(--prefill);background:var(--prefill-soft)',
        tail: 'border-color:var(--rule-strong);border-style:dashed;background:transparent',
        cached: 'border-color:var(--rule-strong);background:var(--paper-2)',
      }[kind];
      return `<div style="border:1px solid;border-radius:4px;padding:4px 6px;min-width:4.6em;${s}">
        <div style="font-size:0.95rem;letter-spacing:0.08em;white-space:pre">${chars.map((c) => W.escapeHtml(c)).join('')}</div>
        <div style="font-size:0.7rem;color:var(--ink-3);margin-top:2px">${label}</div></div>`;
    }

    function update() {
      const a = blocks($('[data-a]').value);
      const b = blocks($('[data-b]').value);
      const cached = a.filter((blk) => blk.length === B).map((blk) => blk.join(''));
      // 链式匹配：前面的块全部命中，这一块的内容也相同，才算命中
      let hitBlocks = 0;
      while (hitBlocks < b.length && b[hitBlocks].length === B && hitBlocks < cached.length && b[hitBlocks].join('') === cached[hitBlocks]) hitBlocks++;
      const tokensB = b.reduce((n, blk) => n + blk.length, 0);
      const rowA = a.map((blk, i) => box(blk, blk.length === B ? 'cached' : 'tail', blk.length === B ? `块 ${i}，已缓存` : '未满，不缓存')).join('');
      const rowB = b.map((blk, i) => {
        if (i < hitBlocks) return box(blk, 'hit', `块 ${i}，命中`);
        if (blk.length < B) return box(blk, 'tail', '未满，需计算');
        const sameText = i < cached.length && blk.join('') === cached[i];
        return box(blk, 'miss', sameText ? `块 ${i}，文本同但前缀不同` : `块 ${i}，需计算`);
      }).join('');
      $('[data-out]').innerHTML = `
        <div style="font-size:0.8rem;color:var(--ink-2);margin:4px 0 6px">请求 1</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">${rowA}</div>
        <div style="font-size:0.8rem;color:var(--ink-2);margin:14px 0 6px">请求 2</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">${rowB}</div>
        <div class="lw-stats" style="margin-top:14px">
          <div class="lw-stat ok"><div class="lw-stat-label">命中的 token</div><div class="lw-stat-value">${hitBlocks * B}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">需要 prefill 的 token</div><div class="lw-stat-value">${tokensB - hitBlocks * B}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">命中率</div><div class="lw-stat-value">${tokensB ? Math.round((hitBlocks * B * 100) / tokensB) : 0}%</div></div>
        </div>
        <div class="lw-callout">${hitBlocks === 0 && b.length ? '第 0 块就不同，后面即使有相同的内容也无法复用：块的哈希包含了前一块的哈希。' : '命中在第一个不同的块处停止。最后不满 4 个字的部分不会被缓存，也不会被命中。'}</div>`;
    }

    function load(i) {
      $('[data-a]').value = PRESETS[i].a;
      $('[data-b]').value = PRESETS[i].b;
      container.querySelectorAll('[data-i]').forEach((x) => x.classList.toggle('active', Number(x.dataset.i) === i));
      update();
    }
    container.querySelectorAll('[data-i]').forEach((x) => x.addEventListener('click', () => load(Number(x.dataset.i))));
    $('[data-a]').addEventListener('input', update);
    $('[data-b]').addEventListener('input', update);
    load(0);
  });

  // ------------------------------------------------------------------ 项目 1：_prepare_inputs
  // 与 MiniServe solution/engine.py 的 _prepare_inputs 逻辑一致。默认值就是分步指南第 7 步的例子。
  W.register('prepare-inputs', function (container) {
    const B = 4;
    const SEQS = [
      { id: 'A', tokens: [11, 12, 13, 14, 15, 16], prompt: 6, pool: [3, 7, 10], computed: 4, n: 2 },
      { id: 'B', tokens: [21, 22, 23, 24, 25], prompt: 3, pool: [5, 2, 11], computed: 4, n: 1 },
      { id: 'C', tokens: [31, 32, 33, 34, 35, 36, 37], prompt: 7, pool: [9, 6, 1], computed: 0, n: 3 },
    ];
    const opt = (lo, hi, v) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).map((x) => `<option ${x === v ? 'selected' : ''}>${x}</option>`).join('');
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">把一批序列展平成模型输入</h4></div>
        <p class="lw-subtitle">块大小为 4。改变每个序列已算的 token 数和本步处理的 token 数，看 <code>_prepare_inputs</code> 的输出怎样变化。默认值是分步指南第 7 步的例子。</p>
        <table style="font-size:0.84rem;margin:0 0 6px">
          <thead><tr><th>序列</th><th>token_ids</th><th>已算</th><th>本步 n</th><th>块表</th></tr></thead>
          <tbody>${SEQS.map((s, i) => `<tr>
            <td>${s.id}</td>
            <td style="font-family:var(--font-mono)">[${s.tokens.join(', ')}]<div style="font-size:0.72rem;color:var(--ink-3)">prompt ${s.prompt} 个${s.tokens.length > s.prompt ? `，已生成 ${s.tokens.length - s.prompt} 个` : ''}</div></td>
            <td><select data-c="${i}"></select></td>
            <td><select data-n="${i}"></select></td>
            <td style="font-family:var(--font-mono)" data-bt="${i}"></td></tr>`).join('')}</tbody>
        </table>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function sync() {
      SEQS.forEach((s, i) => {
        s.computed = Math.min(s.computed, s.tokens.length - 1);
        s.n = Math.max(1, Math.min(s.n, s.tokens.length - s.computed));
        $(`[data-c="${i}"]`).innerHTML = opt(0, s.tokens.length - 1, s.computed);
        $(`[data-n="${i}"]`).innerHTML = opt(1, s.tokens.length - s.computed, s.n);
      });
    }

    function update() {
      const ids = [], pos = [], slots = [], owner = [];
      const pf = [], dec = [], sampleRows = [], sampleSeqs = [];
      let row = 0;
      SEQS.forEach((s, i) => {
        const start = s.computed, ctx = start + s.n;
        const bt = s.pool.slice(0, Math.ceil(ctx / B));
        $(`[data-bt="${i}"]`).textContent = `[${bt.join(', ')}]`;
        const slotOf = (p) => bt[Math.floor(p / B)] * B + (p % B);
        for (let p = start; p < ctx; p++) { ids.push(s.tokens[p]); pos.push(p); slots.push(slotOf(p)); owner.push(s.id); }
        const ctxSlots = Array.from({ length: ctx }, (_, p) => slotOf(p));
        if (s.n === 1 && start > 0) dec.push({ s, row, ctxSlots });
        else pf.push({ s, row, n: s.n, ctxSlots });
        if (ctx === s.tokens.length) { sampleRows.push(row + s.n - 1); sampleSeqs.push(s.id); }
        row += s.n;
      });
      const cellRow = (label, arr, mono = true) => `<tr><td style="color:var(--ink-2);white-space:nowrap">${label}</td>${arr.map((v, k) => `<td style="text-align:center;${mono ? 'font-family:var(--font-mono);' : ''}${owner[k] ? '' : ''}">${v}</td>`).join('')}</tr>`;
      const head = `<tr><th></th>${owner.map((o, k) => `<th style="text-align:center;font-weight:600">${k}<div style="font-size:0.72rem;color:var(--ink-3);font-weight:400">${o}</div></th>`).join('')}</tr>`;
      $('[data-out]').innerHTML = `
        <div style="overflow-x:auto"><table style="font-size:0.84rem;width:auto">
          <thead>${head}</thead>
          <tbody>${cellRow('input_ids', ids)}${cellRow('positions', pos)}${cellRow('slot_mapping', slots)}</tbody>
        </table></div>
        <div style="display:grid;grid-template-columns:9em 1fr;gap:6px 12px;font-size:0.84rem;margin-top:10px">
          <div style="color:var(--ink-2)">prefill 组</div>
          <div>${pf.length ? pf.map((g) => `${g.s.id}：从第 ${g.row} 行起 ${g.n} 行，上下文槽位 <code>[${g.ctxSlots.join(', ')}]</code>`).join('<br>') : '无'}</div>
          <div style="color:var(--ink-2)">decode 组</div>
          <div>${dec.length ? dec.map((g) => `${g.s.id}：第 ${g.row} 行，上下文槽位 <code>[${g.ctxSlots.join(', ')}]</code>`).join('<br>') : '无'}</div>
          <div style="color:var(--ink-2)">sample_rows</div>
          <div><code>[${sampleRows.join(', ')}]</code>${sampleSeqs.length ? `，对应 ${sampleSeqs.join('、')}` : ''}</div>
        </div>
        <div class="lw-callout">只有已算的 token 数加上本步的 n 等于序列的总长度时，这个序列才采样下一个 token，它的最后一行进入 sample_rows。本步只有 1 个 token 且之前算过的序列走 decode 组，其余（包括切块 prefill 和重算）走 prefill 组。</div>`;
    }

    container.addEventListener('change', (e) => {
      const c = e.target.dataset.c, n = e.target.dataset.n;
      if (c !== undefined) SEQS[Number(c)].computed = Number(e.target.value);
      if (n !== undefined) SEQS[Number(n)].n = Number(e.target.value);
      sync(); update();
    });
    sync(); update();
  });
})();
