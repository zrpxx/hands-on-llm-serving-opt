/**
 * 第 0 部分的交互组件
 * - load-balance-sim     0.1 负载均衡策略对比
 * - transformer-flops    0.2 线性层与注意力的计算量
 * - sampling-lab         0.3 温度、top-k、top-p
 * - kv-cache-steps       0.4 有无 KV Cache 时每一步的计算
 * - gpu-compare          0.5 两张 GPU 的规格与推导出的延迟下限
 * - memory-budget        0.6 显存账本与并发上限
 * - roofline             0.7 Roofline 与三类计算的位置
 * - goodput-sim          0.8 请求速率、延迟与 goodput
 *
 * 写法约定：控件只渲染一次，拖动滑块时只更新结果区域，避免重绘打断拖动。
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  }

  function quantile(arr, q) {
    if (!arr.length) return 0;
    const a = arr.slice().sort((x, y) => x - y);
    return a[Math.min(a.length - 1, Math.floor(q * a.length))];
  }

  // ------------------------------------------------------------------ 0.1 负载均衡
  W.register('load-balance-sim', function (container) {
    const N_REPLICA = 3;
    const N_REQ = 36;
    const POLICIES = [
      { id: 'rr', name: '轮询' },
      { id: 'count', name: '排队请求最少' },
      { id: 'work', name: '剩余工作量最少' },
    ];
    let policy = 'rr';
    let seed = 11;
    let gap = 1.25; // 平均到达间隔

    function makeRequests() {
      const r = rng(seed);
      const reqs = [];
      let t = 0;
      for (let i = 0; i < N_REQ; i++) {
        t += -Math.log(1 - r()) * gap;
        const big = r() < 0.18;
        const size = big ? 7 + r() * 8 : 0.4 + r() * 1.2;
        reqs.push({ id: i, arrive: t, size, big });
      }
      return reqs;
    }

    function simulate(reqs) {
      const reps = Array.from({ length: N_REPLICA }, () => ({ jobs: [], freeAt: 0 }));
      let rr = 0;
      reqs.forEach((q) => {
        // 到达时刻各副本的状态
        const pending = reps.map((rep) => rep.jobs.filter((j) => j.end > q.arrive));
        let k;
        if (policy === 'rr') {
          k = rr % N_REPLICA;
          rr += 1;
        } else if (policy === 'count') {
          k = 0;
          for (let i = 1; i < N_REPLICA; i++) if (pending[i].length < pending[k].length) k = i;
        } else {
          const work = reps.map((rep) => Math.max(0, rep.freeAt - q.arrive));
          k = 0;
          for (let i = 1; i < N_REPLICA; i++) if (work[i] < work[k]) k = i;
        }
        const rep = reps[k];
        const start = Math.max(q.arrive, rep.freeAt);
        const end = start + q.size;
        rep.freeAt = end;
        rep.jobs.push({ ...q, start, end, wait: start - q.arrive });
      });
      return reps;
    }

    function render() {
      const reqs = makeRequests();
      const reps = simulate(reqs);
      const all = reps.flatMap((r) => r.jobs);
      const waits = all.map((j) => j.wait);
      const smallWaits = all.filter((j) => !j.big).map((j) => j.wait);
      const mean = waits.reduce((a, b) => a + b, 0) / waits.length;
      const tEnd = Math.max(...all.map((j) => j.end));

      const Wd = 640;
      const left = 54;
      const rowH = 38;
      const H = N_REPLICA * rowH + 28;
      const x = (t) => left + (t / tEnd) * (Wd - left - 8);
      const ink3 = W.color('ink-3');
      const rule = W.color('rule');

      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="各副本处理请求的时间线">`;
      reps.forEach((rep, i) => {
        const y = i * rowH + 6;
        svg += `<text x="0" y="${y + 17}" font-size="12" fill="${W.color('ink-2')}">副本 ${i + 1}</text>`;
        svg += `<line x1="${left}" x2="${Wd - 8}" y1="${y + rowH - 4}" y2="${y + rowH - 4}" stroke="${rule}"/>`;
        rep.jobs.forEach((j) => {
          if (j.wait > 0.05) {
            svg += `<line x1="${x(j.arrive)}" x2="${x(j.start)}" y1="${y + 25}" y2="${y + 25}" stroke="${W.color('err')}" stroke-width="1.5" stroke-dasharray="3 2"/>`;
          }
          svg += `<rect x="${x(j.start)}" y="${y + 4}" width="${Math.max(1.5, x(j.end) - x(j.start) - 0.8)}" height="17" rx="2"
            fill="${j.big ? W.color('decode') : W.color('prefill')}" opacity="${j.big ? 0.9 : 0.75}"><title>请求 ${j.id + 1}：大小 ${j.size.toFixed(1)}，等待 ${j.wait.toFixed(1)}</title></rect>`;
        });
      });
      const axisY = N_REPLICA * rowH + 18;
      svg += `<text x="${left}" y="${axisY}" font-size="11" fill="${ink3}">0</text>`;
      svg += `<text x="${Wd - 8}" y="${axisY}" font-size="11" fill="${ink3}" text-anchor="end">时间 ${tEnd.toFixed(0)}</text>`;
      svg += '</svg>';

      const rrNote = {
        rr: '轮询不看副本的状态。大请求恰好连续分到同一个副本时，后面的小请求要在它后面排很久（红色虚线是等待时间）。',
        count: '按排队请求数分配好一些，但一个大请求和一个小请求在计数上一样，仍然会把请求派到正在处理大请求的副本上。',
        work: '按剩余工作量分配最接近理想情况。实际系统里拿不到精确的剩余工作量，常用正在运行的 token 数、KV Cache 使用率来近似。',
      }[policy];

      container.innerHTML = `
        <div class="lw-card">
          <div class="lw-header"><h4 class="lw-title">分发策略对比</h4></div>
          <p class="lw-subtitle">${N_REQ} 个请求依次到达，分给 3 个副本。蓝色是小请求，琥珀色是大请求（约占 18%，开销是小请求的十倍左右）。为了突出分发策略的影响，这里假设每个副本一次只处理一个请求。</p>
          <div class="lw-tabs">
            ${POLICIES.map((p) => `<button type="button" class="lw-tab ${p.id === policy ? 'active' : ''}" data-p="${p.id}">${p.name}</button>`).join('')}
            <button type="button" class="lw-btn" data-reseed style="margin-left:auto">换一组请求</button>
          </div>
          <div class="lw-controls" style="grid-template-columns:1fr">
            <div class="lw-control">
              <label>平均到达间隔（越小负载越高）<span class="lw-val">${gap.toFixed(2)}</span></label>
              <input type="range" min="0.9" max="2.2" step="0.05" value="${gap}" data-gap />
            </div>
          </div>
          ${svg}
          <div class="lw-stats">
            <div class="lw-stat"><div class="lw-stat-label">平均等待</div><div class="lw-stat-value">${mean.toFixed(2)}</div></div>
            <div class="lw-stat"><div class="lw-stat-label">P90 等待</div><div class="lw-stat-value">${quantile(waits, 0.9).toFixed(2)}</div></div>
            <div class="lw-stat"><div class="lw-stat-label">小请求的最长等待</div><div class="lw-stat-value">${Math.max(...smallWaits).toFixed(2)}</div></div>
          </div>
          <div class="lw-callout">${rrNote}</div>
        </div>`;

      container.querySelectorAll('[data-p]').forEach((b) => b.addEventListener('click', () => { policy = b.dataset.p; render(); }));
      container.querySelector('[data-reseed]').addEventListener('click', () => { seed = (seed * 7 + 3) % 9973; render(); });
      const gapInput = container.querySelector('[data-gap]');
      gapInput.addEventListener('input', (e) => { gapInput.previousElementSibling.querySelector('.lw-val').textContent = Number(e.target.value).toFixed(2); });
      gapInput.addEventListener('change', (e) => { gap = Number(e.target.value); render(); });
    }

    render();
  });

  // ------------------------------------------------------------------ 共用：模型配置
  // 参数来自各模型的 config.json。tied 表示输出头与嵌入表共享权重。
  const MODELS = {
    'qwen3-0.6b': { name: 'Qwen3-0.6B', L: 28, h: 1024, nq: 16, nkv: 8, dh: 128, inter: 3072, V: 151936, tied: true },
    'qwen3-8b': { name: 'Qwen3-8B', L: 36, h: 4096, nq: 32, nkv: 8, dh: 128, inter: 12288, V: 151936, tied: false },
    'qwen3-32b': { name: 'Qwen3-32B', L: 64, h: 5120, nq: 64, nkv: 8, dh: 128, inter: 25600, V: 151936, tied: false },
    'llama3-70b': { name: 'Llama-3.1-70B', L: 80, h: 8192, nq: 64, nkv: 8, dh: 128, inter: 28672, V: 128256, tied: false },
    'llama2-7b': { name: 'Llama-2-7B（MHA）', L: 32, h: 4096, nq: 32, nkv: 32, dh: 128, inter: 11008, V: 32000, tied: false },
  };
  W.MODELS = MODELS;

  function modelParams(m) {
    const attn = m.h * m.nq * m.dh * 2 + m.h * m.nkv * m.dh * 2;
    const mlp = 3 * m.h * m.inter;
    const emb = m.V * m.h;
    const head = m.tied ? 0 : m.V * m.h;
    return { attn: attn * m.L, mlp: mlp * m.L, emb, head, total: (attn + mlp) * m.L + emb + head };
  }
  W.modelParams = modelParams;

  function fmtFlops(x) {
    if (x >= 1e15) return (x / 1e15).toFixed(2) + ' PFLOPs';
    if (x >= 1e12) return (x / 1e12).toFixed(2) + ' TFLOPs';
    return (x / 1e9).toFixed(x >= 1e10 ? 1 : 2) + ' GFLOPs';
  }

  function modelSelect(id, current, keys) {
    return `<select data-model id="${id}">${(keys || Object.keys(MODELS))
      .map((k) => `<option value="${k}" ${k === current ? 'selected' : ''}>${MODELS[k].name}</option>`)
      .join('')}</select>`;
  }

  // ------------------------------------------------------------------ 0.2 计算量构成
  W.register('transformer-flops', function (container) {
    const st = { model: 'qwen3-8b', logT: 10, mode: 'decode' };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">线性层与注意力的计算量</h4></div>
        <div class="lw-tabs">
          <button type="button" class="lw-tab" data-mode="decode">生成一个 token</button>
          <button type="button" class="lw-tab" data-mode="prefill">prefill 整段输入</button>
        </div>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${modelSelect('tf-model', st.model)}</div>
          <div class="lw-control">
            <label><span data-tlabel>上下文长度</span><span class="lw-val" data-tval></span></label>
            <input type="range" min="7" max="17" step="0.5" value="${st.logT}" data-t />
          </div>
        </div>
        <div data-out></div>
      </div>`;
    const out = container.querySelector('[data-out]');

    function update() {
      const m = MODELS[st.model];
      const T = Math.round(Math.pow(2, st.logT));
      const p = modelParams(m);
      const linPerTok = 2 * (p.attn + p.mlp + (m.tied ? p.emb : p.head));
      let parts;
      if (st.mode === 'decode') {
        const proj = 2 * p.attn;
        const mlp = 2 * p.mlp;
        const head = 2 * m.V * m.h;
        const score = 4 * m.L * m.nq * m.dh * T;
        parts = [
          { name: '注意力投影 Q/K/V/O', v: proj, c: 'series-1' },
          { name: '前馈网络', v: mlp, c: 'ink-2' },
          { name: '输出头', v: head, c: 'series-2' },
          { name: '注意力打分（随长度增长）', v: score, c: 'decode' },
        ];
      } else {
        // 一次处理 T 个 token：线性层 T 份；因果注意力平均每个位置看 T/2 个历史位置，输出头只算最后一行
        const proj = 2 * p.attn * T;
        const mlp = 2 * p.mlp * T;
        const head = 2 * m.V * m.h;
        const score = 4 * m.L * m.nq * m.dh * (T * (T + 1)) / 2;
        parts = [
          { name: '注意力投影 Q/K/V/O', v: proj, c: 'series-1' },
          { name: '前馈网络', v: mlp, c: 'ink-2' },
          { name: '输出头（只算最后一行）', v: head, c: 'series-2' },
          { name: '注意力打分（随长度平方增长）', v: score, c: 'decode' },
        ];
      }
      const total = parts.reduce((a, b) => a + b.v, 0);
      const score = parts[3].v;
      const cross = Math.round(linPerTok / (4 * m.L * m.nq * m.dh));
      const crossPrefill = Math.round((2 * (p.attn + p.mlp)) / (2 * m.L * m.nq * m.dh));
      container.querySelector('[data-tval]').textContent = T.toLocaleString('en-US') + ' tokens';
      container.querySelector('[data-tlabel]').textContent = st.mode === 'decode' ? '上下文长度（已有的 token 数）' : '输入长度';
      container.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('active', b.dataset.mode === st.mode));
      out.innerHTML = `
        <div class="lw-stack-bar" style="height:30px">
          ${parts.map((q) => `<div class="lw-stack-seg" style="width:${(q.v / total) * 100}%;background:${W.color(q.c)}" title="${q.name}"></div>`).join('')}
        </div>
        <div class="lw-legend">
          ${parts.map((q) => `<span class="lw-legend-item"><i class="lw-dot" style="background:${W.color(q.c)}"></i>${q.name} ${((q.v / total) * 100).toFixed(1)}%</span>`).join('')}
        </div>
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">${st.mode === 'decode' ? '这一个 token 的总计算量' : '整段 prefill 的总计算量'}</div><div class="lw-stat-value">${fmtFlops(total)}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">其中注意力打分</div><div class="lw-stat-value">${fmtFlops(score)}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">参数量</div><div class="lw-stat-value">${(p.total / 1e9).toFixed(2)} B</div></div>
        </div>
        <div class="lw-callout">${st.mode === 'decode'
          ? `对 ${m.name}，上下文超过约 <strong>${cross.toLocaleString('en-US')}</strong> 个 token 后，注意力打分的计算量会超过全部线性层。`
          : `对 ${m.name}，输入超过约 <strong>${crossPrefill.toLocaleString('en-US')}</strong> 个 token 后，prefill 中注意力打分的计算量会超过注意力投影和前馈网络之和。`}</div>`;
    }

    container.querySelector('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    container.querySelector('[data-t]').addEventListener('input', (e) => { st.logT = Number(e.target.value); update(); });
    container.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => { st.mode = b.dataset.mode; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 0.3 采样
  W.register('sampling-lab', function (container) {
    const CANDS = [
      { tok: '北', z: 6.0 },
      { tok: '一', z: 3.4 },
      { tok: '位于', z: 2.9 },
      { tok: '哪里', z: 2.2 },
      { tok: '上', z: 1.6 },
      { tok: '中', z: 1.1 },
      { tok: '南', z: 0.6 },
      { tok: '世界', z: -0.2 },
    ];
    const st = { temp: 1.0, k: 8, p: 1.0, draws: [] };
    let seed = 5;
    const r = rng(seed);

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">温度、top-k 与 top-p</h4></div>
        <p class="lw-subtitle">输入是“中国的首都是”，下面是 8 个候选 token 的 logits（真实词表有十几万个，这里只取了前几名）。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>温度 τ<span class="lw-val" data-v="temp"></span></label><input type="range" min="0" max="2" step="0.05" value="${st.temp}" data-k="temp" /></div>
          <div class="lw-control"><label>top-k<span class="lw-val" data-v="k"></span></label><input type="range" min="1" max="8" step="1" value="${st.k}" data-k="k" /></div>
          <div class="lw-control"><label>top-p<span class="lw-val" data-v="p"></span></label><input type="range" min="0.5" max="1" step="0.01" value="${st.p}" data-k="p" /></div>
        </div>
        <div data-bars style="display:grid;grid-template-columns:4.5em 3.6em 1fr 4.2em;gap:6px 10px;align-items:center;font-size:0.84rem"></div>
        <div class="lw-btn-row" style="margin-top:14px">
          <button type="button" class="lw-btn" data-draw>按当前设置抽样 10 次</button>
          <span data-draws style="font-size:0.88rem;color:var(--ink-2);align-self:center"></span>
        </div>
      </div>`;
    const bars = container.querySelector('[data-bars]');

    function dist() {
      const greedy = st.temp === 0;
      const t = Math.max(st.temp, 1e-6);
      const m = Math.max(...CANDS.map((c) => c.z / t));
      const ex = CANDS.map((c) => Math.exp(c.z / t - m));
      const s = ex.reduce((a, b) => a + b, 0);
      const raw = ex.map((e) => e / s);
      // 候选已按 logit 从高到低排列
      const keep = raw.map((_, i) => i < st.k);
      let cum = 0;
      raw.forEach((p, i) => {
        if (!keep[i]) return;
        if (cum >= st.p - 1e-9 && i > 0) keep[i] = false;
        cum += p;
      });
      if (greedy) keep.forEach((_, i) => { keep[i] = i === 0; });
      const ks = raw.reduce((a, p, i) => a + (keep[i] ? p : 0), 0);
      return { raw, keep, fin: raw.map((p, i) => (keep[i] ? p / ks : 0)) };
    }

    function update() {
      container.querySelector('[data-v=temp]').textContent = st.temp === 0 ? '0（贪心）' : st.temp.toFixed(2);
      container.querySelector('[data-v=k]').textContent = st.k;
      container.querySelector('[data-v=p]').textContent = st.p.toFixed(2);
      const d = dist();
      bars.innerHTML = '<div style="color:var(--ink-3)">token</div><div style="color:var(--ink-3)">logit</div><div style="color:var(--ink-3)">最终概率（灰色是截断前）</div><div></div>' +
        CANDS.map((c, i) => `
          <div style="font-weight:600;${d.keep[i] ? '' : 'color:var(--ink-3);text-decoration:line-through'}">${c.tok}</div>
          <div style="font-family:var(--font-mono);color:var(--ink-3)">${c.z.toFixed(1)}</div>
          <div style="position:relative;height:16px;background:var(--paper-3);border-radius:2px;overflow:hidden">
            <div style="position:absolute;inset:0 auto 0 0;width:${(d.raw[i] * 100).toFixed(2)}%;background:var(--rule-strong)"></div>
            <div style="position:absolute;inset:0 auto 0 0;width:${(d.fin[i] * 100).toFixed(2)}%;background:var(--prefill)"></div>
          </div>
          <div style="font-family:var(--font-mono);text-align:right">${d.keep[i] ? (d.fin[i] * 100).toFixed(1) + '%' : '—'}</div>`).join('');
      container.querySelector('[data-draws]').textContent = st.draws.length ? '结果：' + st.draws.join(' ') : '';
    }

    container.querySelectorAll('input[data-k]').forEach((inp) => inp.addEventListener('input', () => {
      st[inp.dataset.k] = Number(inp.value);
      st.draws = [];
      update();
    }));
    container.querySelector('[data-draw]').addEventListener('click', () => {
      const d = dist();
      st.draws = Array.from({ length: 10 }, () => {
        let u = r();
        for (let i = 0; i < CANDS.length; i++) { u -= d.fin[i]; if (u <= 0) return CANDS[i].tok; }
        return CANDS[0].tok;
      });
      update();
    });
    update();
  });

  // ------------------------------------------------------------------ 0.4 KV Cache 逐步对比
  W.register('kv-cache-steps', function (container) {
    const PROMPT = ['AI', '推理', '为什么'];
    const GEN = ['需要', '缓存', '？'];
    const MAX = GEN.length;
    let step = 0; // 0 = prefill

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">有无 KV Cache 时，每一步要计算哪些位置</h4>
          <div class="lw-btn-row" style="margin:0">
            <button type="button" class="lw-btn" data-prev>上一步</button>
            <button type="button" class="lw-btn primary" data-next>下一步</button>
          </div>
        </div>
        <div data-out></div>
      </div>`;
    const out = container.querySelector('[data-out]');

    function cell(text, kind) {
      const style = {
        compute: 'background:var(--prefill-soft);border-color:var(--prefill);color:var(--prefill-ink)',
        redo: 'background:var(--err-soft);border-color:var(--err);color:var(--ink)',
        cached: 'background:var(--paper-2);border-color:var(--rule-strong);color:var(--ink-2)',
        none: 'background:transparent;border-color:transparent;color:var(--ink-3)',
      }[kind];
      return `<div style="border:1px solid;border-radius:4px;padding:6px 4px;text-align:center;font-size:0.82rem;${style}">${text}</div>`;
    }

    function update() {
      const len = PROMPT.length + step;
      const toks = PROMPT.concat(GEN).slice(0, len + (step < MAX ? 0 : 0));
      const cols = PROMPT.length + MAX;
      let noCacheTotal = 0;
      let cacheTotal = 0;
      for (let s = 0; s <= step; s++) {
        noCacheTotal += PROMPT.length + s;
        cacheTotal += s === 0 ? PROMPT.length : 1;
      }
      const all = PROMPT.concat(GEN);
      const head = all.map((w, i) => `<div style="text-align:center;font-size:0.78rem;color:${i < len ? 'var(--ink)' : 'var(--ink-3)'}">${i < len ? w : '·'}</div>`).join('');
      const rowNo = all.map((w, i) => (i < len ? cell('计算 q k v', 'redo') : cell('', 'none'))).join('');
      const rowYes = all.map((w, i) => {
        if (i >= len) return cell('', 'none');
        const isNew = step === 0 || i === len - 1;
        return isNew ? cell('计算 q k v', 'compute') : cell('读 k v', 'cached');
      }).join('');
      const label = step === 0 ? `第 0 步（prefill）：处理 prompt 的 ${PROMPT.length} 个 token，输出“${GEN[0]}”` :
        step < MAX ? `第 ${step} 步（decode）：输入“${all[len - 1]}”，输出“${GEN[step]}”` : `第 ${step} 步（decode）：输入“${all[len - 1]}”，输出结束符`;
      out.innerHTML = `
        <p class="lw-subtitle" style="margin:0 0 12px">${label}</p>
        <div style="display:grid;grid-template-columns:6.5em repeat(${cols}, minmax(0,1fr));gap:4px;align-items:center">
          <div></div>${head}
          <div style="font-size:0.8rem;color:var(--ink-2)">不用缓存</div>${rowNo}
          <div style="font-size:0.8rem;color:var(--ink-2)">KV Cache</div>${rowYes}
        </div>
        <div class="lw-stats" style="margin-top:14px">
          <div class="lw-stat"><div class="lw-stat-label">本步计算的位置数（不用缓存 / KV Cache）</div><div class="lw-stat-value">${len} / ${step === 0 ? PROMPT.length : 1}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">累计计算的位置数（不用缓存 / KV Cache）</div><div class="lw-stat-value">${noCacheTotal} / ${cacheTotal}</div></div>
        </div>
        <div class="lw-callout">${step === 0
          ? '第一步两种方式相同：prompt 的每个位置都要计算一次。区别从下一步开始。'
          : '不用缓存时，前面的位置每一步都被重新计算，结果和上一步完全相同（红色）。使用 KV Cache 时，只计算新位置，前面位置的 k、v 直接从缓存读取，它们的 q 已不再需要。'}</div>`;
      container.querySelector('[data-prev]').disabled = step === 0;
      container.querySelector('[data-next]').disabled = step === MAX;
    }

    container.querySelector('[data-prev]').addEventListener('click', () => { if (step > 0) { step--; update(); } });
    container.querySelector('[data-next]').addEventListener('click', () => { if (step < MAX) { step++; update(); } });
    update();
  });

  // ------------------------------------------------------------------ 共用：GPU 规格（稠密算力）
  const GPUS = {
    b200: { name: 'B200', mem: 180, bw: 8.0, bf16: 2250, fp8: 4500, link: 'NVLink 1800 GB/s' },
    h200: { name: 'H200 SXM', mem: 141, bw: 4.8, bf16: 989, fp8: 1979, link: 'NVLink 900 GB/s' },
    h100: { name: 'H100 SXM', mem: 80, bw: 3.35, bf16: 989, fp8: 1979, link: 'NVLink 900 GB/s' },
    h20: { name: 'H20', mem: 96, bw: 4.0, bf16: 148, fp8: 296, link: 'NVLink 900 GB/s' },
    a100: { name: 'A100 SXM 80GB', mem: 80, bw: 2.04, bf16: 312, fp8: 0, link: 'NVLink 600 GB/s' },
    a800: { name: 'A800 SXM 80GB', mem: 80, bw: 2.04, bf16: 312, fp8: 0, link: 'NVLink 400 GB/s' },
    l40s: { name: 'L40S', mem: 48, bw: 0.864, bf16: 362, fp8: 733, link: 'PCIe 64 GB/s' },
    a10: { name: 'A10', mem: 24, bw: 0.6, bf16: 125, fp8: 0, link: 'PCIe 64 GB/s' },
    rtx4090: { name: 'RTX 4090', mem: 24, bw: 1.008, bf16: 165, fp8: 330, link: 'PCIe 64 GB/s' },
  };
  W.GPUS = GPUS;

  function gpuSelect(attr, current) {
    return `<select ${attr}>${Object.keys(GPUS).map((k) => `<option value="${k}" ${k === current ? 'selected' : ''}>${GPUS[k].name}</option>`).join('')}</select>`;
  }

  // ------------------------------------------------------------------ 0.5 GPU 对比
  W.register('gpu-compare', function (container) {
    const st = { a: 'h100', b: 'h20', model: 'qwen3-32b', fp8: false };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">GPU 对比</h4></div>
        <div class="lw-controls">
          <div class="lw-control"><label>GPU 甲</label>${gpuSelect('data-g="a"', st.a)}</div>
          <div class="lw-control"><label>GPU 乙</label>${gpuSelect('data-g="b"', st.b)}</div>
          <div class="lw-control"><label>模型</label>${modelSelect('gc-model', st.model, ['qwen3-8b', 'qwen3-32b', 'llama3-70b'])}</div>
          <div class="lw-control"><label>权重精度</label><select data-prec><option value="bf16">BF16</option><option value="fp8">FP8</option></select></div>
        </div>
        <div data-out></div>
      </div>`;
    const out = container.querySelector('[data-out]');

    function row(label, va, vb, unit, fmt, note) {
      const max = Math.max(va, vb) || 1;
      const bar = (v, c) => `<div style="height:8px;background:var(--paper-3);border-radius:2px;margin-top:4px"><div style="height:100%;width:${(v / max) * 100}%;background:${W.color(c)};border-radius:2px"></div></div>`;
      return `<div style="display:grid;grid-template-columns:8.5em 1fr 1fr;gap:6px 16px;padding:9px 0;border-bottom:1px solid var(--rule);font-size:0.86rem;align-items:start">
        <div style="color:var(--ink-2)">${label}${note ? `<div style="font-size:0.74rem;color:var(--ink-3);margin-top:2px">${note}</div>` : ''}</div>
        <div><span style="font-family:var(--font-mono);font-weight:600">${fmt(va)}</span> ${unit}${bar(va, 'prefill')}</div>
        <div><span style="font-family:var(--font-mono);font-weight:600">${fmt(vb)}</span> ${unit}${bar(vb, 'decode')}</div>
      </div>`;
    }

    function update() {
      const A = GPUS[st.a];
      const B = GPUS[st.b];
      const m = MODELS[st.model];
      const P = modelParams(m).total;
      const bytes = P * (st.fp8 ? 1 : 2);
      const gb = bytes / 1e9;
      const flops = (g) => (st.fp8 && g.fp8 ? g.fp8 : g.bf16) * 1e12;
      const decode = (g) => (bytes / (g.bw * 1e12)) * 1000;
      const prefill = (g) => ((2 * P * 2048) / flops(g)) * 1000;
      const fits = (g) => gb < g.mem * 0.9;
      const fitNote = (g) => (fits(g) ? `放得下，剩余约 ${(g.mem * 0.9 - gb).toFixed(0)} GB` : `放不下，至少需要 ${Math.ceil(gb / (g.mem * 0.9))} 张`);
      const f1 = (v) => v.toLocaleString('en-US', { maximumFractionDigits: v < 10 ? 2 : 0 });
      out.innerHTML = `
        <div style="display:grid;grid-template-columns:8.5em 1fr 1fr;gap:16px;font-weight:650;font-size:0.88rem;padding-bottom:6px;border-bottom:1.5px solid var(--ink-2)">
          <div></div><div style="color:var(--prefill-ink)">${A.name}</div><div style="color:var(--decode-ink)">${B.name}</div>
        </div>
        ${row('显存容量', A.mem, B.mem, 'GB', f1)}
        ${row('显存带宽', A.bw, B.bw, 'TB/s', f1)}
        ${row('BF16 算力', A.bf16, B.bf16, 'TFLOPS', f1)}
        ${row('FP8 算力', A.fp8, B.fp8, 'TFLOPS', (v) => (v ? f1(v) : '不支持'))}
        <div style="display:grid;grid-template-columns:8.5em 1fr 1fr;gap:6px 16px;padding:9px 0;border-bottom:1px solid var(--rule);font-size:0.86rem">
          <div style="color:var(--ink-2)">放 ${m.name}<div style="font-size:0.74rem;color:var(--ink-3);margin-top:2px">权重 ${gb.toFixed(1)} GB，按 90% 显存计</div></div>
          <div style="color:${fits(A) ? 'var(--ok)' : 'var(--err)'}">${fitNote(A)}</div>
          <div style="color:${fits(B) ? 'var(--ok)' : 'var(--err)'}">${fitNote(B)}</div>
        </div>
        ${row('decode 每 token 下限', decode(A), decode(B), 'ms', (v) => v.toFixed(1), '单个请求：权重字节 ÷ 带宽')}
        ${row('prefill 2048 token 下限', prefill(A), prefill(B), 'ms', (v) => v.toFixed(0), '计算量 ÷ 算力，不计注意力')}
        <div class="lw-callout">两个下限都越短越好。decode 下限只看带宽，prefill 下限只看算力，所以同一对 GPU 在两项上的胜负可能相反。${st.fp8 && (!A.fp8 || !B.fp8) ? '不支持 FP8 计算的卡，FP8 权重只能减少读取的字节，计算仍按 BF16 算力。' : ''}${(!fits(A) || !fits(B)) ? '放不下的卡需要多卡部署，这里的下限按单卡的规格计算，只作对比参考。' : ''}</div>`;
    }

    container.querySelectorAll('[data-g]').forEach((s) => s.addEventListener('change', () => { st[s.dataset.g] = s.value; update(); }));
    container.querySelector('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    container.querySelector('[data-prec]').addEventListener('change', (e) => { st.fp8 = e.target.value === 'fp8'; update(); });
    update();
  });

  // ------------------------------------------------------------------ 0.6 显存账本
  W.register('memory-budget', function (container) {
    const st = { model: 'llama2-7b', gpu: 'a10', tp: 1, wb: 2, kvb: 2, logLen: 12, util: 0.9 };
    const OTHER = 2; // GB，激活等其他开销的经验值
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">显存账本</h4></div>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${modelSelect('mb-model', st.model)}</div>
          <div class="lw-control"><label>GPU</label>${gpuSelect('data-gpu', st.gpu)}</div>
          <div class="lw-control"><label>张量并行卡数</label><select data-tp><option>1</option><option>2</option><option>4</option><option>8</option></select></div>
          <div class="lw-control"><label>权重精度</label><select data-wb><option value="2">BF16</option><option value="1">FP8 / INT8</option><option value="0.52">INT4</option></select></div>
          <div class="lw-control"><label>KV Cache 精度</label><select data-kvb><option value="2">BF16</option><option value="1">FP8</option></select></div>
          <div class="lw-control"><label>每个请求的最大长度<span class="lw-val" data-v="len"></span></label><input type="range" min="10" max="17" step="1" value="${st.logLen}" data-len /></div>
        </div>
        <div data-out></div>
      </div>`;
    const out = container.querySelector('[data-out]');
    const $ = (s) => container.querySelector(s);

    function update() {
      const m = MODELS[st.model];
      const g = GPUS[st.gpu];
      const P = modelParams(m).total;
      const tp = st.tp;
      const wGB = (P * st.wb) / 1e9 / tp;
      const kvHeads = Math.max(1, m.nkv / tp); // KV 头数少于卡数时，各卡会复制 KV 头
      const kvTok = 2 * m.L * kvHeads * m.dh * st.kvb;
      const usable = g.mem * st.util;
      const budget = usable - wGB - OTHER;
      const maxLen = Math.pow(2, st.logLen);
      $('[data-v=len]').textContent = maxLen.toLocaleString('en-US') + ' tokens';
      const seg = (v, c, label) => (v > 0 ? `<div class="lw-stack-seg" style="width:${(v / g.mem) * 100}%;background:${W.color(c)}" title="${label} ${v.toFixed(1)} GB">${(v / g.mem) > 0.12 ? label : ''}</div>` : '');
      if (budget <= 0) {
        out.innerHTML = `
          <div class="lw-stack-bar" style="height:30px">${seg(Math.min(wGB, g.mem), 'ink-2', '权重')}</div>
          <div class="lw-callout err">每张卡要放 ${wGB.toFixed(1)} GB 权重，加上约 ${OTHER} GB 其他开销，超过了可用的 ${usable.toFixed(1)} GB（${g.mem} GB × ${st.util}）。可以增加张量并行的卡数，或者量化权重。</div>`;
        return;
      }
      const tokens = (budget * 1e9) / kvTok;
      const conc = tokens / maxLen;
      out.innerHTML = `
        <div class="lw-stack-bar" style="height:30px">
          ${seg(wGB, 'ink-2', '权重')}${seg(OTHER, 'series-4', '其他')}${seg(budget, 'prefill', 'KV Cache')}${seg(g.mem - usable, 'rule-strong', '预留')}
        </div>
        <div class="lw-legend">
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('ink-2')}"></i>权重 ${wGB.toFixed(1)} GB</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('series-4')}"></i>其他 ${OTHER} GB</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('prefill')}"></i>KV Cache ${budget.toFixed(1)} GB</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('rule-strong')}"></i>预留 ${(g.mem - usable).toFixed(1)} GB</span>
        </div>
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">每 token 的 KV（每卡）</div><div class="lw-stat-value">${(kvTok / 1024).toFixed(0)} KiB</div><div class="lw-stat-sub">2 × ${m.L} 层 × ${kvHeads} 个 KV 头 × ${m.dh} × ${st.kvb} 字节</div></div>
          <div class="lw-stat"><div class="lw-stat-label">能缓存的 token 数</div><div class="lw-stat-value">${Math.floor(tokens).toLocaleString('en-US')}</div></div>
          <div class="lw-stat ${conc < 1 ? 'err' : ''}"><div class="lw-stat-label">请求都用满最大长度时的并发数</div><div class="lw-stat-value">${conc.toFixed(1)}</div></div>
        </div>
        ${conc < 1 ? `<div class="lw-callout err">KV Cache 连一个满长度的请求都放不下，vLLM 会拒绝启动，提示减小 max_model_len。</div>` : ''}
        ${m.nkv < tp ? `<div class="lw-callout">${m.name} 只有 ${m.nkv} 个 KV 头，少于 ${tp} 张卡，每张卡至少要放一个完整的 KV 头，部分 KV 头会在多张卡上重复存放。</div>` : ''}`;
    }

    $('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    $('[data-tp]').addEventListener('change', (e) => { st.tp = Number(e.target.value); update(); });
    $('[data-wb]').addEventListener('change', (e) => { st.wb = Number(e.target.value); update(); });
    $('[data-kvb]').addEventListener('change', (e) => { st.kvb = Number(e.target.value); update(); });
    $('[data-len]').addEventListener('input', (e) => { st.logLen = Number(e.target.value); update(); });
    update();
  });

  // ------------------------------------------------------------------ 0.7 Roofline
  W.register('roofline', function (container) {
    const st = { gpu: 'a100', logB: 0, logS: 11 };
    const h = 4096;
    const g = 4; // Qwen3-8B：32 个 Q 头 / 8 个 KV 头
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">Roofline：Qwen3-8B 的三类计算</h4></div>
        <div class="lw-controls">
          <div class="lw-control"><label>GPU</label>${gpuSelect('data-gpu', st.gpu)}</div>
          <div class="lw-control"><label>decode 批大小 b<span class="lw-val" data-v="b"></span></label><input type="range" min="0" max="10" step="0.25" value="${st.logB}" data-b /></div>
          <div class="lw-control"><label>prefill 输入长度 s<span class="lw-val" data-v="s"></span></label><input type="range" min="4" max="14" step="0.25" value="${st.logS}" data-s /></div>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      const G = GPUS[st.gpu];
      const peak = G.bf16;
      const bw = G.bw; // TB/s -> TFLOPS = I * bw
      const ridge = peak / bw;
      const b = Math.round(Math.pow(2, st.logB));
      const s = Math.round(Math.pow(2, st.logS));
      $('[data-v=b]').textContent = b;
      $('[data-v=s]').textContent = s.toLocaleString('en-US');
      const pts = [
        { name: `decode 线性层（b = ${b}）`, I: (b * h) / (2 * b + h), c: 'decode' },
        { name: `prefill 线性层（s = ${s.toLocaleString('en-US')}）`, I: (s * h) / (2 * s + h), c: 'prefill' },
        { name: 'decode 注意力（任意 b）', I: g, c: 'err' },
      ].map((p) => ({ ...p, perf: Math.min(peak, p.I * bw) }));

      const Wd = 640, Hd = 300, L = 52, R = 14, T = 12, B = 36;
      const xMin = Math.log10(0.5), xMax = Math.log10(20000);
      const yMin = Math.log10(0.5), yMax = Math.log10(3000);
      const X = (I) => L + ((Math.log10(I) - xMin) / (xMax - xMin)) * (Wd - L - R);
      const Y = (p) => T + (1 - (Math.log10(p) - yMin) / (yMax - yMin)) * (Hd - T - B);
      const ink3 = W.color('ink-3'), rule = W.color('rule');
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="Roofline 图">`;
      [1, 10, 100, 1000, 10000].forEach((v) => {
        svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${Hd - B}" stroke="${rule}"/><text x="${X(v)}" y="${Hd - B + 16}" font-size="11" fill="${ink3}" text-anchor="middle">${v.toLocaleString('en-US')}</text>`;
      });
      [1, 10, 100, 1000].forEach((v) => {
        svg += `<line x1="${L}" x2="${Wd - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${rule}"/><text x="${L - 6}" y="${Y(v) + 4}" font-size="11" fill="${ink3}" text-anchor="end">${v}</text>`;
      });
      svg += `<text x="${(L + Wd - R) / 2}" y="${Hd - 4}" font-size="11" fill="${ink3}" text-anchor="middle">算术强度（FLOPs / Byte）</text>`;
      svg += `<text x="12" y="${(T + Hd - B) / 2}" font-size="11" fill="${ink3}" text-anchor="middle" transform="rotate(-90 12 ${(T + Hd - B) / 2})">TFLOPS</text>`;
      const x0 = 0.5, x1 = 20000;
      svg += `<polyline fill="none" stroke="${W.color('ink')}" stroke-width="2" points="${X(x0)},${Y(x0 * bw)} ${X(ridge)},${Y(peak)} ${X(x1)},${Y(peak)}"/>`;
      svg += `<line x1="${X(ridge)}" x2="${X(ridge)}" y1="${Y(peak)}" y2="${Hd - B}" stroke="${ink3}" stroke-dasharray="3 3"/>`;
      svg += `<text x="${X(ridge) + 5}" y="${Hd - B - 6}" font-size="11" fill="${W.color('ink-2')}">拐点 ${ridge.toFixed(0)}</text>`;
      svg += `<text x="${Wd - R}" y="${Y(peak) - 6}" font-size="11" fill="${W.color('ink-2')}" text-anchor="end">峰值 ${peak} TFLOPS</text>`;
      pts.forEach((p) => {
        svg += `<circle cx="${X(p.I)}" cy="${Y(p.perf)}" r="6.5" fill="${W.color(p.c)}" stroke="${W.color('paper')}" stroke-width="2"><title>${p.name}</title></circle>`;
      });
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          ${pts.map((p) => `<div class="lw-stat"><div class="lw-stat-label"><i class="lw-dot" style="background:${W.color(p.c)};margin-right:6px"></i>${p.name}</div>
            <div class="lw-stat-value">${p.perf < 10 ? p.perf.toFixed(1) : p.perf.toFixed(0)} TFLOPS</div>
            <div class="lw-stat-sub">强度 ${p.I < 10 ? p.I.toFixed(1) : p.I.toFixed(0)}，利用率 ${((p.perf / peak) * 100).toFixed(1)}%，${p.I < ridge ? '访存受限' : '计算受限'}</div></div>`).join('')}
        </div>`;
    }

    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    $('[data-b]').addEventListener('input', (e) => { st.logB = Number(e.target.value); update(); });
    $('[data-s]').addEventListener('input', (e) => { st.logS = Number(e.target.value); update(); });
    update();
  });

  // ------------------------------------------------------------------ 0.8 goodput
  // 一个简化的连续批处理引擎：每步时间 = 读权重 + 读 KV + prefill 计算。参数大致对应 A100 上的 Qwen3-8B。
  function simulateServe(rate, seed, n) {
    const r = rng(seed);
    const W_MS = 8.0;           // 读一遍 16.4 GB 权重
    const KV_MS_PER_K = 0.072;  // 每 1000 个缓存 token 的读取时间
    const PF_MS_PER_TOK = 0.0875; // 每个 prefill token 的计算时间（60% MFU）
    const BUDGET = 2048, MAX_SEQS = 256, KV_CAP = 360000;
    const reqs = [];
    let t = 0;
    for (let i = 0; i < n; i++) {
      t += (-Math.log(1 - r()) / rate) * 1000;
      reqs.push({ arrive: t, inp: 512 + Math.floor(r() * 1025), out: 128 + Math.floor(r() * 257), done: 0, gen: 0, first: null, end: null });
    }
    let clock = 0, next = 0;
    const waiting = [], running = [];
    let finished = 0, guard = 0;
    while (finished < n && guard++ < 200000) {
      while (next < n && reqs[next].arrive <= clock) waiting.push(reqs[next++]);
      if (!running.length && !waiting.length) { clock = reqs[next].arrive; continue; }
      let budget = BUDGET, pf = 0, kvTokens = 0;
      const decoders = [], prefills = [];
      running.forEach((q) => {
        if (q.done >= q.inp) { decoders.push(q); budget -= 1; }
      });
      running.forEach((q) => {
        if (q.done < q.inp && budget > 0) { const c = Math.min(budget, q.inp - q.done); prefills.push([q, c]); budget -= c; pf += c; }
      });
      let used = running.reduce((a, q) => a + q.inp + q.out, 0);
      while (waiting.length && budget > 0 && running.length < MAX_SEQS && used + waiting[0].inp + waiting[0].out <= KV_CAP) {
        const q = waiting.shift();
        running.push(q);
        used += q.inp + q.out;
        const c = Math.min(budget, q.inp);
        prefills.push([q, c]); budget -= c; pf += c;
      }
      running.forEach((q) => { kvTokens += q.done + q.gen; });
      const dt = W_MS + (KV_MS_PER_K * kvTokens) / 1000 + PF_MS_PER_TOK * pf;
      clock += dt;
      decoders.forEach((q) => { q.gen += 1; });
      prefills.forEach(([q, c]) => { q.done += c; if (q.done >= q.inp) { q.gen = 1; q.first = clock; } });
      for (let i = running.length - 1; i >= 0; i--) {
        const q = running[i];
        if (q.first !== null && q.gen >= q.out) { q.end = clock; running.splice(i, 1); finished++; }
      }
    }
    return reqs.map((q) => ({ ttft: q.first - q.arrive, tpot: (q.end - q.first) / Math.max(1, q.out - 1), arrive: q.arrive, end: q.end, out: q.out }));
  }

  W.register('goodput-sim', function (container) {
    const RATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16];
    const N = 400;
    const results = RATES.map((rt) => simulateServe(rt, 17 + rt, N));
    const st = { idx: 5, ttftSlo: 1000, tpotSlo: 50 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">请求速率、延迟与 goodput</h4></div>
        <div class="lw-controls">
          <div class="lw-control"><label>请求到达速率<span class="lw-val" data-v="rate"></span></label><input type="range" min="0" max="${RATES.length - 1}" step="1" value="${st.idx}" data-rate /></div>
          <div class="lw-control"><label>TTFT 上限<span class="lw-val" data-v="ttft"></span></label><input type="range" min="300" max="5000" step="100" value="${st.ttftSlo}" data-ttft /></div>
          <div class="lw-control"><label>TPOT 上限<span class="lw-val" data-v="tpot"></span></label><input type="range" min="20" max="120" step="5" value="${st.tpotSlo}" data-tpot /></div>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function summarize(res) {
      const dur = (Math.max(...res.map((q) => q.end)) - Math.min(...res.map((q) => q.arrive))) / 1000;
      const ok = res.filter((q) => q.ttft <= st.ttftSlo && q.tpot <= st.tpotSlo).length;
      return {
        thr: res.length / dur,
        good: ok / dur,
        okFrac: ok / res.length,
        tok: res.reduce((a, q) => a + q.out, 0) / dur,
        ttft50: quantile(res.map((q) => q.ttft), 0.5), ttft99: quantile(res.map((q) => q.ttft), 0.99),
        tpot50: quantile(res.map((q) => q.tpot), 0.5), tpot99: quantile(res.map((q) => q.tpot), 0.99),
      };
    }

    function update() {
      const S = results.map(summarize);
      const cur = S[st.idx];
      $('[data-v=rate]').textContent = RATES[st.idx] + ' 个/秒';
      $('[data-v=ttft]').textContent = st.ttftSlo + ' ms';
      $('[data-v=tpot]').textContent = st.tpotSlo + ' ms';
      const Wd = 640, Hd = 220, L = 40, R = 12, T = 12, B = 34;
      const xMax = 16, yMax = Math.max(12, Math.ceil(Math.max(...S.map((s) => s.thr)) + 1));
      const X = (v) => L + (v / xMax) * (Wd - L - R);
      const Y = (v) => T + (1 - v / yMax) * (Hd - T - B);
      const ink3 = W.color('ink-3'), rule = W.color('rule');
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="完成速率与 goodput 随到达速率的变化">`;
      for (let v = 0; v <= xMax; v += 2) svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${Hd - B}" stroke="${rule}"/><text x="${X(v)}" y="${Hd - B + 15}" font-size="11" fill="${ink3}" text-anchor="middle">${v}</text>`;
      for (let v = 0; v <= yMax; v += 4) svg += `<text x="${L - 6}" y="${Y(v) + 4}" font-size="11" fill="${ink3}" text-anchor="end">${v}</text>`;
      svg += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(Math.min(xMax, yMax))}" y2="${Y(Math.min(xMax, yMax))}" stroke="${ink3}" stroke-dasharray="3 3"/>`;
      svg += `<text x="${(L + Wd - R) / 2}" y="${Hd - 4}" font-size="11" fill="${ink3}" text-anchor="middle">请求到达速率（个/秒）</text>`;
      const line = (key, c) => `<polyline fill="none" stroke="${W.color(c)}" stroke-width="2.2" points="${S.map((s, i) => `${X(RATES[i])},${Y(s[key])}`).join(' ')}"/>` +
        S.map((s, i) => `<circle cx="${X(RATES[i])}" cy="${Y(s[key])}" r="${i === st.idx ? 5 : 2.5}" fill="${W.color(c)}"/>`).join('');
      svg += line('thr', 'ink-2') + line('good', 'prefill');
      svg += `<line x1="${X(RATES[st.idx])}" x2="${X(RATES[st.idx])}" y1="${T}" y2="${Hd - B}" stroke="${W.color('decode')}" stroke-width="1.5"/>`;
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg + `<div class="lw-legend">
        <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('ink-2')}"></i>完成速率（请求/秒）</span>
        <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('prefill')}"></i>goodput（达标请求/秒）</span>
        <span class="lw-legend-item"><i class="lw-dot" style="background:transparent;border:1px dashed ${ink3}"></i>完成速率 = 到达速率</span></div>`;
      const sec = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + ' s' : ms.toFixed(0) + ' ms');
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">输出吞吐</div><div class="lw-stat-value">${cur.tok.toFixed(0)}</div><div class="lw-stat-sub">token/秒</div></div>
          <div class="lw-stat ${cur.ttft99 > st.ttftSlo ? 'err' : ''}"><div class="lw-stat-label">TTFT P50 / P99</div><div class="lw-stat-value">${sec(cur.ttft50)} / ${sec(cur.ttft99)}</div></div>
          <div class="lw-stat ${cur.tpot99 > st.tpotSlo ? 'err' : ''}"><div class="lw-stat-label">TPOT P50 / P99</div><div class="lw-stat-value">${cur.tpot50.toFixed(0)} / ${cur.tpot99.toFixed(0)} ms</div></div>
          <div class="lw-stat"><div class="lw-stat-label">达标请求比例</div><div class="lw-stat-value">${(cur.okFrac * 100).toFixed(0)}%</div></div>
        </div>
        <div class="lw-callout">${cur.okFrac > 0.95 ? '负载不高，几乎所有请求都达标，goodput 和完成速率基本重合。' : cur.thr < RATES[st.idx] * 0.9 ? '到达速率已经超过了系统的处理能力，请求在排队中不断堆积，TTFT 随测试时间持续增长。这时完成速率接近上限，goodput 却很低。' : '批变大后每步变慢，排队也开始出现。完成速率还在增长，但越来越多的请求超出了 SLO。'}</div>`;
    }

    $('[data-rate]').addEventListener('input', (e) => { st.idx = Number(e.target.value); update(); });
    $('[data-ttft]').addEventListener('input', (e) => { st.ttftSlo = Number(e.target.value); update(); });
    $('[data-tpot]').addEventListener('input', (e) => { st.tpotSlo = Number(e.target.value); update(); });
    update();
  });
})();
