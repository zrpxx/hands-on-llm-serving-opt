/**
 * 第 6 部分的交互组件
 * - fp4-block        6.2 INT4、MXFP4、NVFP4 的量化误差
 * - eplb-balance     6.3 专家负载与冗余专家
 * - afd-pingpong     6.4 Attention 与 FFN 分离后的微批流水线
 * - spec-confidence  6.5 按置信度决定草稿长度
 * - longctx-kv       6.6 长上下文的 KV 与每步读取量
 * - fp-order         6.7 累加顺序与批不变
 * - async-timeline   6.8 同步调度与异步调度
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  }
  function gauss(r) {
    const u = Math.max(r(), 1e-12), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const slider = (label, attr, min, max, step, val) => `<div class="lw-control"><label><span>${label}</span><span class="lw-val" data-v-${attr}></span></label><input type="range" min="${min}" max="${max}" step="${step}" value="${val}" data-${attr} /></div>`;
  const tabs = (attr, items) => `<div class="lw-tabs">${items.map(([v, t]) => `<button type="button" class="lw-tab" data-${attr}="${v}">${t}</button>`).join('')}</div>`;

  // ------------------------------------------------------------------ 6.2 FP4 格式
  W.register('fp4-block', function (container) {
    const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
    const toGrid = (x) => { const a = Math.abs(x); let best = 0; for (const g of E2M1) if (Math.abs(a - g) < Math.abs(a - best)) best = g; return Math.sign(x) * best; };
    const e4m3 = (s) => { s = Math.min(Math.max(s, 2 ** -9), 448); const e = Math.max(Math.floor(Math.log2(s)), -6); const step = 2 ** (e - 3); return Math.min(Math.max(Math.round(s / step) * step, 2 ** -9), 448); };
    const f16 = (s) => { const e = Math.floor(Math.log2(s)); const step = 2 ** (e - 10); return Math.round(s / step) * step; };
    const st = { mag: 20, frac: 1 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">4 位格式的量化误差</h4></div>
        <p class="lw-subtitle">模拟一个 256 × 2048 的激活矩阵：大部分通道服从标准正态分布，少数通道被放大，模拟 LLM 激活中的离群通道。分组沿着每一行进行。误差用 SQNR 衡量，每高 6 dB，误差的幅度约减半。</p>
        <div class="lw-controls">
          ${slider('离群通道放大倍数', 'mag', 1, 100, 1, st.mag)}
          ${slider('离群通道的比例（%）', 'frac', 0, 10, 0.5, st.frac)}
        </div>
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    const R = 256, C = 2048;
    const base = new Float64Array(R * C);
    const r0 = rng(11);
    for (let i = 0; i < base.length; i++) base[i] = gauss(r0);
    const order = Array.from({ length: C }, (_, i) => i);
    const r1 = rng(12);
    for (let i = C - 1; i > 0; i--) { const j = Math.floor(r1() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }

    function quant(x, fmt, gmax) {
      const n = fmt === 'int' ? 128 : fmt === 'mx' ? 32 : 16;
      let num = 0, den = 0;
      for (let s = 0; s < x.length; s += n) {
        let amax = 0;
        for (let i = s; i < s + n; i++) amax = Math.max(amax, Math.abs(x[i]));
        if (amax === 0) continue;
        let sc;
        if (fmt === 'int') sc = f16(amax / 7);
        else if (fmt === 'mx') sc = 2 ** (Math.floor(Math.log2(amax)) - 2);
        else { const g = gmax / (6 * 448); sc = e4m3(amax / 6 / g) * g; }
        for (let i = s; i < s + n; i++) {
          const q = fmt === 'int' ? Math.max(-8, Math.min(7, Math.round(x[i] / sc))) * sc : toGrid(Math.max(-6, Math.min(6, x[i] / sc))) * sc;
          num += x[i] * x[i]; den += (x[i] - q) ** 2;
        }
      }
      return 10 * Math.log10(num / den);
    }
    function update() {
      $('[data-v-mag]').textContent = st.mag + ' 倍';
      $('[data-v-frac]').textContent = st.frac + '%';
      const hot = new Set(order.slice(0, Math.round((C * st.frac) / 100)));
      const x = new Float64Array(R * C);
      let gmax = 0;
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) { const v = base[r * C + c] * (hot.has(c) ? st.mag : 1); x[r * C + c] = v; gmax = Math.max(gmax, Math.abs(v)); }
      const rows = [
        ['INT4，每 128 个一组，FP16 缩放', 4.125, quant(x, 'int', gmax)],
        ['MXFP4：E2M1，每 32 个一组，E8M0 缩放', 4.25, quant(x, 'mx', gmax)],
        ['NVFP4：E2M1，每 16 个一组，E4M3 缩放 + 张量级 FP32', 4.5, quant(x, 'nv', gmax)],
      ];
      const best = Math.max(...rows.map((r) => r[2]));
      $('[data-res]').innerHTML = `
        <div class="lw-table-wrap"><table class="lw-table">
          <thead><tr><th>格式</th><th>每个元素的位数</th><th>SQNR</th></tr></thead>
          <tbody>${rows.map((r) => `<tr${r[2] === best ? ' class="lw-row-hi"' : ''}><td>${r[0]}</td><td>${r[1].toFixed(3)}</td><td>${r[2].toFixed(1)} dB</td></tr>`).join('')}</tbody>
        </table></div>
        <div class="lw-callout">${st.frac === 0 || st.mag === 1
          ? '没有离群值时，三者相差不大：E2M1 的 15 个取值不是等间距的，对正态分布未必比 INT4 的等间距网格好。'
          : '有离群通道时，一个组里只要混进一个离群值，整组的缩放因子就被它撑大，其余元素只能落在最靠近 0 的几个格点上。组越小，被拖累的元素越少：NVFP4 每 16 个一组，几乎不受影响。MXFP4 的缩放因子只能是 2 的幂，最多浪费将近一位的精度。'}</div>`;
    }
    ['mag', 'frac'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 6.3 EPLB
  W.register('eplb-balance', function (container) {
    const st = { skew: 0.7, red: 1, gpus: 32 };
    const NE = 256;
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">专家负载与冗余专家</h4></div>
        <p class="lw-subtitle">256 个路由专家分到若干张卡上，每张卡原本放 256 / 卡数 个专家。专家的热度服从 Zipf 分布。EPLB 给每张卡多留几个槽位：先把副本分给最热的专家，副本平分这个专家的 token；再从负载最大的副本开始，依次放进当前最空的卡。图中每根柱子是一张卡的负载，虚线是平均值。</p>
        <div class="lw-controls">
          ${slider('热度不均的程度（Zipf 指数）', 'skew', 0, 1.5, 0.05, st.skew)}
          ${slider('每卡的冗余槽位', 'red', 0, 8, 1, st.red)}
          <div class="lw-control"><label>卡数</label><select data-gpus>${[8, 16, 32, 64].map((v) => `<option ${v === st.gpus ? 'selected' : ''}>${v}</option>`).join('')}</select></div>
        </div>
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    const perm = Array.from({ length: NE }, (_, i) => i);
    const r0 = rng(7);
    for (let i = NE - 1; i > 0; i--) { const j = Math.floor(r0() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    function bars(loads, mx, color, label) {
      const Wd = 640, H = 90, bw = (Wd - 20) / loads.length;
      const mean = loads.reduce((a, b) => a + b, 0) / loads.length;
      let s = `<svg viewBox="0 0 ${Wd} ${H + 18}" width="100%" role="img" aria-label="${label}">`;
      loads.forEach((v, i) => { s += `<rect x="${10 + i * bw}" y="${H - (v / mx) * (H - 6)}" width="${Math.max(1, bw - 1)}" height="${(v / mx) * (H - 6)}" fill="${color}" opacity="0.85"/>`; });
      const y = H - (mean / mx) * (H - 6);
      s += `<line x1="10" x2="${Wd - 10}" y1="${y}" y2="${y}" stroke="${W.color('ink')}" stroke-dasharray="4 3"/>`;
      s += `<text x="10" y="${H + 14}" font-size="11" fill="${W.color('ink-3')}">${label}</text></svg>`;
      return s;
    }
    function update() {
      $('[data-v-skew]').textContent = st.skew.toFixed(2);
      $('[data-v-red]').textContent = st.red;
      const pop = new Array(NE);
      for (let i = 0; i < NE; i++) pop[perm[i]] = 1 / Math.pow(i + 1, st.skew);
      const sum = pop.reduce((a, b) => a + b, 0);
      const load = pop.map((p) => p / sum);
      const per = NE / st.gpus;
      const stat = Array.from({ length: st.gpus }, (_, g) => load.slice(g * per, (g + 1) * per).reduce((a, b) => a + b, 0));
      const slots = per + st.red;
      const rep = new Array(NE).fill(1);
      for (let k = 0; k < st.gpus * st.red; k++) {
        let best = 0, bv = -1;
        for (let e = 0; e < NE; e++) { const v = load[e] / rep[e]; if (v > bv) { bv = v; best = e; } }
        rep[best]++;
      }
      const items = [];
      for (let e = 0; e < NE; e++) for (let k = 0; k < rep[e]; k++) items.push([load[e] / rep[e], e]);
      items.sort((a, b) => b[0] - a[0]);
      const gl = new Array(st.gpus).fill(0), gn = new Array(st.gpus).fill(0), gs = Array.from({ length: st.gpus }, () => new Set());
      for (const [w, e] of items) {
        let g = -1;
        for (let i = 0; i < st.gpus; i++) if (gn[i] < slots && !gs[i].has(e) && (g < 0 || gl[i] < gl[g])) g = i;
        if (g < 0) for (let i = 0; i < st.gpus; i++) if (gn[i] < slots && (g < 0 || gl[i] < gl[g])) g = i;
        gl[g] += w; gn[g]++; gs[g].add(e);
      }
      const bal = (a) => a.reduce((x, y) => x + y, 0) / a.length / Math.max(...a);
      const mx = Math.max(...stat, ...gl);
      const hotShare = Math.max(...load) * NE;
      $('[data-res]').innerHTML = `
        ${bars(stat, mx, W.color('series-1'), '静态放置：按编号均匀分配')}
        ${bars(gl, mx, W.color('prefill'), `EPLB：每卡 ${per} 个专家 + ${st.red} 个冗余槽位`)}
        <div class="lw-stats">
          <div class="lw-stat warn"><div class="lw-stat-label">静态放置的均衡度</div><div class="lw-stat-value">${bal(stat).toFixed(2)}</div><div class="lw-stat-sub">平均负载 / 最大负载</div></div>
          <div class="lw-stat ok"><div class="lw-stat-label">EPLB 的均衡度</div><div class="lw-stat-value">${bal(gl).toFixed(2)}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">最热的专家</div><div class="lw-stat-value">${hotShare.toFixed(1)} 倍</div><div class="lw-stat-sub">相对于平均专家的负载</div></div>
        </div>
        <div class="lw-callout">MoE 层要等最忙的那张卡算完，均衡度就是算力的有效利用率。${st.red === 0 ? '没有冗余槽位时，EPLB 只能调整专家的位置，把热专家和冷专家搭配在一起；单个专家本身太热时，无论放在哪张卡都是瓶颈。' : `冗余副本把最热的专家分摊到多张卡上，代价是每卡多存 ${st.red} 个专家的权重。这里假设热度已知且不变；实际中热度随请求内容漂移，要定期重新统计、重新放置。`}</div>`;
    }
    ['skew', 'red'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    $('[data-gpus]').addEventListener('change', (e) => { st.gpus = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 6.4 AFD
  W.register('afd-pingpong', function (container) {
    const st = { ta: 3, tf: 2, comm: 1, mb: 2 };
    const LAYERS = 4;
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">Attention 与 FFN 分离后的微批流水线</h4></div>
        <p class="lw-subtitle">一组卡只算 Attention，另一组只算 FFN（MoE）。每一层的数据要先发给 FFN 一侧，算完再发回来。只有一个 batch 时，一侧计算，另一侧就在等。把 batch 切成几个微批，就能让两侧交替工作，同时用计算掩盖通信。图中画了 ${LAYERS} 层，时间单位任意。</p>
        <div class="lw-controls">
          ${slider('Attention 每层耗时（整个 batch）', 'ta', 1, 10, 0.5, st.ta)}
          ${slider('FFN 每层耗时（整个 batch）', 'tf', 1, 10, 0.5, st.tf)}
          ${slider('单向通信耗时（每个微批）', 'comm', 0, 4, 0.25, st.comm)}
          ${slider('微批数', 'mb', 1, 4, 1, st.mb)}
        </div>
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      ['ta', 'tf', 'comm', 'mb'].forEach((k) => { $(`[data-v-${k}]`).textContent = st[k]; });
      const m = st.mb, ta = st.ta / m, tf = st.tf / m, c = st.comm;
      let aFree = 0, fFree = 0;
      const ready = new Array(m).fill(0);
      const segs = [];
      for (let l = 0; l < LAYERS; l++) {
        for (let j = 0; j < m; j++) {
          const as = Math.max(aFree, ready[j]); aFree = as + ta; segs.push(['A', as, ta, j]);
          const fs = Math.max(fFree, aFree + c); fFree = fs + tf; segs.push(['F', fs, tf, j]);
          if (c > 0) { segs.push(['C1', aFree, c, j]); segs.push(['C2', fFree, c, j]); }
          ready[j] = fFree + c;
        }
      }
      const total = Math.max(...ready);
      const Wd = 640, lane = 26, L = 92;
      const X = (t) => L + (t / total) * (Wd - L - 10);
      const lanes = { A: 8, C1: 40, F: 72, C2: 104 };
      const names = { A: 'Attention 卡', C1: '发往 FFN', F: 'FFN 卡', C2: '发回 Attention' };
      const cols = [W.color('prefill'), W.color('decode'), W.color('series-2'), W.color('series-3')];
      let svg = `<svg viewBox="0 0 ${Wd} 150" width="100%" role="img" aria-label="流水线时间线">`;
      Object.entries(lanes).forEach(([k, y]) => { svg += `<text x="4" y="${y + 17}" font-size="11" fill="${W.color('ink-2')}">${names[k]}</text><line x1="${L}" x2="${Wd - 10}" y1="${y + lane}" y2="${y + lane}" stroke="${W.color('rule')}"/>`; });
      segs.forEach(([k, s, d, j]) => {
        const comm = k[0] === 'C';
        svg += `<rect x="${X(s)}" y="${lanes[k] + (comm ? 6 : 0)}" width="${Math.max(1, X(s + d) - X(s) - 0.5)}" height="${comm ? lane - 12 : lane}" fill="${cols[j]}" opacity="${comm ? 0.45 : 0.85}"/>`;
      });
      svg += `<text x="${Wd - 10}" y="146" font-size="11" fill="${W.color('ink-3')}" text-anchor="end">总耗时 ${total.toFixed(1)}，颜色区分微批</text></svg>`;
      const ua = (LAYERS * st.ta) / total, uf = (LAYERS * st.tf) / total;
      const ideal = LAYERS * Math.max(st.ta, st.tf);
      $('[data-res]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">Attention 卡的利用率</div><div class="lw-stat-value">${(ua * 100).toFixed(0)}%</div></div>
          <div class="lw-stat"><div class="lw-stat-label">FFN 卡的利用率</div><div class="lw-stat-value">${(uf * 100).toFixed(0)}%</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">总耗时 / 理想值</div><div class="lw-stat-value">${(total / ideal).toFixed(2)}</div><div class="lw-stat-sub">理想值：较慢一侧一直在算</div></div>
        </div>
        <div class="lw-callout">${m === 1 ? '只有一个 batch 时，两侧轮流空闲，通信也完全暴露在关键路径上。' : Math.abs(st.ta - st.tf) / Math.max(st.ta, st.tf) > 0.3 ? '两侧的耗时相差较大，较快的一侧仍然要等。实际部署中要调整 Attention 卡和 FFN 卡的数量之比，让两侧每层的耗时接近。' : st.comm * 2 > Math.max(st.ta, st.tf) / m * (m - 1) ? '微批变小后，每个微批的计算已经藏不住通信，继续增加微批数也没有用，还会让每次计算的效率下降。' : '两侧耗时接近、通信被计算掩盖时，两边都接近满负荷。这就是 MegaScale-Infer 中 ping-pong 流水线的思路。'}</div>`;
    }
    ['ta', 'tf', 'comm', 'mb'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 6.5 按置信度验证
  W.register('spec-confidence', function (container) {
    const st = { logb: 4, noise: 0.15 };
    const DECAY = [1, 0.95, 0.9, 0.86, 0.82];
    const T_MEM = 4.8, T_TOK = (2 * 8e9) / (989e12 * 0.6) * 1e3, DRAFT = 0.15;
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">固定草稿长度，还是按置信度决定</h4></div>
        <p class="lw-subtitle">与 <code>code/stage6/mtp_sim.py</code> 的模型相同：8B 模型、H100，读一遍权重约 4.8 ms，一步验证的 token 总数超过约 180 时开始受算力限制。每段内容好猜的程度不同，草稿模块给出带噪声的置信度。折线是按置信度决定草稿长度时的加速比随阈值的变化，横线是固定 K 中最好的一个。</p>
        <div class="lw-controls">
          ${slider('batch 大小', 'logb', 0, 9, 1, st.logb)}
          ${slider('置信度估计的噪声', 'noise', 0, 0.4, 0.05, st.noise)}
        </div>
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function samples() {
      const r = rng(3), out = [];
      const beta = () => { // Beta(1.6, 0.6)，Jöhnk 方法
        for (;;) { const a = Math.pow(r(), 1 / 1.6), b = Math.pow(r(), 1 / 0.6); if (a + b <= 1 && a + b > 0) return a / (a + b); }
      };
      for (let i = 0; i < 3000; i++) {
        const e = beta();
        const a = DECAY.map((d) => Math.min(0.99, Math.max(0.01, e * d + gauss(r) * 0.05)));
        const z = DECAY.map(() => gauss(r));
        out.push({ a, z });
      }
      return out;
    }
    const S = samples();
    function evalPolicy(B, kOf) {
      let prod = 0, ver = 0, kmax = 0;
      for (const s of S) {
        const k = kOf(s);
        let p = 1, acc = 0;
        for (let i = 0; i < k; i++) { p *= s.a[i]; acc += p; }
        prod += acc + 1; ver += k + 1; kmax = Math.max(kmax, k);
      }
      const n = S.length;
      // 草稿逐个生成，一步的草稿耗时取决于 batch 中最长的草稿；batch 很小时近似为平均长度
      const draftLen = B < 4 ? ver / n - 1 : kmax;
      const t = Math.max(T_MEM, (B * ver) / n * T_TOK) + 0.3 + DRAFT * draftLen;
      return (B * prod) / n / t;
    }
    function update() {
      const B = 2 ** st.logb;
      $('[data-v-logb]').textContent = B;
      $('[data-v-noise]').textContent = st.noise.toFixed(2);
      const base = evalPolicy(B, () => 0);
      const fixed = [1, 2, 3, 4, 5].map((k) => [k, evalPolicy(B, () => k) / base]);
      const bestFixed = fixed.reduce((a, b) => (b[1] > a[1] ? b : a));
      const ths = []; for (let t = 0.05; t <= 0.951; t += 0.05) ths.push(+t.toFixed(2));
      const conf = ths.map((th) => [th, evalPolicy(B, (s) => { let k = 0, cum = 1; while (k < 5) { const c = Math.min(0.99, Math.max(0.01, s.a[k] + s.z[k] * st.noise)); if (cum * c < th) break; cum *= c; k++; } return k; }) / base]);
      const bestConf = conf.reduce((a, b) => (b[1] > a[1] ? b : a));
      const Wd = 640, H = 170, L = 40, Bm = 140;
      const ymax = Math.max(1.2, bestFixed[1], bestConf[1]) * 1.08, ymin = Math.min(0.4, ...conf.map((c) => c[1])) * 0.95;
      const X = (t) => L + ((t - 0.05) / 0.9) * (Wd - L - 20);
      const Y = (v) => Bm - ((v - ymin) / (ymax - ymin)) * (Bm - 10);
      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="加速比随阈值的变化">`;
      [ymin, 1, ymax].forEach((v) => { svg += `<text x="${L - 6}" y="${Y(v) + 4}" font-size="10" fill="${W.color('ink-3')}" text-anchor="end">${v.toFixed(1)}</text>`; });
      svg += `<line x1="${L}" x2="${Wd - 20}" y1="${Y(1)}" y2="${Y(1)}" stroke="${W.color('rule-strong')}"/>`;
      svg += `<line x1="${L}" x2="${Wd - 20}" y1="${Y(bestFixed[1])}" y2="${Y(bestFixed[1])}" stroke="${W.color('decode')}" stroke-dasharray="5 4"/>`;
      svg += `<text x="${Wd - 22}" y="${Y(bestFixed[1]) - 5}" font-size="11" fill="${W.color('decode-ink')}" text-anchor="end">固定 K = ${bestFixed[0]}</text>`;
      svg += `<polyline fill="none" stroke="${W.color('prefill')}" stroke-width="2" points="${conf.map(([t, v]) => `${X(t)},${Y(v)}`).join(' ')}"/>`;
      svg += `<circle cx="${X(bestConf[0])}" cy="${Y(bestConf[1])}" r="4" fill="${W.color('prefill')}"/>`;
      [0.1, 0.3, 0.5, 0.7, 0.9].forEach((t) => { svg += `<text x="${X(t)}" y="${Bm + 14}" font-size="10" fill="${W.color('ink-3')}" text-anchor="middle">${t}</text>`; });
      svg += `<text x="${(L + Wd) / 2}" y="${Bm + 28}" font-size="11" fill="${W.color('ink-3')}" text-anchor="middle">置信度阈值（累计置信度低于它就停止起草）；纵轴：相对不用投机的加速比</text></svg>`;
      const verdict = Math.max(bestFixed[1], bestConf[1]) < 1;
      $('[data-res]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">最好的固定 K</div><div class="lw-stat-value">${bestFixed[1].toFixed(2)} 倍</div><div class="lw-stat-sub">K = ${bestFixed[0]}</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">按置信度，最好的阈值</div><div class="lw-stat-value">${bestConf[1].toFixed(2)} 倍</div><div class="lw-stat-sub">阈值 ${bestConf[0]}</div></div>
        </div>
        <div class="lw-callout">${verdict ? 'batch 已经大到 GPU 受算力限制，验证任何草稿都要付出与不用投机时相同的计算，投机解码本身不再划算。阈值调到最高（几乎不起草），按置信度的策略会退化为不用投机。' : B <= 16 ? 'batch 小时 GPU 受带宽限制，多验证几个 token 几乎不花时间，草稿越长越好，阈值越低越好。' : '随着 batch 增大，最优的固定 K 变小，最优的阈值变高：被拒绝的草稿开始占用宝贵的算力。按置信度决定，把验证留给有把握的序列。'}${st.noise >= 0.3 ? '置信度的噪声很大时，它提供的信息有限，与固定 K 的差距也随之缩小。' : ''}</div>`;
    }
    $('[data-logb]').addEventListener('input', (e) => { st.logb = +e.target.value; update(); });
    $('[data-noise]').addEventListener('input', (e) => { st.noise = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 6.6 长上下文
  W.register('longctx-kv', function (container) {
    const MD = {
      llama8b: { name: 'Llama-3.1-8B（GQA）', kv: 32 * 2 * 8 * 128 * 2, idx: 0 },
      qwen235: { name: 'Qwen3-235B-A22B（GQA）', kv: 94 * 2 * 4 * 128 * 2, idx: 0 },
      dsv3: { name: 'DeepSeek-V3（MLA）', kv: 61 * 576 * 2, idx: 0 },
      dsv32: { name: 'DeepSeek-V3.2（MLA + DSA）', kv: 61 * 576 * 2, idx: 61 * (128 + 4) },
    };
    const st = { model: 'dsv32', logc: 17 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">上下文变长时，KV 有多大、每步要读多少</h4></div>
        <p class="lw-subtitle">KV 按 BF16 计。decode 每生成一个 token，完整注意力要把这条序列的全部 KV 读一遍；DSA 先用索引器为每个位置打分（索引键为 FP8 的 128 维，加一个缩放因子），只对选出的 2048 个位置读取完整的 KV。滑动窗口按 4096 计。只算每条序列的注意力部分，不含权重。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label><select data-model>${Object.keys(MD).map((k) => `<option value="${k}" ${k === st.model ? 'selected' : ''}>${MD[k].name}</option>`).join('')}</select></div>
          ${slider('上下文长度', 'logc', 12, 20, 0.5, st.logc)}
        </div>
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    const fmt = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${(b / 2 ** 20).toFixed(0)} MB`);
    function update() {
      const L = Math.round(2 ** st.logc);
      $('[data-v-logc]').textContent = L >= 2 ** 20 ? `${(L / 2 ** 20).toFixed(1)}M` : `${Math.round(L / 1024)}K`;
      const m = MD[st.model];
      const kvSeq = m.kv * L;
      const full = kvSeq;
      const swa = m.kv * Math.min(L, 4096);
      const dsa = m.idx ? m.idx * L + m.kv * Math.min(L, 2048) : null;
      const bw = 3.35e12;
      $('[data-res]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">这条序列的 KV</div><div class="lw-stat-value">${fmt(kvSeq)}</div><div class="lw-stat-sub">一张 80 GB 的卡约能放 ${Math.max(0, Math.floor(70e9 / kvSeq))} 条（不含权重）</div></div>
          <div class="lw-stat warn"><div class="lw-stat-label">完整注意力：每步读取</div><div class="lw-stat-value">${fmt(full)}</div><div class="lw-stat-sub">H800 上约 ${((full / bw) * 1e3).toFixed(2)} ms</div></div>
          <div class="lw-stat ${dsa ? 'ok' : ''}"><div class="lw-stat-label">${dsa ? 'DSA：每步读取' : '滑动窗口：每步读取'}</div><div class="lw-stat-value">${fmt(dsa || swa)}</div><div class="lw-stat-sub">${dsa ? `其中索引键 ${fmt(m.idx * L)}` : '只看最近 4096 个 token'}</div></div>
        </div>
        <div class="lw-callout">${dsa ? `DSA 并没有消除与长度成正比的部分：索引器仍要给每个位置打分，但每个位置只需读约 ${(m.idx / 61).toFixed(0)} 字节，而不是完整的 ${(m.kv / 61).toFixed(0)} 字节。上下文越长，节省越明显。注意 DSA 只减少了读取和计算，KV 本身仍要全部存着。` : '每生成一个 token 都要读一遍整条序列的 KV，长上下文的 decode 很快就被 KV 的读取主导。滑动窗口读得少，但看不到窗口之外的内容；稀疏注意力按相关性挑选位置，试试 DeepSeek-V3.2。'}</div>`;
    }
    $('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    $('[data-logc]').addEventListener('input', (e) => { st.logc = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 6.7 累加顺序
  W.register('fp-order', function (container) {
    const st = { b: 1, inv: false };
    const N = 64;
    const r = rng(21);
    const vals = Array.from({ length: N }, () => Math.fround(gauss(r) * Math.pow(10, Math.floor(r() * 5) - 1)));
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">同一行点积，换一个 batch 就变了</h4></div>
        <p class="lw-subtitle">矩阵乘法中，输出的每个元素都是一个长度为 K 的点积。这里取 K = 64 个乘积（量级从 0.1 到 1000 不等），用 FP32 累加。GPU 的 kernel 会根据 batch 大小选择不同的切分：batch 小时把 K 切成多段并行累加（split-K），再把各段的部分和相加。</p>
        <div class="lw-controls">${slider('这一步的 batch 大小', 'b', 1, 512, 1, st.b)}</div>
        ${tabs('m', [['0', '分段数由 batch 决定'], ['1', '批不变：始终分 4 段']])}
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function dot(nSplit) {
      const per = N / nSplit, parts = [];
      for (let p = 0; p < nSplit; p++) { let acc = 0; for (let i = p * per; i < (p + 1) * per; i++) acc = Math.fround(acc + vals[i]); parts.push(acc); }
      let tot = 0; for (const x of parts) tot = Math.fround(tot + x);
      return tot;
    }
    const heur = (b) => (b < 8 ? 16 : b < 32 ? 8 : b < 128 ? 4 : b < 256 ? 2 : 1);
    function update() {
      $('[data-v-b]').textContent = st.b;
      container.querySelectorAll('[data-m]').forEach((x) => x.classList.toggle('active', (x.dataset.m === '1') === st.inv));
      const ns = st.inv ? 4 : heur(st.b);
      const ref = st.inv ? dot(4) : dot(heur(1));
      const cur = dot(ns);
      const rows = [1, 8, 32, 128, 256, 512].map((b) => { const n = st.inv ? 4 : heur(b); return [b, n, dot(n)]; });
      $('[data-res]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">当前切分</div><div class="lw-stat-value">${ns} 段</div><div class="lw-stat-sub">每段 ${N / ns} 个乘积</div></div>
          <div class="lw-stat ${cur === ref ? 'ok' : 'warn'}"><div class="lw-stat-label">结果</div><div class="lw-stat-value" style="font-size:1rem">${cur.toPrecision(9)}</div><div class="lw-stat-sub">${cur === ref ? '与 batch = 1 时完全相同' : `与 batch = 1 时相差 ${Math.abs(cur - ref).toExponential(1)}`}</div></div>
        </div>
        <div class="lw-table-wrap"><table class="lw-table">
          <thead><tr><th>batch</th><th>切成几段</th><th>这一行的结果</th></tr></thead>
          <tbody>${rows.map(([b, n, v]) => `<tr${v !== ref ? ' class="lw-row-hi"' : ''}><td>${b}</td><td>${n}</td><td><code>${v.toPrecision(9)}</code></td></tr>`).join('')}</tbody>
        </table></div>
        <div class="lw-callout">${st.inv ? '无论 batch 多大都按同样的方式切分、按同样的顺序相加，同一行的结果就不再依赖于它和谁在一个 batch 里。代价是小 batch 时失去了多切几段带来的并行度，大 batch 时又多了一次合并。' : '每一种切分在数学上都等价，但浮点加法不满足结合律，相加的顺序不同，最后几位就不同。对用户来说，他的请求和谁被分到一个 batch 是随机的，于是温度为 0 的输出也会变。高亮的行与 batch = 1 时的结果不同。'}</div>`;
    }
    $('[data-b]').addEventListener('input', (e) => { st.b = +e.target.value; update(); });
    container.querySelectorAll('[data-m]').forEach((x) => x.addEventListener('click', () => { st.inv = x.dataset.m === '1'; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 6.8 异步调度
  W.register('async-timeline', function (container) {
    const st = { sched: 2.5, gpu: 8, post: 2 };
    const STEPS = 5;
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">同步调度与异步调度</h4></div>
        <p class="lw-subtitle">decode 的每一步：CPU 调度并准备输入，GPU 前向，CPU 后处理（拷回采样结果、检查停止条件、detokenize）。异步调度在 GPU 算第 N 步时，CPU 已经在准备第 N+1 步，并处理第 N−1 步的结果。下面画了 ${STEPS} 步。</p>
        <div class="lw-controls">
          ${slider('调度与准备输入（ms）', 'sched', 0, 10, 0.5, st.sched)}
          ${slider('GPU 前向（ms）', 'gpu', 1, 20, 0.5, st.gpu)}
          ${slider('后处理（ms）', 'post', 0, 10, 0.5, st.post)}
        </div>
        <div data-res></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      ['sched', 'gpu', 'post'].forEach((k) => { $(`[data-v-${k}]`).textContent = st[k]; });
      const { sched: s, gpu: g, post: p } = st;
      const sync = [], asy = [];
      let t = 0;
      for (let i = 0; i < STEPS; i++) { sync.push(['S', t, s, i]); t += s; sync.push(['G', t, g, i]); t += g; sync.push(['P', t, p, i]); t += p; }
      const tSync = t;
      // 异步：GPU 算第 i 步时，CPU 先调度第 i+1 步；第 i 步算完后立刻提交第 i+1 步，再处理第 i 步的结果
      let cpu = s, gs = s;
      asy.push(['S', 0, s, 0]);
      for (let i = 0; i < STEPS; i++) {
        asy.push(['G', gs, g, i]);
        const gEnd = gs + g;
        if (i + 1 < STEPS) { const ss = Math.max(cpu, gs); asy.push(['S', ss, s, i + 1]); cpu = ss + s; }
        const tt = Math.max(cpu, gEnd);
        asy.push(['P', tt, p, i]);
        cpu = tt + p;
        gs = tt;
      }
      const tAsync = Math.max(...asy.map(([, a, d]) => a + d));
      const total = Math.max(tSync, tAsync);
      const Wd = 640, L = 84, lane = 20;
      const X = (v) => L + (v / total) * (Wd - L - 10);
      const col = { S: W.color('series-1'), G: W.color('prefill'), P: W.color('decode') };
      const draw = (segs, y0, label) => {
        let o = `<text x="4" y="${y0 + 14}" font-size="11" fill="${W.color('ink-2')}">${label} CPU</text><text x="4" y="${y0 + 40}" font-size="11" fill="${W.color('ink-2')}">${label} GPU</text>`;
        segs.forEach(([k, a, d, i]) => {
          const y = k === 'G' ? y0 + 26 : y0;
          o += `<rect x="${X(a)}" y="${y}" width="${Math.max(0.5, X(a + d) - X(a) - 0.6)}" height="${lane}" fill="${col[k]}" opacity="${0.55 + 0.45 * ((i % 2) ? 0.6 : 1)}"/>`;
          if (X(a + d) - X(a) > 14) o += `<text x="${(X(a) + X(a + d)) / 2}" y="${y + 14}" font-size="10" fill="#fff" text-anchor="middle">${i + 1}</text>`;
        });
        return o;
      };
      let svg = `<svg viewBox="0 0 ${Wd} 130" width="100%" role="img" aria-label="同步与异步调度的时间线">`;
      svg += draw(sync, 4, '同步') + draw(asy, 66, '异步') + '</svg>';
      const legend = `<div class="lw-legend"><span class="lw-legend-item"><i class="lw-dot" style="background:${col.S}"></i>调度与准备输入</span><span class="lw-legend-item"><i class="lw-dot" style="background:${col.G}"></i>GPU 前向</span><span class="lw-legend-item"><i class="lw-dot" style="background:${col.P}"></i>后处理</span><span class="lw-legend-item">块上的数字是第几步</span></div>`;
      const stepS = tSync / STEPS, stepA = Math.max(g, s + p);
      $('[data-res]').innerHTML = `${legend}${svg}
        <div class="lw-stats">
          <div class="lw-stat warn"><div class="lw-stat-label">同步：每步</div><div class="lw-stat-value">${stepS.toFixed(1)} ms</div><div class="lw-stat-sub">GPU 利用率 ${((g / stepS) * 100).toFixed(0)}%</div></div>
          <div class="lw-stat ok"><div class="lw-stat-label">异步：稳定后每步</div><div class="lw-stat-value">${stepA.toFixed(1)} ms</div><div class="lw-stat-sub">GPU 利用率 ${((g / stepA) * 100).toFixed(0)}%</div></div>
        </div>
        <div class="lw-callout">${s + p > g ? 'CPU 每步的工作已经比 GPU 前向还长，异步调度也藏不住，每一步的耗时由 CPU 决定。这在小模型、大 batch、多卡 TP 时很常见，此时要减少 CPU 工作本身：把输入的准备搬到 GPU 上、用更快的数据结构、把 detokenize 放到另一个进程。' : '稳定之后，每一步的耗时等于 GPU 前向的时间，CPU 的工作全部藏在了 GPU 计算的背后。前提是第 N+1 步的调度不需要等第 N 步的采样结果：调度只决定哪些请求参与、分配哪些 KV 块，具体的 token 由 GPU 在上一步结束时直接写入下一步的输入。'}</div>`;
    }
    ['sched', 'gpu', 'post'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    update();
  });
})();
