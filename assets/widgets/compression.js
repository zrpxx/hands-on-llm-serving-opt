/**
 * 第 2 部分的交互组件
 * - quant-grid          2.1 数值格式、粒度与离群值
 * - gptq-steps          2.2 GPTQ 逐列量化与误差补偿
 * - awq-scale           2.3 AWQ 缩放因子的网格搜索
 * - smooth-migrate      2.4 SmoothQuant 把量化难度从激活迁移到权重
 * - kv-quant-axis       2.5 K 按通道、V 按 token 量化
 * - lowbit-gemm         2.6 不同量化方案的矩阵乘法耗时随 batch 的变化
 * - quant-chooser       2.7 按模型、显卡和目标选择量化方案
 * - sparse-24           2.8 2:4 结构化稀疏与剪枝准则
 *
 * 写法约定同 basics.js：控件只渲染一次，交互时只更新结果区域。
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

  // ------------------------------------------------------------------ 格式
  // 小浮点数的舍入：ebits 位指数、mbits 位尾数，含次正规数，超过 max 的截断到 max
  function minifloat(x, ebits, mbits, max) {
    const a = Math.abs(x);
    if (a === 0) return 0;
    const bias = Math.pow(2, ebits - 1) - 1;
    const e = Math.max(Math.floor(Math.log2(a)), 1 - bias);
    const step = Math.pow(2, e - mbits);
    return Math.sign(x) * Math.min(Math.round(a / step) * step, max);
  }
  const FORMATS = {
    int8: { name: 'INT8', max: 127, q: (v) => Math.max(-128, Math.min(127, Math.round(v))) },
    int4: { name: 'INT4', max: 7, q: (v) => Math.max(-8, Math.min(7, Math.round(v))) },
    fp8: { name: 'FP8 E4M3', max: 448, q: (v) => minifloat(v, 4, 3, 448) },
    fp4: { name: 'FP4 E2M1', max: 6, q: (v) => minifloat(v, 2, 1, 6) },
  };
  W.minifloat = minifloat;

  function levels(fmt) {
    const out = new Set();
    if (fmt.startsWith('int')) {
      const f = FORMATS[fmt];
      for (let i = -f.max - 1; i <= f.max; i++) out.add(i);
    } else {
      const f = FORMATS[fmt];
      for (let i = -2000; i <= 2000; i++) out.add(f.q((i / 2000) * f.max));
    }
    return [...out].sort((a, b) => a - b);
  }

  // ------------------------------------------------------------------ 2.1 量化网格
  W.register('quant-grid', function (container) {
    const ROWS = 8, COLS = 128;
    const st = { fmt: 'int4', gran: 'row', outlier: true };
    const base = [];
    const r = rng(3);
    for (let i = 0; i < ROWS; i++) {
      const row = [];
      for (let j = 0; j < COLS; j++) row.push(gauss(r) * 0.02);
      base.push(row);
    }

    function data() {
      return base.map((row, i) => row.map((v, j) => (st.outlier && j === 17 + i * 13 ? 0.02 * (6 + 3 * i) * (i % 2 ? -1 : 1) : v)));
    }

    function quantize(rows, fmt, gran) {
      const f = FORMATS[fmt];
      const groups = [];
      if (gran === 'tensor') groups.push(rows.flatMap((row, i) => row.map((_, j) => [i, j])));
      else if (gran === 'row') rows.forEach((row, i) => groups.push(row.map((_, j) => [i, j])));
      else rows.forEach((row, i) => { for (let g = 0; g < COLS; g += 32) groups.push(row.slice(g, g + 32).map((_, k) => [i, g + k])); });
      const out = rows.map((row) => row.slice());
      const scales = rows.map(() => []);
      groups.forEach((idx) => {
        const amax = Math.max(...idx.map(([i, j]) => Math.abs(rows[i][j]))) || 1e-8;
        const scale = amax / f.max;
        idx.forEach(([i, j]) => { out[i][j] = f.q(rows[i][j] / scale) * scale; scales[i][j] = scale; });
      });
      let num = 0, den = 0;
      rows.forEach((row, i) => row.forEach((v, j) => { num += (out[i][j] - v) ** 2; den += v * v; }));
      return { out, scales, err: num / den };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">数值格式、粒度与离群值</h4></div>
        <p class="lw-subtitle">8 行模拟权重，每行 128 个数，服从标准差 0.02 的正态分布。开启离群值后，每行有一个数被替换成 0.12～0.54 之间的大值，各行不同。</p>
        <div class="lw-tabs" data-fmts>${Object.entries(FORMATS).map(([k, f]) => `<button type="button" class="lw-tab" data-f="${k}">${f.name}</button>`).join('')}</div>
        <div class="lw-tabs" data-grans>
          <button type="button" class="lw-tab" data-g="tensor">按张量</button>
          <button type="button" class="lw-tab" data-g="row">按行（通道）</button>
          <button type="button" class="lw-tab" data-g="g32">按组，每组 32 个</button>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:0.84rem;margin-left:auto"><input type="checkbox" data-outlier checked /> 离群值</label>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      container.querySelectorAll('[data-f]').forEach((b) => b.classList.toggle('active', b.dataset.f === st.fmt));
      container.querySelectorAll('[data-g]').forEach((b) => b.classList.toggle('active', b.dataset.g === st.gran));
      const rows = data();
      const res = quantize(rows, st.fmt, st.gran);
      // 画第 0 行：值的直方图 + 第一组的量化等级
      const row = rows[0];
      const span = 0.08;
      const Wd = 640, Hd = 150, L = 24, R = 24;
      const X = (v) => L + ((Math.max(-span, Math.min(span, v)) + span) / (2 * span)) * (Wd - L - R);
      const bins = new Array(80).fill(0);
      row.forEach((v) => { const k = Math.floor(((Math.max(-span, Math.min(span - 1e-9, v)) + span) / (2 * span)) * 80); bins[k]++; });
      const bmax = Math.max(...bins);
      const scale = res.scales[0][0];
      const lv = levels(st.fmt).map((q) => q * scale).filter((v) => Math.abs(v) <= span);
      const ink3 = W.color('ink-3');
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="第 0 行的数值分布与量化等级">`;
      lv.forEach((v) => { svg += `<line x1="${X(v)}" x2="${X(v)}" y1="14" y2="112" stroke="${W.color('decode')}" stroke-width="1" opacity="0.75"/>`; });
      bins.forEach((c, k) => {
        if (!c) return;
        const h = (c / bmax) * 90;
        const x0 = L + (k / 80) * (Wd - L - R);
        svg += `<rect x="${x0 + 0.5}" y="${112 - h}" width="${(Wd - L - R) / 80 - 1}" height="${h}" fill="${W.color('prefill')}" opacity="0.8"/>`;
      });
      svg += `<line x1="${L}" x2="${Wd - R}" y1="112" y2="112" stroke="${W.color('rule-strong')}"/>`;
      [-0.08, -0.04, 0, 0.04, 0.08].forEach((v) => { svg += `<text x="${X(v)}" y="128" font-size="11" fill="${ink3}" text-anchor="middle">${v}</text>`; });
      const outside = row.filter((v) => Math.abs(v) > span).length;
      svg += `<text x="${L}" y="146" font-size="11" fill="${ink3}">蓝色：第 0 行的数值分布${outside ? `（另有 ${outside} 个数在 ±0.08 之外）` : ''}。琥珀色竖线：这一行第一组可以表示的值，共 ${lv.length} 个落在图中范围内。</text>`;
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const all = ['tensor', 'row', 'g32'].map((g) => ({ g, e: quantize(rows, st.fmt, g).err }));
      const gName = { tensor: '按张量', row: '按行', g32: '按组 32' };
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          ${all.map((x) => `<div class="lw-stat ${x.g === st.gran ? 'info' : ''}"><div class="lw-stat-label">${FORMATS[st.fmt].name}，${gName[x.g]}</div><div class="lw-stat-value">${x.e.toExponential(2)}</div><div class="lw-stat-sub">相对均方误差</div></div>`).join('')}
        </div>
        <div class="lw-callout">${st.outlier && st.gran !== 'g32'
          ? '离群值决定了 scale：为了不截断它，等级被拉得很开，大多数数值附近只剩下很少几个等级可用。'
          : st.gran === 'g32' ? '按组量化时，离群值只影响它所在的那一组，其他组的 scale 不受影响。' : '没有离群值时，数值集中在一个较小的范围内，等级都能用上。'}${st.fmt.startsWith('fp') ? ' 浮点格式的等级在 0 附近更密，对钟形分布更友好。' : ''}</div>`;
    }

    container.querySelectorAll('[data-f]').forEach((b) => b.addEventListener('click', () => { st.fmt = b.dataset.f; update(); }));
    container.querySelectorAll('[data-g]').forEach((b) => b.addEventListener('click', () => { st.gran = b.dataset.g; update(); }));
    $('[data-outlier]').addEventListener('change', (e) => { st.outlier = e.target.checked; update(); });
    update();
  });

  // ------------------------------------------------------------------ 线性代数小工具
  function matInv(A) {
    const n = A.length;
    const M = A.map((row, i) => row.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c];
        if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map((row) => row.slice(n));
  }

  // ------------------------------------------------------------------ 2.2 GPTQ
  W.register('gptq-steps', function (container) {
    const D = 16, N = 96, BITS = 3, QMAX = 3;
    const r = rng(11);
    const X = [];
    for (let s = 0; s < N; s++) {
      const row = [];
      for (let k = 0; k < D / 2; k++) {
        const a = gauss(r);
        row.push(a, 0.92 * a + 0.4 * gauss(r));     // 第 2k 与 2k+1 个通道高度相关
      }
      X.push(row);
    }
    const w0 = Array.from({ length: D }, () => gauss(r));
    const scale = Math.max(...w0.map(Math.abs)) / QMAX;
    const quant = (v) => Math.max(-QMAX - 1, Math.min(QMAX, Math.round(v / scale))) * scale;
    const H = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => (2 / N) * X.reduce((acc, x) => acc + x[i] * x[j], 0)));
    const damp = 0.01 * H.reduce((a, row, i) => a + row[i], 0) / D;
    H.forEach((row, i) => { row[i] += damp; });
    const Hinv0 = matInv(H);
    const outErr = (w) => {
      let num = 0, den = 0;
      X.forEach((x) => {
        let y = 0, yh = 0;
        for (let i = 0; i < D; i++) { y += w0[i] * x[i]; yh += w[i] * x[i]; }
        num += (yh - y) ** 2; den += y * y;
      });
      return num / den;
    };
    function state(k) {
      const w = w0.slice();
      let Hi = Hinv0.map((row) => row.slice());
      for (let i = 0; i < k; i++) {
        const q = quant(w[i]);
        const err = (w[i] - q) / Hi[i][i];
        for (let j = i + 1; j < D; j++) w[j] -= err * Hi[i][j];
        w[i] = q;
        const next = Hi.map((row) => row.slice());
        for (let a = i + 1; a < D; a++) for (let b = i + 1; b < D; b++) next[a][b] = Hi[a][b] - (Hi[a][i] * Hi[i][b]) / Hi[i][i];
        Hi = next;
      }
      return w;
    }
    const rtnW = w0.map(quant);
    const gptqW = state(D);
    let k = 0;

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">GPTQ：逐列量化，补偿后面的列</h4>
          <div class="lw-btn-row" style="margin:0">
            <button type="button" class="lw-btn" data-reset>重来</button>
            <button type="button" class="lw-btn primary" data-next>量化下一列</button>
          </div>
        </div>
        <p class="lw-subtitle">一行 16 个权重，3 位对称量化（8 个等级，水平虚线）。输入的第 1、2 个通道高度相关，第 3、4 个高度相关，以此类推。灰色框是原始值。</p>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      const w = state(k);
      const Wd = 640, Hd = 220, L = 30, top = 12, bot = 186;
      const vmax = scale * (QMAX + 1.2);
      const Y = (v) => top + ((vmax - v) / (2 * vmax)) * (bot - top);
      const bw = (Wd - L - 8) / D;
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="GPTQ 逐列量化的过程">`;
      for (let q = -QMAX - 1; q <= QMAX; q++) {
        svg += `<line x1="${L}" x2="${Wd - 8}" y1="${Y(q * scale)}" y2="${Y(q * scale)}" stroke="${W.color('rule-strong')}" stroke-dasharray="3 3"/>`;
      }
      svg += `<line x1="${L}" x2="${Wd - 8}" y1="${Y(0)}" y2="${Y(0)}" stroke="${W.color('ink-3')}"/>`;
      for (let i = 0; i < D; i++) {
        const x = L + i * bw + 4;
        const y0 = Y(Math.max(0, w0[i])), h0 = Math.abs(Y(w0[i]) - Y(0));
        svg += `<rect x="${x}" y="${y0}" width="${bw - 8}" height="${h0}" fill="none" stroke="${W.color('ink-3')}" stroke-width="1"/>`;
        const v = w[i];
        const y1 = Y(Math.max(0, v)), h1 = Math.abs(Y(v) - Y(0));
        const c = i < k ? W.color('prefill') : (k > 0 && i >= k ? W.color('decode') : W.color('rule-strong'));
        svg += `<rect x="${x + 4}" y="${y1}" width="${bw - 16}" height="${Math.max(1, h1)}" fill="${c}" opacity="${i < k ? 0.95 : 0.75}"><title>第 ${i + 1} 列：原始 ${w0[i].toFixed(3)}，当前 ${v.toFixed(3)}</title></rect>`;
        svg += `<text x="${x + (bw - 8) / 2}" y="${Hd - 18}" font-size="10" fill="${W.color('ink-3')}" text-anchor="middle">${i + 1}</text>`;
      }
      if (k < D) svg += `<text x="${L + k * bw + bw / 2}" y="${Hd - 4}" font-size="11" fill="${W.color('ink')}" text-anchor="middle">下一列</text>`;
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const curErr = outErr(w);
      $('[data-out]').innerHTML = `
        <div class="lw-legend">
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('prefill')}"></i>已量化</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('decode')}"></i>还没量化，已被调整</span>
          <span class="lw-legend-item"><i class="lw-dot" style="border:1px solid ${W.color('ink-3')}"></i>原始值</span>
        </div>
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">已量化</div><div class="lw-stat-value">${k} / ${D} 列</div></div>
          <div class="lw-stat"><div class="lw-stat-label">当前状态的输出误差</div><div class="lw-stat-value">${curErr.toExponential(2)}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">RTN 的输出误差</div><div class="lw-stat-value">${outErr(rtnW).toExponential(2)}</div></div>
          <div class="lw-stat ok"><div class="lw-stat-label">GPTQ 完成后</div><div class="lw-stat-value">${outErr(gptqW).toExponential(2)}</div></div>
        </div>
        <div class="lw-callout">${k === 0 ? '点击“量化下一列”。每量化一列，它的误差会按 H⁻¹ 分摊到后面的列上，和它相关性强的列（通常是紧挨着的那一列）调整得最多。'
          : k < D ? `第 ${k} 列被量化到最近的等级，后面的列随之调整。注意有些权重最终被量化到了不是离原始值最近的那个等级上，这正是误差补偿的结果。`
          : '全部量化完成。和 RTN 相比，每个权重离自己的原始值可能更远，但这一行的输出误差小得多。'}</div>`;
      $('[data-next]').disabled = k === D;
    }

    $('[data-next]').addEventListener('click', () => { if (k < D) { k++; update(); } });
    $('[data-reset]').addEventListener('click', () => { k = 0; update(); });
    update();
  });

  // ------------------------------------------------------------------ 2.3 AWQ
  W.register('awq-scale', function (container) {
    const DI = 64, DO = 32, N = 128, G = 32, SAL = [5, 22, 41, 58];
    const r = rng(7);
    const Wt = Array.from({ length: DO }, () => Array.from({ length: DI }, () => gauss(r) * 0.05));
    const Xb = Array.from({ length: N }, () => Array.from({ length: DI }, () => gauss(r)));
    const st = { mult: 10, bits: 4 };

    function run(mult, bits) {
      const qmax = 2 ** (bits - 1) - 1;
      const X = Xb.map((x) => x.map((v, j) => (SAL.includes(j) ? v * mult : v)));
      const mean = Array.from({ length: DI }, (_, j) => X.reduce((a, x) => a + Math.abs(x[j]), 0) / N);
      const Y = X.map((x) => Wt.map((w) => w.reduce((a, v, j) => a + v * x[j], 0)));
      const den = Y.reduce((a, y) => a + y.reduce((b, v) => b + v * v, 0), 0);
      const err = (Wq) => {
        let num = 0;
        X.forEach((x, n) => Wq.forEach((w, o) => {
          let y = 0;
          for (let j = 0; j < DI; j++) y += w[j] * x[j];
          num += (y - Y[n][o]) ** 2;
        }));
        return num / den;
      };
      const quant = (s, keep) => Wt.map((w) => {
        const ws = w.map((v, j) => v * s[j]);
        const out = ws.slice();
        for (let g = 0; g < DI; g += G) {
          let am = 0;
          for (let j = g; j < g + G; j++) if (!(keep && SAL.includes(j))) am = Math.max(am, Math.abs(ws[j]));
          const sc = am / qmax || 1e-8;
          for (let j = g; j < g + G; j++) {
            if (keep && SAL.includes(j)) continue;
            out[j] = Math.max(-qmax - 1, Math.min(qmax, Math.round(ws[j] / sc))) * sc;
          }
        }
        return out.map((v, j) => v / s[j]);
      });
      const curve = [];
      for (let i = 0; i <= 20; i++) {
        const a = i / 20;
        let s = mean.map((m) => m ** a);
        const norm = Math.sqrt(Math.max(...s) * Math.min(...s));
        s = s.map((v) => v / norm);
        curve.push(err(quant(s, false)));
      }
      return { curve, keep: err(quant(new Array(DI).fill(1), true)) };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">AWQ：搜索缩放的强度 α</h4></div>
        <p class="lw-subtitle">一个 64 维输入、32 维输出的线性层，每组 32 个权重。输入的 64 个通道中有 4 个（约 6%）幅度更大，是显著通道。曲线上每个点是用对应的 α 缩放后再量化得到的输出误差。</p>
        <div class="lw-tabs" data-bits>
          <button type="button" class="lw-tab" data-b="4">4 位</button>
          <button type="button" class="lw-tab" data-b="3">3 位</button>
        </div>
        <div class="lw-controls" style="grid-template-columns:1fr">
          <div class="lw-control">
            <label><span>显著通道的幅度</span><span class="lw-val" data-mv></span></label>
            <input type="range" min="1" max="30" step="1" value="${st.mult}" data-mult />
          </div>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      container.querySelectorAll('[data-b]').forEach((b) => b.classList.toggle('active', +b.dataset.b === st.bits));
      $('[data-mv]').textContent = `${st.mult} 倍`;
      const res = run(st.mult, st.bits);
      const best = Math.min(...res.curve);
      const bi = res.curve.indexOf(best);
      const Wd = 640, Hd = 220, L = 58, R = 16, T = 12, B = 180;
      const ymax = Math.min(Math.max(...res.curve), res.curve[0] * 2) * 1.08;
      const X = (a) => L + a * (Wd - L - R);
      const Yp = (v) => B - (Math.min(v, ymax) / ymax) * (B - T);
      const ink3 = W.color('ink-3');
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="输出误差随 α 的变化">`;
      [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
        const v = ymax * f;
        svg += `<line x1="${L}" x2="${Wd - R}" y1="${Yp(v)}" y2="${Yp(v)}" stroke="${W.color('rule')}"/>`;
        svg += `<text x="${L - 6}" y="${Yp(v) + 4}" font-size="10" fill="${ink3}" text-anchor="end">${f ? v.toExponential(1) : '0'}</text>`;
      });
      for (let i = 0; i <= 4; i++) svg += `<text x="${X(i / 4)}" y="${B + 16}" font-size="11" fill="${ink3}" text-anchor="middle">${(i / 4).toFixed(2)}</text>`;
      svg += `<text x="${(L + Wd - R) / 2}" y="${B + 34}" font-size="11" fill="${ink3}" text-anchor="middle">α（0 即 RTN）</text>`;
      if (res.keep < ymax) {
        svg += `<line x1="${L}" x2="${Wd - R}" y1="${Yp(res.keep)}" y2="${Yp(res.keep)}" stroke="${W.color('ok')}" stroke-dasharray="5 4" stroke-width="1.5"/>`;
        svg += `<text x="${Wd - R}" y="${Yp(res.keep) - 5}" font-size="11" fill="${W.color('ok')}" text-anchor="end">显著通道保留 FP16</text>`;
      }
      const pts = res.curve.map((v, i) => `${X(i / 20)},${Yp(v)}`).join(' ');
      svg += `<polyline points="${pts}" fill="none" stroke="${W.color('prefill')}" stroke-width="2"/>`;
      res.curve.forEach((v, i) => { svg += `<circle cx="${X(i / 20)}" cy="${Yp(v)}" r="${i === bi ? 5 : 2.5}" fill="${i === bi ? W.color('decode') : W.color('prefill')}"><title>α = ${(i / 20).toFixed(2)}，误差 ${v.toExponential(2)}</title></circle>`; });
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const gain = (res.curve[0] - best) / Math.max(1e-12, res.curve[0] - res.keep);
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">RTN（α = 0）</div><div class="lw-stat-value">${res.curve[0].toExponential(2)}</div><div class="lw-stat-sub">输出的相对均方误差</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">AWQ，α = ${(bi / 20).toFixed(2)}</div><div class="lw-stat-value">${best.toExponential(2)}</div><div class="lw-stat-sub">比 RTN 低 ${((1 - best / res.curve[0]) * 100).toFixed(0)}%</div></div>
          <div class="lw-stat ok"><div class="lw-stat-label">显著通道保留 FP16</div><div class="lw-stat-value">${res.keep.toExponential(2)}</div><div class="lw-stat-sub">需要混合精度</div></div>
        </div>
        <div class="lw-callout">${st.mult <= 2
          ? '显著通道和其他通道差不多大，缩放没有什么可以保护的，曲线几乎是平的。'
          : `AWQ 拿到了"保留 FP16"这种做法约 ${Math.max(0, Math.min(100, gain * 100)).toFixed(0)}% 的收益，而模型仍然是统一的 ${st.bits} 位。α 过大时，被放大的权重成为组内最大值，其他权重的量化间隔变大，误差又回升。`}</div>`;
    }

    container.querySelectorAll('[data-b]').forEach((b) => b.addEventListener('click', () => { st.bits = +b.dataset.b; update(); }));
    $('[data-mult]').addEventListener('input', (e) => { st.mult = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 2.4 SmoothQuant
  W.register('smooth-migrate', function (container) {
    const N = 64, D = 32, O = 32, OUT = [3, 19];
    const r = rng(5);
    const X = Array.from({ length: N }, () => Array.from({ length: D }, (_, j) => gauss(r) * (OUT.includes(j) ? 40 : 1) * (0.6 + 0.8 * r())));
    const Wm = Array.from({ length: O }, () => Array.from({ length: D }, () => gauss(r) * 0.05));
    const Y = X.map((x) => Wm.map((w) => w.reduce((a, v, j) => a + v * x[j], 0)));
    const den = Y.flat().reduce((a, v) => a + v * v, 0);
    const xm = Array.from({ length: D }, (_, j) => Math.max(...X.map((x) => Math.abs(x[j]))));
    const wm = Array.from({ length: D }, (_, j) => Math.max(...Wm.map((w) => Math.abs(w[j]))));
    const q8 = (v, sc) => Math.max(-127, Math.min(127, Math.round(v / sc))) * sc;
    const st = { alpha: 0.5, gran: 'tensor' };

    function run(a, gran) {
      const s = xm.map((m, j) => m ** a / wm[j] ** (1 - a));
      const Xs = X.map((x) => x.map((v, j) => v / s[j]));
      const Ws = Wm.map((w) => w.map((v, j) => v * s[j]));
      const tsc = Math.max(...Xs.flat().map(Math.abs)) / 127;
      const Xq = Xs.map((x) => { const sc = gran === 'tensor' ? tsc : Math.max(...x.map(Math.abs)) / 127; return x.map((v) => q8(v, sc)); });
      const Wq = Ws.map((w) => { const sc = Math.max(...w.map(Math.abs)) / 127; return w.map((v) => q8(v, sc)); });
      let e = 0, ea = 0, ew = 0;
      Xq.forEach((x, n) => Wq.forEach((w, o) => {
        let y = 0, ya = 0, yw = 0;
        for (let j = 0; j < D; j++) { y += w[j] * x[j]; ya += Ws[o][j] * x[j]; yw += w[j] * Xs[n][j]; }
        e += (y - Y[n][o]) ** 2; ea += (ya - Y[n][o]) ** 2; ew += (yw - Y[n][o]) ** 2;
      }));
      return {
        s, err: e / den, act: ea / den, w: ew / den,
        xmax: Array.from({ length: D }, (_, j) => Math.max(...Xs.map((x) => Math.abs(x[j])))),
        wmax: Array.from({ length: D }, (_, j) => Math.max(...Ws.map((w) => Math.abs(w[j])))),
      };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">SmoothQuant：把难度从激活挪到权重</h4></div>
        <p class="lw-subtitle">一个 32 维输入的线性层，64 个 token。输入的第 4、20 个通道是离群通道，幅度约为其他通道的 40 倍。权重按输出通道、激活按所选粒度做 INT8 量化。</p>
        <div class="lw-tabs" data-grans>
          <button type="button" class="lw-tab" data-g="tensor">激活按张量</button>
          <button type="button" class="lw-tab" data-g="token">激活按 token</button>
        </div>
        <div class="lw-controls" style="grid-template-columns:1fr">
          <div class="lw-control">
            <label><span>迁移强度 α</span><span class="lw-val" data-av></span></label>
            <input type="range" min="0" max="1" step="0.05" value="${st.alpha}" data-alpha />
          </div>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (sel) => container.querySelector(sel);

    function bars(vals, x0, w, top, h, color, label, ink3) {
      const vmax = Math.max(...vals);
      const bw = w / vals.length;
      let g = `<text x="${x0}" y="${top - 6}" font-size="11" fill="${ink3}">${label}</text>`;
      vals.forEach((v, j) => {
        const bh = Math.max(1, (v / vmax) * h);
        const hot = OUT.includes(j);
        g += `<rect x="${x0 + j * bw + 1}" y="${top + h - bh}" width="${bw - 2}" height="${bh}" fill="${hot ? W.color('decode') : color}" opacity="0.85"><title>通道 ${j + 1}：${v.toFixed(3)}</title></rect>`;
      });
      g += `<line x1="${x0}" x2="${x0 + w}" y1="${top + h}" y2="${top + h}" stroke="${W.color('rule-strong')}"/>`;
      const ratio = vmax / (vals.slice().sort((a, b) => a - b)[Math.floor(vals.length / 2)] || 1e-9);
      g += `<text x="${x0}" y="${top + h + 15}" font-size="11" fill="${ink3}">最大值是中位数的 ${ratio.toFixed(1)} 倍</text>`;
      return g;
    }

    const curves = {};
    function curve(gran) {
      if (!curves[gran]) curves[gran] = Array.from({ length: 21 }, (_, i) => run(i / 20, gran).err);
      return curves[gran];
    }

    function update() {
      container.querySelectorAll('[data-g]').forEach((b) => b.classList.toggle('active', b.dataset.g === st.gran));
      $('[data-av]').textContent = st.alpha.toFixed(2);
      const res = run(st.alpha, st.gran);
      const base = run(0, st.gran);
      const ink3 = W.color('ink-3');
      const Wd = 640, Hd = 290;
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="平滑后激活和权重每个输入通道的最大值，以及误差随 α 的变化">`;
      svg += bars(res.xmax, 16, 290, 24, 90, W.color('prefill'), '激活：每个输入通道的最大绝对值', ink3);
      svg += bars(res.wmax, 334, 290, 24, 90, W.color('ok'), '权重：每个输入通道的最大绝对值', ink3);
      // 误差曲线
      const cv = curve(st.gran);
      const L = 58, R = 16, T = 160, B = 256;
      const ymax = Math.max(...cv) * 1.08;
      const Xc = (a) => L + a * (Wd - L - R);
      const Yc = (v) => B - (v / ymax) * (B - T);
      [0, 0.5, 1].forEach((f) => {
        svg += `<line x1="${L}" x2="${Wd - R}" y1="${Yc(ymax * f)}" y2="${Yc(ymax * f)}" stroke="${W.color('rule')}"/>`;
        svg += `<text x="${L - 6}" y="${Yc(ymax * f) + 4}" font-size="10" fill="${ink3}" text-anchor="end">${f ? (ymax * f).toExponential(1) : '0'}</text>`;
      });
      svg += `<polyline points="${cv.map((v, i) => `${Xc(i / 20)},${Yc(v)}`).join(' ')}" fill="none" stroke="${W.color('ink-2')}" stroke-width="1.5"/>`;
      svg += `<circle cx="${Xc(st.alpha)}" cy="${Yc(res.err)}" r="5" fill="${W.color('decode')}"/>`;
      for (let i = 0; i <= 4; i++) svg += `<text x="${Xc(i / 4)}" y="${B + 15}" font-size="10" fill="${ink3}" text-anchor="middle">${(i / 4).toFixed(2)}</text>`;
      svg += `<text x="${(L + Wd - R) / 2}" y="${B + 31}" font-size="11" fill="${ink3}" text-anchor="middle">α 与输出误差</text>`;
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const best = cv.indexOf(Math.min(...cv)) / 20;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">只量化激活的误差</div><div class="lw-stat-value">${res.act.toExponential(2)}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">只量化权重的误差</div><div class="lw-stat-value">${res.w.toExponential(2)}</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">W8A8 的输出误差</div><div class="lw-stat-value">${res.err.toExponential(2)}</div><div class="lw-stat-sub">不平滑时 ${base.err.toExponential(2)}</div></div>
        </div>
        <div class="lw-callout">${st.alpha === 0
          ? '不平滑时，两个离群通道决定了激活的 scale，误差几乎全部来自激活；权重很容易量化。'
          : st.alpha >= 0.85 ? '迁移过头了：激活变得平坦，但离群通道的难度转到了权重上，权重的误差成为主要来源。'
          : `激活的离群通道被压平，权重对应的通道被放大。这个例子中 α 取 ${best.toFixed(2)} 左右时输出误差最小。SmoothQuant 论文对大多数模型取 0.5。`}${st.gran === 'token' ? '按 token 量化本身已经减小了激活误差，但离群通道出现在每个 token 里，仍然需要平滑。' : ''}</div>`;
    }

    container.querySelectorAll('[data-g]').forEach((b) => b.addEventListener('click', () => { st.gran = b.dataset.g; update(); }));
    $('[data-alpha]').addEventListener('input', (e) => { st.alpha = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 2.5 KV Cache 量化
  W.register('kv-quant-axis', function (container) {
    const T = 48, C = 32, HOT = [5, 6, 21];
    const r = rng(23);
    const tokScale = Array.from({ length: T }, () => 0.4 + 1.6 * r());
    const K = Array.from({ length: T }, () => Array.from({ length: C }, (_, c) => gauss(r) * 0.6 + (HOT.includes(c) ? (c === 21 ? -6 : 7) * (0.85 + 0.3 * r()) : 0)));
    const V = Array.from({ length: T }, (_, t) => Array.from({ length: C }, () => gauss(r) * tokScale[t]));
    const st = { bits: 4, axis: 'mixed' };

    // 非对称量化：每组用自己的 min 和 max
    function quant(M, axis, bits) {
      const L = 2 ** bits - 1;
      const out = M.map((row) => row.slice());
      const groups = axis === 'token'
        ? M.map((row, t) => row.map((_, c) => [t, c]))
        : Array.from({ length: C }, (_, c) => M.map((_, t) => [t, c]));
      groups.forEach((g) => {
        const vals = g.map(([t, c]) => M[t][c]);
        const lo = Math.min(...vals), hi = Math.max(...vals);
        const sc = (hi - lo) / L || 1e-8;
        g.forEach(([t, c]) => { out[t][c] = Math.round((M[t][c] - lo) / sc) * sc + lo; });
      });
      return out;
    }
    const relErr = (A, B) => {
      let n = 0, d = 0;
      A.forEach((row, t) => row.forEach((v, c) => { n += (v - B[t][c]) ** 2; d += v * v; }));
      return n / d;
    };

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">K 和 V 应该沿哪个方向量化</h4></div>
        <p class="lw-subtitle">模拟一个注意力头的 48 个 token，每个 32 维。K 的第 6、7、22 个通道在所有 token 上都很大；V 没有固定的大通道，但不同 token 的幅度相差较大。颜色越深，绝对值越大。</p>
        <div class="lw-tabs" data-bitsel>
          <button type="button" class="lw-tab" data-b="8">8 位</button>
          <button type="button" class="lw-tab" data-b="4">4 位</button>
          <button type="button" class="lw-tab" data-b="2">2 位</button>
        </div>
        <div class="lw-tabs" data-axes>
          <button type="button" class="lw-tab" data-a="token">K、V 都按 token</button>
          <button type="button" class="lw-tab" data-a="channel">K、V 都按通道</button>
          <button type="button" class="lw-tab" data-a="mixed">K 按通道，V 按 token</button>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (sel) => container.querySelector(sel);

    function heat(M, x0, y0, w, h, vmax, color, label, ink3) {
      const cw = w / C, ch = h / T;
      let g = `<text x="${x0}" y="${y0 - 6}" font-size="11" fill="${ink3}">${label}</text>`;
      M.forEach((row, t) => row.forEach((v, c) => {
        const a = Math.min(1, Math.abs(v) / vmax);
        if (a > 0.02) g += `<rect x="${(x0 + c * cw).toFixed(1)}" y="${(y0 + t * ch).toFixed(1)}" width="${(cw + 0.3).toFixed(1)}" height="${(ch + 0.3).toFixed(1)}" fill="${color}" opacity="${a.toFixed(2)}"/>`;
      }));
      g += `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="none" stroke="${W.color('rule-strong')}"/>`;
      return g;
    }

    function update() {
      container.querySelectorAll('[data-b]').forEach((b) => b.classList.toggle('active', +b.dataset.b === st.bits));
      container.querySelectorAll('[data-a]').forEach((b) => b.classList.toggle('active', b.dataset.a === st.axis));
      const kAxis = st.axis === 'token' ? 'token' : 'channel';
      const vAxis = st.axis === 'channel' ? 'channel' : 'token';
      const Kq = quant(K, kAxis, st.bits), Vq = quant(V, vAxis, st.bits);
      const kE = K.map((row, t) => row.map((v, c) => v - Kq[t][c]));
      const vE = V.map((row, t) => row.map((v, c) => v - Vq[t][c]));
      const ink3 = W.color('ink-3');
      const Wd = 640, Hd = 330, w = 140, h = 132;
      const emax = Math.max(...kE.flat().map(Math.abs), ...vE.flat().map(Math.abs), 1e-6);
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="K、V 的数值和量化误差热力图">`;
      svg += heat(K, 16, 24, w, h, 7, W.color('prefill'), 'K', ink3);
      svg += heat(kE, 176, 24, w, h, emax, W.color('decode'), `K 的误差（按${kAxis === 'token' ? ' token' : '通道'}）`, ink3);
      svg += heat(V, 336, 24, w, h, 3, W.color('ok'), 'V', ink3);
      svg += heat(vE, 496, 24, w, h, emax, W.color('decode'), `V 的误差（按${vAxis === 'token' ? ' token' : '通道'}）`, ink3);
      svg += `<text x="16" y="${24 + h + 16}" font-size="10" fill="${ink3}">横向：32 个通道；纵向：48 个 token。两张误差图用同一个色阶。</text>`;
      // 显存
      const perTok = 2 * 36 * 8 * 128 / 1024;  // KiB，乘以每个数的字节数
      const rows = [['BF16', 2], ['8 位', 1], ['4 位', 0.5], ['2 位', 0.25]];
      const y0 = 214;
      svg += `<text x="16" y="${y0}" font-size="11" fill="${ink3}">Qwen3-8B 每个 token 的 KV（不计 scale），以及 40 GiB 能存的 token 数</text>`;
      rows.forEach(([name, bytes], i) => {
        const kib = perTok * bytes;
        const cap = 40 * 1024 * 1024 / kib;
        const bw = (bytes / 2) * 420;
        const yy = y0 + 12 + i * 24;
        const on = (st.bits === 8 && i === 1) || (st.bits === 4 && i === 2) || (st.bits === 2 && i === 3);
        svg += `<text x="16" y="${yy + 13}" font-size="11" fill="${W.color('ink')}">${name}</text>`;
        svg += `<rect x="64" y="${yy}" width="${bw}" height="16" fill="${on ? W.color('decode') : W.color('rule-strong')}" opacity="${on ? 0.9 : 0.6}"/>`;
        svg += `<text x="${64 + bw + 8}" y="${yy + 13}" font-size="11" fill="${ink3}">${kib.toFixed(0)} KiB，约 ${(cap / 1e4).toFixed(1)} 万个 token</text>`;
      });
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const errs = {
        kt: relErr(K, quant(K, 'token', st.bits)), kc: relErr(K, quant(K, 'channel', st.bits)),
        vt: relErr(V, quant(V, 'token', st.bits)), vc: relErr(V, quant(V, 'channel', st.bits)),
      };
      const tile = (key, label, on) => `<div class="lw-stat ${on ? 'info' : ''}"><div class="lw-stat-label">${label}</div><div class="lw-stat-value">${errs[key].toExponential(2)}</div></div>`;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          ${tile('kt', 'K 按 token', kAxis === 'token')}${tile('kc', 'K 按通道', kAxis === 'channel')}
          ${tile('vt', 'V 按 token', vAxis === 'token')}${tile('vc', 'V 按通道', vAxis === 'channel')}
        </div>
        <div class="lw-callout">${st.axis === 'mixed'
          ? 'K 的大通道各自有一个 scale，不再挤占其他通道的精度；V 的每个 token 各自有一个 scale，幅度小的 token 也能保持精度。这就是 KIVI 的做法。'
          : st.axis === 'token' ? 'K 按 token 量化时，每个 token 的 scale 都被那三个大通道决定，其他通道的误差明显偏大（误差图中颜色均匀铺开）。'
          : 'V 按通道量化时，同一个通道的 scale 由幅度最大的 token 决定，幅度小的 token 相对误差很大。'}${st.bits === 8 ? '8 位时各种方式的误差都很小，粒度的影响要到 4 位以下才明显。' : ''}</div>`;
    }

    container.querySelectorAll('[data-b]').forEach((b) => b.addEventListener('click', () => { st.bits = +b.dataset.b; update(); }));
    container.querySelectorAll('[data-a]').forEach((b) => b.addEventListener('click', () => { st.axis = b.dataset.a; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 2.6 低比特矩阵乘法
  W.register('lowbit-gemm', function (container) {
    const GP = W.GPUS || {};
    const P = 6.95e9;                       // Qwen3-8B 全部线性层的参数量（不含词表）
    const EFF_BW = 0.8, EFF_C = 0.7, DEQ = 0.85;
    const st = { gpu: GP.h100 ? 'h100' : Object.keys(GP)[0], logb: 4 };
    const schemes = [
      { key: 'bf16', name: 'BF16', color: 'ink-3' },
      { key: 'w4a16', name: 'W4A16', color: 'prefill' },
      { key: 'w8a8', name: 'FP8 W8A8', color: 'decode' },
    ];
    function time(g, key, b) {
      const bw = g.bw * 1e12 * EFF_BW;
      const bf = g.bf16 * 1e12 * EFF_C;
      const flops = 2 * P * b;
      if (key === 'bf16') return Math.max((P * 2) / bw, flops / bf);
      if (key === 'w4a16') return Math.max((P * 0.53) / bw, flops / (bf * DEQ));
      // FP8：支持 FP8 的卡用 FP8 Tensor Core；否则退化为只量化权重的 W8A16
      if (g.fp8) return Math.max(P / bw, flops / (g.fp8 * 1e12 * EFF_C));
      return Math.max(P / bw, flops / (bf * 0.95));
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">低比特矩阵乘法的收益随 batch 怎样变化</h4></div>
        <p class="lw-subtitle">Qwen3-8B 全部线性层（约 69.5 亿参数）做一次前向的估算耗时，batch 指这一步一共处理多少个 token。按 Roofline 估算：取读权重的时间和计算时间中较大的一个，带宽按峰值的 80%、算力按 70% 计；W4A16 的反量化让计算效率再打 85 折。这是示意模型，实测会有出入。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>GPU</label><select data-gpu>${Object.keys(GP).map((k) => `<option value="${k}" ${k === st.gpu ? 'selected' : ''}>${GP[k].name}</option>`).join('')}</select></div>
          <div class="lw-control">
            <label><span>batch（token 数）</span><span class="lw-val" data-bv></span></label>
            <input type="range" min="0" max="12" step="1" value="${st.logb}" data-b />
          </div>
        </div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (sel) => container.querySelector(sel);

    function update() {
      const g = GP[st.gpu];
      const b = 2 ** st.logb;
      $('[data-bv]').textContent = b;
      const Wd = 640, Hd = 230, L = 52, R = 16, T = 14, B = 186;
      const bs = Array.from({ length: 13 }, (_, i) => 2 ** i);
      // 纵轴：相对 BF16 的加速比
      const sp = schemes.map((sc) => bs.map((x) => time(g, 'bf16', x) / time(g, sc.key, x)));
      const ymax = Math.max(4.2, ...sp.flat()) * 1.05;
      const X = (i) => L + (i / 12) * (Wd - L - R);
      const Y = (v) => B - (v / ymax) * (B - T);
      const ink3 = W.color('ink-3');
      let svg = `<svg viewBox="0 0 ${Wd} ${Hd}" width="100%" role="img" aria-label="相对 BF16 的加速比随 batch 的变化">`;
      for (let v = 0; v <= ymax; v += 1) {
        svg += `<line x1="${L}" x2="${Wd - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${W.color('rule')}"/>`;
        svg += `<text x="${L - 6}" y="${Y(v) + 4}" font-size="10" fill="${ink3}" text-anchor="end">${v}×</text>`;
      }
      bs.forEach((x, i) => { if (i % 2 === 0) svg += `<text x="${X(i)}" y="${B + 15}" font-size="10" fill="${ink3}" text-anchor="middle">${x}</text>`; });
      svg += `<text x="${(L + Wd - R) / 2}" y="${B + 32}" font-size="11" fill="${ink3}" text-anchor="middle">batch（token 数，对数刻度）与相对 BF16 的加速比</text>`;
      svg += `<line x1="${X(st.logb)}" x2="${X(st.logb)}" y1="${T}" y2="${B}" stroke="${W.color('ink-3')}" stroke-dasharray="3 3"/>`;
      schemes.forEach((sc, k) => {
        svg += `<polyline points="${sp[k].map((v, i) => `${X(i)},${Y(v)}`).join(' ')}" fill="none" stroke="${W.color(sc.color)}" stroke-width="2"/>`;
        svg += `<circle cx="${X(st.logb)}" cy="${Y(sp[k][st.logb])}" r="4" fill="${W.color(sc.color)}"/>`;
      });
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const ridge = (g.bf16 * EFF_C) / (g.bw * EFF_BW);   // FLOPs per byte
      const crossBf16 = ridge;                              // BF16：每个权重 2 字节、每 token 2 FLOPs，batch ≈ ridge 时到达计算瓶颈
      $('[data-out]').innerHTML = `
        <div class="lw-legend">${schemes.map((sc) => `<span class="lw-legend-item"><i class="lw-dot" style="background:${W.color(sc.color)}"></i>${sc.name}${sc.key === 'w8a8' && !g.fp8 ? '（这张卡不支持 FP8 计算，按 W8A16 计）' : ''}</span>`).join('')}</div>
        <div class="lw-stats">
          ${schemes.map((sc) => `<div class="lw-stat ${sc.key === 'bf16' ? '' : 'info'}"><div class="lw-stat-label">${sc.name}，batch ${b}</div><div class="lw-stat-value">${(time(g, sc.key, b) * 1e3).toFixed(2)} ms</div><div class="lw-stat-sub">${sc.key === 'bf16' ? '基准' : `${(time(g, 'bf16', b) / time(g, sc.key, b)).toFixed(2)} 倍`}</div></div>`).join('')}
        </div>
        <div class="lw-callout">在这张卡上，BF16 的线性层大约在 batch ${Math.round(crossBf16)} 附近从带宽瓶颈转为计算瓶颈。${b < crossBf16 / 4
          ? 'batch 很小时，耗时几乎全是读权重，W4A16 读的字节最少，加速接近 4 倍。'
          : b < crossBf16 * 2 ? '在这个区间，W4A16 的优势随 batch 增大而迅速缩小。'
          : `已进入计算瓶颈。W4A16 的矩阵乘法仍是 BF16，还要反量化，反而比 BF16 慢；${g.fp8 ? 'FP8 W8A8 用上了 FP8 Tensor Core，保持约 2 倍的优势。' : '这张卡没有 FP8 Tensor Core，要提高计算吞吐只能用 INT8 W8A8。'}`}</div>`;
    }

    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    $('[data-b]').addEventListener('input', (e) => { st.logb = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 2.7 选择量化方案
  W.register('quant-chooser', function (container) {
    const GP = W.GPUS || {}, MD = W.MODELS || {};
    if (!W.modelParams) { container.innerHTML = '<div class="lw-card">需要先加载 basics.js。</div>'; return; }
    const st = { model: MD['qwen3-32b'] ? 'qwen3-32b' : Object.keys(MD)[0], gpu: GP.l40s ? 'l40s' : Object.keys(GP)[0], goal: 'latency', ctx: 8192 };
    const GiB = 1024 ** 3;

    function plans(m, g) {
      const p = W.modelParams(m);
      const body = p.attn + p.mlp;                     // 被量化的线性层
      const keep = (p.emb + p.head) * 2;               // 词表和输出头保持 BF16
      const kvTok = 2 * m.L * m.nkv * m.dh;            // 每个 token 的 KV 元素数
      const usable = g.mem * 1e9 * 0.9 - 2 * GiB;      // 留 10% 和约 2 GiB 给激活、CUDA Graph 等
      const rows = [
        { key: 'bf16', name: 'BF16', wb: 2, kvb: 2, compute: 'BF16' },
        { key: 'fp8', name: 'FP8 W8A8', wb: 1, kvb: 2, compute: g.fp8 ? 'FP8' : 'BF16（按 W8A16 运行）' },
        { key: 'fp8kv', name: 'FP8 W8A8 + FP8 KV', wb: 1, kvb: 1, compute: g.fp8 ? 'FP8' : 'BF16（按 W8A16 运行）' },
        { key: 'int8', name: 'INT8 W8A8（SmoothQuant）', wb: 1, kvb: 2, compute: 'INT8' },
        { key: 'w4', name: 'W4A16（GPTQ / AWQ）', wb: 4.13 / 8, kvb: 2, compute: 'BF16' },
        { key: 'w4kv', name: 'W4A16 + FP8 KV', wb: 4.13 / 8, kvb: 1, compute: 'BF16' },
      ];
      rows.forEach((r) => {
        r.weights = body * r.wb + keep;
        r.kvTokens = Math.max(0, (usable - r.weights) / (kvTok * r.kvb));
        r.reqs = Math.floor(r.kvTokens / st.ctx);
        r.fits = r.weights < usable && r.reqs >= 1;
      });
      return rows;
    }

    function recommend(rows, g) {
      const by = Object.fromEntries(rows.map((r) => [r.key, r]));
      if (!rows.some((r) => r.fits)) return { key: null, why: '即使 4 位量化，这张卡也放不下模型和至少一个请求的 KV Cache。需要多卡并行（第 4 部分），或者换一个更小的模型。' };
      if (st.goal === 'latency') {
        if (by.w4.fits) return { key: 'w4', why: '并发低时瓶颈是读权重的带宽，W4A16 读的字节最少，单个请求的速度最快。显存吃紧时再加 FP8 KV Cache。' };
        return { key: 'w4kv', why: 'W4A16 的权重最小，再用 FP8 KV Cache 腾出显存，才能放下这个上下文长度。' };
      }
      // 吞吐优先
      if (g.fp8) {
        if (by.fp8kv.fits && by.fp8kv.reqs >= 16) return { key: by.fp8.reqs >= 16 ? 'fp8' : 'fp8kv', why: `高并发时瓶颈转向计算，FP8 W8A8 用上了 FP8 Tensor Core，精度几乎无损。${by.fp8.reqs >= 16 ? '显存足够时先不量化 KV Cache；需要更多并发时再加上 FP8 KV Cache。' : '需要 FP8 KV Cache 才能放下足够多的并发请求。'}` };
        return { key: 'w4kv', why: 'FP8 的权重放下后剩给 KV Cache 的空间太少，并发上不去。先用 W4A16 加 FP8 KV Cache，或者考虑多卡。' };
      }
      if (by.int8.fits && by.int8.reqs >= 16) return { key: 'int8', why: '这张卡不支持 FP8 计算。要在高并发时提高计算吞吐，可以用 INT8 W8A8，需要 SmoothQuant 和校准数据，并检查精度。' };
      return { key: 'w4', why: '这张卡不支持 FP8 计算，8 位的权重又占用了太多显存。W4A16 能留出更多 KV Cache，用并发弥补计算上的不足。' };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">选择量化方案</h4></div>
        <p class="lw-subtitle">单张显卡，按显存的 90% 减去约 2 GiB 的运行时开销计算可用空间；词表和输出头保持 BF16。"并发请求数"指每个请求都用满所选上下文长度时，KV Cache 能同时容纳多少个。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label><select data-model>${Object.keys(MD).map((k) => `<option value="${k}" ${k === st.model ? 'selected' : ''}>${MD[k].name}</option>`).join('')}</select></div>
          <div class="lw-control"><label>GPU</label><select data-gpu>${Object.keys(GP).map((k) => `<option value="${k}" ${k === st.gpu ? 'selected' : ''}>${GP[k].name}</option>`).join('')}</select></div>
          <div class="lw-control"><label>每个请求的上下文</label><select data-ctx>${[2048, 4096, 8192, 32768].map((c) => `<option value="${c}" ${c === st.ctx ? 'selected' : ''}>${c / 1024}K token</option>`).join('')}</select></div>
        </div>
        <div class="lw-tabs" data-goals>
          <button type="button" class="lw-tab" data-goal="latency">延迟优先，并发低</button>
          <button type="button" class="lw-tab" data-goal="throughput">吞吐优先，并发高</button>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (sel) => container.querySelector(sel);

    function update() {
      container.querySelectorAll('[data-goal]').forEach((b) => b.classList.toggle('active', b.dataset.goal === st.goal));
      const m = MD[st.model], g = GP[st.gpu];
      const rows = plans(m, g);
      const rec = recommend(rows, g);
      $('[data-out]').innerHTML = `
        <div class="lw-table-wrap"><table class="lw-table">
          <thead><tr><th>方案</th><th>权重</th><th>矩阵乘法</th><th>KV Cache 可容纳</th><th>并发请求数</th></tr></thead>
          <tbody>${rows.map((r) => `<tr${r.key === rec.key ? ' class="lw-row-hi"' : ''}${r.fits ? '' : ' style="opacity:0.5"'}>
            <td>${r.name}${r.key === rec.key ? '（推荐）' : ''}</td>
            <td>${(r.weights / GiB).toFixed(1)} GiB</td>
            <td>${r.compute}</td>
            <td>${r.fits ? `${(r.kvTokens / 1e4).toFixed(1)} 万 token` : '放不下'}</td>
            <td>${r.fits ? r.reqs : '-'}</td></tr>`).join('')}</tbody>
        </table></div>
        <div class="lw-callout">${rec.why}</div>`;
    }

    $('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    $('[data-ctx]').addEventListener('change', (e) => { st.ctx = +e.target.value; update(); });
    container.querySelectorAll('[data-goal]').forEach((b) => b.addEventListener('click', () => { st.goal = b.dataset.goal; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 2.8 2:4 稀疏
  W.register('sparse-24', function (container) {
    const O = 8, D = 32, N = 64, HOT = [2, 9, 17, 26];
    const r = rng(31);
    const xs = Array.from({ length: D }, (_, j) => (HOT.includes(j) ? 6 : 0.5 + r()));
    const X = Array.from({ length: N }, () => Array.from({ length: D }, (_, j) => gauss(r) * xs[j]));
    const Wm = Array.from({ length: O }, () => Array.from({ length: D }, () => gauss(r)));
    const xn = Array.from({ length: D }, (_, j) => Math.sqrt(X.reduce((a, x) => a + x[j] * x[j], 0)));
    const rr = rng(77);
    const rnd = Wm.map((w) => w.map(() => rr()));
    const crit = {
      random: { name: '随机', score: (o, j) => rnd[o][j] },
      magnitude: { name: '按权重大小 |W|', score: (o, j) => Math.abs(Wm[o][j]) },
      wanda: { name: '按 |W| × ‖X‖（Wanda）', score: (o, j) => Math.abs(Wm[o][j]) * xn[j] },
    };
    const st = { crit: 'magnitude' };

    function prune(key) {
      return Wm.map((w, o) => {
        const out = w.slice();
        for (let g = 0; g < D; g += 4) {
          const idx = [0, 1, 2, 3].map((k) => g + k).sort((a, b) => crit[key].score(o, a) - crit[key].score(o, b));
          out[idx[0]] = 0; out[idx[1]] = 0;
        }
        return out;
      });
    }
    function err(Wp) {
      let n = 0, d = 0;
      X.forEach((x) => Wm.forEach((w, o) => {
        let y = 0, yp = 0;
        for (let j = 0; j < D; j++) { y += w[j] * x[j]; yp += Wp[o][j] * x[j]; }
        n += (y - yp) ** 2; d += y * y;
      }));
      return n / d;
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">2:4 稀疏：每 4 个权重保留 2 个</h4></div>
        <p class="lw-subtitle">8 行、32 列的权重，每行每连续 4 个数中剪掉 2 个。输入的第 3、10、18、27 个通道幅度较大（上方的三角形）。比较三种决定剪掉哪两个的准则。</p>
        <div class="lw-tabs" data-crits>${Object.entries(crit).map(([k, c]) => `<button type="button" class="lw-tab" data-c="${k}">${c.name}</button>`).join('')}</div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (sel) => container.querySelector(sel);

    function update() {
      container.querySelectorAll('[data-c]').forEach((b) => b.classList.toggle('active', b.dataset.c === st.crit));
      const Wp = prune(st.crit);
      const Wd = 640, L = 16, top = 30, cw = (Wd - 2 * L) / D, ch = 20;
      const vmax = Math.max(...Wm.flat().map(Math.abs));
      const ink3 = W.color('ink-3');
      let svg = `<svg viewBox="0 0 ${Wd} ${top + O * ch + 40}" width="100%" role="img" aria-label="2:4 稀疏后保留和剪掉的权重">`;
      HOT.forEach((j) => { svg += `<path d="M${L + j * cw + cw / 2 - 5},8 L${L + j * cw + cw / 2 + 5},8 L${L + j * cw + cw / 2},17 Z" fill="${W.color('decode')}"/>`; });
      for (let g = 0; g <= D; g += 4) svg += `<line x1="${L + g * cw}" x2="${L + g * cw}" y1="${top - 4}" y2="${top + O * ch + 4}" stroke="${W.color('rule-strong')}"/>`;
      Wm.forEach((w, o) => w.forEach((v, j) => {
        const kept = Wp[o][j] !== 0;
        const a = 0.25 + 0.75 * Math.abs(v) / vmax;
        svg += `<rect x="${L + j * cw + 1.5}" y="${top + o * ch + 1.5}" width="${cw - 3}" height="${ch - 3}" rx="2" fill="${kept ? W.color('prefill') : 'none'}" opacity="${kept ? a.toFixed(2) : 1}" stroke="${kept ? 'none' : W.color('rule-strong')}" stroke-dasharray="${kept ? '' : '2 2'}"><title>第 ${o + 1} 行第 ${j + 1} 列：${v.toFixed(2)}${kept ? '' : '，已剪掉'}</title></rect>`;
      }));
      svg += `<text x="${L}" y="${top + O * ch + 26}" font-size="11" fill="${ink3}">实心：保留，颜色越深绝对值越大；虚线框：剪掉。竖线分隔每 4 个一组。</text>`;
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      const errs = Object.keys(crit).map((k) => ({ k, e: err(prune(k)) }));
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          ${errs.map((x) => `<div class="lw-stat ${x.k === st.crit ? 'info' : ''}"><div class="lw-stat-label">${crit[x.k].name}</div><div class="lw-stat-value">${x.e.toExponential(2)}</div><div class="lw-stat-sub">输出的相对均方误差</div></div>`).join('')}
        </div>
        <div class="lw-callout">${st.crit === 'wanda'
          ? '把输入的幅度考虑进来以后，大通道上的权重即使本身不大也被保留，误差降到按权重大小剪枝的约 1/6。这和 AWQ 的观察一致：权重是否重要，要看它乘的激活有多大。'
          : st.crit === 'magnitude' ? '只看权重大小，大通道上那些绝对值不大的权重也会被剪掉，而它们对输出的贡献其实很大。'
          : '随机剪掉一半权重，误差最大。'}无论用哪个准则，一次剪掉一半都会带来明显的误差，实际中还需要用 SparseGPT 式的补偿，或者剪枝后继续训练来恢复精度。</div>
        <div class="lw-callout">存储：BF16 的 2:4 稀疏矩阵只存一半的数值，每个保留的数另用 2 位记录它在 4 个中的位置，每个原始元素平均 9 位，约为稠密矩阵的 56%。</div>`;
    }

    container.querySelectorAll('[data-c]').forEach((b) => b.addEventListener('click', () => { st.crit = b.dataset.c; update(); }));
    update();
  });
})();
