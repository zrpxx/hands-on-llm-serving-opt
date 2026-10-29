/**
 * 第 3 部分的交互组件
 * - attn-variants       3.1 不同注意力结构的 KV Cache 大小
 * - flash-tiling        3.2 FlashAttention 的分块与在线 softmax
 * - fusion-io           3.3 算子融合前后的显存读写与 kernel 数
 * - spec-speedup        3.4 投机解码的期望接受长度与加速比
 * - spec-batch          3.5 并发增大时投机解码的收益
 * - launch-timeline     3.6 CPU 发射与 GPU 执行的时间线，CUDA Graph 的作用
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
  const fmtBytes = (b) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(1)} GiB` : b >= 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(1)} MiB` : `${(b / 1024).toFixed(1)} KiB`);

  // ------------------------------------------------------------------ 3.1 注意力结构与 KV Cache
  // perLayer：每个 token 每层缓存的元素数；window：滑动窗口层的窗口长度；local：滑动窗口层的层数
  const ATTN = [
    { key: 'llama2', name: 'Llama-2-7B', kind: 'MHA', L: 32, perLayer: 2 * 32 * 128, note: '32 个查询头，32 个 KV 头' },
    { key: 'llama3', name: 'Llama-3.1-8B', kind: 'GQA', L: 32, perLayer: 2 * 8 * 128, note: '32 个查询头共用 8 个 KV 头' },
    { key: 'qwen3', name: 'Qwen3-8B', kind: 'GQA', L: 36, perLayer: 2 * 8 * 128, note: '32 个查询头共用 8 个 KV 头' },
    { key: 'l70', name: 'Llama-3.1-70B', kind: 'GQA', L: 80, perLayer: 2 * 8 * 128, note: '64 个查询头共用 8 个 KV 头' },
    { key: 'gemma3', name: 'Gemma-3-27B', kind: 'GQA + 滑动窗口', L: 62, perLayer: 2 * 16 * 128, window: 1024, local: 52, note: '每 6 层中 5 层只看最近 1024 个 token' },
    { key: 'gptoss', name: 'gpt-oss-120b', kind: 'GQA + 滑动窗口', L: 36, perLayer: 2 * 8 * 64, window: 128, local: 18, note: '一半的层只看最近 128 个 token' },
    { key: 'dsv3', name: 'DeepSeek-V3', kind: 'MLA', L: 61, perLayer: 512 + 64, note: '每层只存 512 维的隐向量和 64 维的 RoPE 部分' },
    { key: 'dsv3mha', name: '假如 V3 改用 MHA', kind: '对照', L: 61, perLayer: 128 * (192 + 128), note: '128 个头，K 192 维、V 128 维' },
  ];

  function kvBytes(m, T) {
    const local = m.local || 0;
    const full = m.L - local;
    const tokens = full * T + local * Math.min(T, m.window || T);
    return tokens * m.perLayer * 2;            // BF16
  }

  W.register('attn-variants', function (container) {
    const st = { logT: 15, pick: 'qwen3' };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">注意力结构与 KV Cache 的大小</h4></div>
        <p class="lw-subtitle">一个请求在给定上下文长度下的 KV Cache 总量，BF16。参数来自各模型的 config.json。点击一行查看说明。</p>
        <div class="lw-controls" style="grid-template-columns:1fr">
          <div class="lw-control">
            <label><span>上下文长度</span><span class="lw-val" data-tv></span></label>
            <input type="range" min="10" max="17" step="1" value="${st.logT}" data-t />
          </div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      const T = 2 ** st.logT;
      $('[data-tv]').textContent = `${T >= 1024 ? `${T / 1024}K` : T} token`;
      const rows = ATTN.map((m) => ({ m, b: kvBytes(m, T) }));
      const bmax = Math.max(...rows.map((r) => r.b));
      const Wd = 640, rowH = 26, L = 196, R = 110;
      let svg = `<svg viewBox="0 0 ${Wd} ${rows.length * rowH + 8}" width="100%" role="img" aria-label="各模型的 KV Cache 大小">`;
      rows.forEach((r, i) => {
        const y = i * rowH + 4;
        const w = Math.max(2, (r.b / bmax) * (Wd - L - R));
        const on = r.m.key === st.pick;
        const c = r.m.kind === 'MLA' ? W.color('ok') : r.m.kind === '对照' ? W.color('rule-strong') : r.m.kind === 'MHA' ? W.color('decode') : W.color('prefill');
        svg += `<g data-k="${r.m.key}" style="cursor:pointer">`;
        svg += `<rect x="0" y="${y - 2}" width="${Wd}" height="${rowH - 2}" fill="${on ? W.color('prefill-soft') : 'transparent'}"/>`;
        svg += `<text x="8" y="${y + 15}" font-size="12" fill="${W.color('ink')}">${r.m.name}</text>`;
        svg += `<text x="${L - 8}" y="${y + 15}" font-size="10" fill="${W.color('ink-3')}" text-anchor="end">${r.m.kind}</text>`;
        svg += `<rect x="${L}" y="${y + 3}" width="${w}" height="${rowH - 10}" fill="${c}" opacity="0.85"/>`;
        svg += `<text x="${L + w + 6}" y="${y + 15}" font-size="11" fill="${W.color('ink-2')}">${fmtBytes(r.b)}</text>`;
        svg += '</g>';
      });
      svg += '</svg>';
      const m = ATTN.find((x) => x.key === st.pick);
      const perTok = kvBytes(m, T) / T;
      const base = kvBytes(ATTN[0], T);
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat info"><div class="lw-stat-label">${m.name}</div><div class="lw-stat-value">${fmtBytes(kvBytes(m, T))}</div><div class="lw-stat-sub">一个请求，${T} token</div></div>
          <div class="lw-stat"><div class="lw-stat-label">平均每个 token</div><div class="lw-stat-value">${fmtBytes(perTok)}</div><div class="lw-stat-sub">${m.window ? '上下文越长，平均越小' : '与上下文长度无关'}</div></div>
          <div class="lw-stat"><div class="lw-stat-label">相对 Llama-2-7B</div><div class="lw-stat-value">${(kvBytes(m, T) / base).toFixed(2)} 倍</div></div>
        </div>
        <div class="lw-callout">${m.note}。${m.kind === 'MLA' ? 'DeepSeek-V3 有 6710 亿参数，每个 token 的 KV 却比 8B 的 GQA 模型还小。' : m.window ? `上下文超过 ${m.window} 之后，滑动窗口层的 KV 不再增长，只有全局注意力层随上下文线性增长。` : m.kind === '对照' ? '这是用 MHA 实现同样的头数和维度时的大小，比实际的 MLA 大约 70 倍。' : ''}</div>`;
      $('[data-out]').querySelectorAll('[data-k]').forEach((g) => g.addEventListener('click', () => { st.pick = g.dataset.k; update(); }));
    }
    $('[data-t]').addEventListener('input', (e) => { st.logT = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 3.2 FlashAttention 分块
  W.register('flash-tiling', function (container) {
    const M = 8, N = 16, BR = 4, BC = 4;
    const r = rng(9);
    const S = Array.from({ length: M }, () => Array.from({ length: N }, () => gauss(r) * 1.5));
    const st = { causal: false, step: 0, row: 1, logN: 13 };

    function tiles() {
      const out = [];
      for (let i = 0; i < M / BR; i++) {
        for (let j = 0; j < N / BC; j++) {
          // 因果时，查询 i 的绝对位置取 N - M + i，与 key 的位置比较
          if (st.causal && j * BC > N - M + (i + 1) * BR - 1) continue;
          out.push([i, j]);
        }
      }
      return out;
    }
    const visible = (q, k) => !st.causal || k <= N - M + q;

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">FlashAttention：一块一块地算</h4>
          <div class="lw-btn-row" style="margin:0">
            <button type="button" class="lw-btn" data-reset>重来</button>
            <button type="button" class="lw-btn primary" data-next>处理下一块</button>
          </div>
        </div>
        <p class="lw-subtitle">8 个查询、16 个 key 的注意力分数矩阵，按 4×4 分块。每个 program 负责一行块（4 个查询），从左到右遍历 key 块。点击矩阵的某一行，跟踪这个查询的 m 和 l。</p>
        <div class="lw-tabs">
          <button type="button" class="lw-tab" data-c="0">不加掩码</button>
          <button type="button" class="lw-tab" data-c="1">因果掩码</button>
        </div>
        <div data-plot></div>
        <div data-out></div>
        <div class="lw-controls" style="grid-template-columns:1fr;margin-top:10px">
          <div class="lw-control">
            <label><span>估算显存读写量：序列长度（单个头，head_dim 128，BF16，分块大小 128）</span><span class="lw-val" data-nv></span></label>
            <input type="range" min="10" max="17" step="1" value="${st.logN}" data-n />
          </div>
        </div>
        <div data-io></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function rowState(q, done) {
      let m = -Infinity, l = 0;
      const hist = [];
      done.forEach(([i, j]) => {
        if (i !== Math.floor(q / BR)) return;
        const vals = [];
        for (let k = j * BC; k < (j + 1) * BC; k++) if (visible(q, k)) vals.push(S[q][k]);
        if (!vals.length) return;
        const mNew = Math.max(m, ...vals);
        const scale = m === -Infinity ? 0 : Math.exp(m - mNew);
        l = l * scale + vals.reduce((a, v) => a + Math.exp(v - mNew), 0);
        hist.push({ j, m: mNew, l, scale });
        m = mNew;
      });
      return { m, l, hist };
    }

    function update() {
      container.querySelectorAll('[data-c]').forEach((b) => b.classList.toggle('active', +b.dataset.c === (st.causal ? 1 : 0)));
      const all = tiles();
      const done = all.slice(0, st.step);
      const cur = all[st.step - 1];
      const Wd = 640, cs = 26, x0 = 40, y0 = 20;
      let svg = `<svg viewBox="0 0 ${Wd} ${y0 + M * cs + 30}" width="100%" role="img" aria-label="分块的注意力分数矩阵">`;
      for (let q = 0; q < M; q++) {
        svg += `<text x="${x0 - 8}" y="${y0 + q * cs + 17}" font-size="10" fill="${W.color('ink-3')}" text-anchor="end">q${q}</text>`;
        for (let k = 0; k < N; k++) {
          const ti = Math.floor(q / BR), tj = Math.floor(k / BC);
          const isDone = done.some(([i, j]) => i === ti && j === tj);
          const isCur = cur && cur[0] === ti && cur[1] === tj;
          const vis = visible(q, k);
          const fill = !vis ? W.color('rule') : isCur ? W.color('decode') : isDone ? W.color('prefill') : 'none';
          const op = !vis ? 0.5 : isDone || isCur ? 0.25 + 0.6 * Math.min(1, Math.exp(S[q][k]) / 8) : 1;
          svg += `<rect data-q="${q}" x="${x0 + k * cs + 1}" y="${y0 + q * cs + 1}" width="${cs - 2}" height="${cs - 2}" fill="${fill}" opacity="${op.toFixed(2)}" stroke="${W.color('rule-strong')}" stroke-width="0.5" style="cursor:pointer"/>`;
        }
        if (q === st.row) svg += `<rect x="${x0 - 2}" y="${y0 + q * cs - 1}" width="${N * cs + 4}" height="${cs + 2}" fill="none" stroke="${W.color('ink')}" stroke-width="1.5"/>`;
      }
      for (let j = 0; j <= N; j += BC) svg += `<line x1="${x0 + j * cs}" x2="${x0 + j * cs}" y1="${y0}" y2="${y0 + M * cs}" stroke="${W.color('ink-3')}" stroke-width="1.2"/>`;
      for (let i = 0; i <= M; i += BR) svg += `<line x1="${x0}" x2="${x0 + N * cs}" y1="${y0 + i * cs}" y2="${y0 + i * cs}" stroke="${W.color('ink-3')}" stroke-width="1.2"/>`;
      for (let k = 0; k < N; k += 4) svg += `<text x="${x0 + k * cs + 2}" y="${y0 - 6}" font-size="10" fill="${W.color('ink-3')}">k${k}</text>`;
      svg += `<text x="${x0 + N * cs + 14}" y="${y0 + 14}" font-size="11" fill="${W.color('ink-2')}">已处理 ${done.length} / ${all.length} 块</text>`;
      if (st.causal) svg += `<text x="${x0 + N * cs + 14}" y="${y0 + 32}" font-size="11" fill="${W.color('ink-3')}">灰色：被掩码</text><text x="${x0 + N * cs + 14}" y="${y0 + 48}" font-size="11" fill="${W.color('ink-3')}">右上角整块跳过</text>`;
      svg += '</svg>';
      $('[data-plot]').innerHTML = svg;
      $('[data-plot]').querySelectorAll('[data-q]').forEach((el) => el.addEventListener('click', () => { st.row = +el.dataset.q; update(); }));

      const rs = rowState(st.row, done);
      const vals = S[st.row].filter((_, k) => visible(st.row, k));
      const trueM = Math.max(...vals);
      const trueL = vals.reduce((a, v) => a + Math.exp(v - trueM), 0);
      const finished = done.filter(([i]) => i === Math.floor(st.row / BR)).length === all.filter(([i]) => i === Math.floor(st.row / BR)).length;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">q${st.row} 当前的 m</div><div class="lw-stat-value">${rs.m === -Infinity ? '−∞' : rs.m.toFixed(3)}</div><div class="lw-stat-sub">目前为止的最大分数</div></div>
          <div class="lw-stat"><div class="lw-stat-label">q${st.row} 当前的 l</div><div class="lw-stat-value">${rs.l.toFixed(3)}</div><div class="lw-stat-sub">exp(s − m) 之和</div></div>
          <div class="lw-stat ${finished ? 'ok' : ''}"><div class="lw-stat-label">一次性计算的结果</div><div class="lw-stat-value">${trueM.toFixed(3)} / ${trueL.toFixed(3)}</div><div class="lw-stat-sub">m / l，处理完这一行后应相同</div></div>
        </div>
        <div class="lw-callout">${!rs.hist.length ? '还没有处理到这一行所在的块。' : rs.hist.map((h, i) => `第 ${i + 1} 块（k${h.j * BC} 到 k${h.j * BC + BC - 1}）：${i === 0 ? '初始化' : h.scale < 1 ? `出现了更大的分数，已累加的 l 和输出乘以 ${h.scale.toFixed(3)} 修正` : '最大值没有变化，修正系数为 1'}`).join('；')}。</div>`;
      $('[data-next]').disabled = st.step >= all.length;

      // 显存读写估算
      const Nn = 2 ** st.logN, d = 128, B = 128, e = 2;
      const naive = (4 * Nn * d + 8 * Nn * Nn) * e;            // 不融合：QKᵀ 写 S；缩放、掩码、softmax 各读写一遍；PV 读 P
      const flash = (2 * Nn * d + (Nn / B) * 2 * Nn * d) * e;   // 读 Q、写 O 各一次；K、V 被每个查询块读一遍
      $('[data-nv]').textContent = `${Nn >= 1024 ? `${Nn / 1024}K` : Nn}`;
      $('[data-io]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">分数矩阵 S 本身</div><div class="lw-stat-value">${fmtBytes(Nn * Nn * e)}</div><div class="lw-stat-sub">标准实现要把它放进显存</div></div>
          <div class="lw-stat"><div class="lw-stat-label">不融合的实现，读写量</div><div class="lw-stat-value">${fmtBytes(naive)}</div><div class="lw-stat-sub">S 与 P 被读写约 8 遍</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">FlashAttention，读写量</div><div class="lw-stat-value">${fmtBytes(flash)}</div><div class="lw-stat-sub">上限：K、V 每个查询块读一遍</div></div>
        </div>
        <div class="lw-callout">FlashAttention 的额外显存只有 O(N)，不随 N² 增长。它对 K、V 的重复读取有很大一部分会命中 L2 缓存，实际的显存流量比这里的上限还小。</div>        </div>`;
    }

    $('[data-next]').addEventListener('click', () => { st.step++; update(); });
    $('[data-reset]').addEventListener('click', () => { st.step = 0; update(); });
    container.querySelectorAll('[data-c]').forEach((b) => b.addEventListener('click', () => { st.causal = b.dataset.c === '1'; st.step = 0; update(); }));
    $('[data-n]').addEventListener('input', (e) => { st.logN = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 3.3 算子融合
  W.register('fusion-io', function (container) {
    const GP = W.GPUS || {};
    const D = 4096, I = 12288, LAUNCH = 4e-6;     // Qwen3-8B 的维度；每个 kernel 约 4 微秒的固定开销
    // reads / writes 以 N×D（或 N×I）个元素为单位
    const CHAINS = {
      norm: {
        name: '残差 + RMSNorm', dim: D,
        unfused: [['x + residual', 2, 1], ['h.pow(2)', 1, 1], ['.mean(-1)', 1, 0], ['torch.rsqrt', 0, 0], ['h * rsqrt', 1, 1], ['* weight', 1, 1]],
        fused: [['融合 kernel', 2, 2]],
      },
      swiglu: {
        name: 'SwiGLU 激活', dim: I,
        unfused: [['silu(gate)', 1, 1], ['silu * up', 2, 1]],
        fused: [['融合 kernel', 2, 1]],
      },
    };
    const st = { chain: 'norm', gpu: GP.h100 ? 'h100' : Object.keys(GP)[0], logN: 4 };

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">算子融合省下了什么</h4></div>
        <p class="lw-subtitle">按 Qwen3-8B 的维度（hidden 4096，FFN 中间层 12288，BF16）估算一层中的一个小环节。每个 kernel 的耗时取"读写字节数 ÷ 峰值带宽的 80%"，再加约 4 微秒的启动和调度开销。</p>
        <div class="lw-tabs">${Object.entries(CHAINS).map(([k, c]) => `<button type="button" class="lw-tab" data-ch="${k}">${c.name}</button>`).join('')}</div>
        <div class="lw-controls">
          <div class="lw-control"><label>GPU</label><select data-gpu>${Object.keys(GP).map((k) => `<option value="${k}" ${k === st.gpu ? 'selected' : ''}>${GP[k].name}</option>`).join('')}</select></div>
          <div class="lw-control">
            <label><span>这一步的 token 数</span><span class="lw-val" data-nv></span></label>
            <input type="range" min="0" max="14" step="1" value="${st.logN}" data-n />
          </div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function cost(list, N, dim, bw) {
      let bytes = 0, t = 0;
      const rows = list.map(([name, r, w]) => {
        const b = (r + w) * N * dim * 2;
        const tt = b / bw + LAUNCH;
        bytes += b; t += tt;
        return { name, b, t: tt };
      });
      return { rows, bytes, t };
    }

    function update() {
      container.querySelectorAll('[data-ch]').forEach((b) => b.classList.toggle('active', b.dataset.ch === st.chain));
      const N = 2 ** st.logN;
      $('[data-nv]').textContent = N;
      const c = CHAINS[st.chain], g = GP[st.gpu];
      const bw = g.bw * 1e12 * 0.8;
      const u = cost(c.unfused, N, c.dim, bw), f = cost(c.fused, N, c.dim, bw);
      const launchShare = (c.unfused.length * LAUNCH) / u.t;
      $('[data-out]').innerHTML = `
        <div class="lw-table-wrap"><table class="lw-table">
          <thead><tr><th>不融合时的 kernel</th><th>读写量</th><th>估算耗时</th></tr></thead>
          <tbody>${u.rows.map((r) => `<tr><td><code>${r.name}</code></td><td>${(r.b / 2 ** 20).toFixed(2)} MiB</td><td>${(r.t * 1e6).toFixed(1)} μs</td></tr>`).join('')}</tbody>
        </table></div>
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">不融合：${c.unfused.length} 个 kernel</div><div class="lw-stat-value">${(u.t * 1e6).toFixed(1)} μs</div><div class="lw-stat-sub">读写 ${(u.bytes / 2 ** 20).toFixed(2)} MiB</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">融合：1 个 kernel</div><div class="lw-stat-value">${(f.t * 1e6).toFixed(1)} μs</div><div class="lw-stat-sub">读写 ${(f.bytes / 2 ** 20).toFixed(2)} MiB</div></div>
          <div class="lw-stat"><div class="lw-stat-label">加速</div><div class="lw-stat-value">${(u.t / f.t).toFixed(1)} 倍</div></div>
        </div>
        <div class="lw-callout">${launchShare > 0.5
          ? `token 很少时（decode 阶段的典型情况），读写的数据量很小，不融合时约 ${(launchShare * 100).toFixed(0)}% 的时间是每个 kernel 的固定开销。融合减少了 kernel 的数量；3.6 节的 CUDA Graph 则进一步减少每个 kernel 的启动开销。`
          : `token 很多时，固定开销可以忽略，加速比接近读写量之比 ${(u.bytes / f.bytes).toFixed(1)} 倍。这类逐元素和归约的算子完全受限于带宽，减少一次读写就是减少一次时间。`}</div>`;
    }

    container.querySelectorAll('[data-ch]').forEach((b) => b.addEventListener('click', () => { st.chain = b.dataset.ch; update(); }));
    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    $('[data-n]').addEventListener('input', (e) => { st.logN = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 3.4 投机解码
  W.register('spec-speedup', function (container) {
    const st = { a: 0.7, k: 4, c: 0.1, seed: 1 };
    const expTok = (a, k) => (a >= 0.9999 ? k + 1 : (1 - a ** (k + 1)) / (1 - a));
    const speedup = (a, k, c) => expTok(a, k) / (1 + k * c);

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">投机解码能快多少</h4>
          <div class="lw-btn-row" style="margin:0"><button type="button" class="lw-btn" data-sim>重新模拟</button></div>
        </div>
        <p class="lw-subtitle">假设每个草稿 token 被接受的概率都是 α，且相互独立；一轮的耗时是目标模型一次前向（记为 1）加上草稿模型 K 次前向（每次 c）。</p>
        <div class="lw-controls">
          <div class="lw-control"><label><span>接受率 α</span><span class="lw-val" data-av></span></label><input type="range" min="0.2" max="0.95" step="0.05" value="${st.a}" data-a /></div>
          <div class="lw-control"><label><span>草稿长度 K</span><span class="lw-val" data-kv></span></label><input type="range" min="1" max="10" step="1" value="${st.k}" data-k /></div>
          <div class="lw-control"><label><span>草稿的相对开销 c</span><span class="lw-val" data-cv></span></label><input type="range" min="0" max="0.5" step="0.02" value="${st.c}" data-c /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function simulate() {
      const r = rng(st.seed * 7919 + Math.round(st.a * 100) + st.k * 13);
      const rounds = [];
      for (let i = 0; i < 12; i++) {
        let n = 0;
        while (n < st.k && r() < st.a) n++;
        rounds.push(n);
      }
      return rounds;
    }

    function update() {
      $('[data-av]').textContent = st.a.toFixed(2);
      $('[data-kv]').textContent = st.k;
      $('[data-cv]').textContent = st.c.toFixed(2);
      const E = expTok(st.a, st.k), S = speedup(st.a, st.k, st.c);
      const ks = Array.from({ length: 10 }, (_, i) => i + 1);
      const sp = ks.map((k) => speedup(st.a, k, st.c));
      const best = ks[sp.indexOf(Math.max(...sp))];
      const Wd = 640, H = 150, L = 40, B = 120, smax = Math.max(2, ...sp) * 1.1;
      const bw = (Wd - L - 10) / ks.length;
      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="不同 K 下的加速比">`;
      [1, Math.floor(smax)].forEach((v) => { const y = B - (v / smax) * (B - 10); svg += `<line x1="${L}" x2="${Wd - 10}" y1="${y}" y2="${y}" stroke="${W.color(v === 1 ? 'ink-3' : 'rule')}" stroke-dasharray="${v === 1 ? '4 3' : ''}"/><text x="${L - 6}" y="${y + 4}" font-size="10" fill="${W.color('ink-3')}" text-anchor="end">${v}×</text>`; });
      ks.forEach((k, i) => {
        const h = (sp[i] / smax) * (B - 10);
        const c = k === st.k ? W.color('decode') : k === best ? W.color('ok') : W.color('prefill');
        svg += `<rect x="${L + i * bw + 6}" y="${B - h}" width="${bw - 12}" height="${h}" fill="${c}" opacity="0.85"><title>K = ${k}：${sp[i].toFixed(2)} 倍</title></rect>`;
        svg += `<text x="${L + i * bw + bw / 2}" y="${B + 14}" font-size="10" fill="${W.color('ink-3')}" text-anchor="middle">${k}</text>`;
      });
      svg += `<text x="${(L + Wd) / 2}" y="${B + 28}" font-size="11" fill="${W.color('ink-3')}" text-anchor="middle">K 与加速比（琥珀色：当前 K；绿色：最优 K）</text></svg>`;
      // 模拟若干轮
      const rounds = simulate();
      const cw = 18;
      let sim = `<svg viewBox="0 0 ${Wd} ${6 * 24 + 6}" width="100%" role="img" aria-label="模拟的投机解码过程">`;
      let total = 0;
      rounds.forEach((n, i) => {
        const x0 = i < 6 ? 0 : 330, y = 4 + (i % 6) * 24;
        sim += `<text x="${x0}" y="${y + 13}" font-size="11" fill="${W.color('ink-3')}">第 ${i + 1} 轮</text>`;
        for (let j = 0; j <= st.k; j++) {
          if (j === st.k && n < st.k) continue;
          const fill = j < n ? W.color('ok') : j === n ? (n === st.k ? W.color('prefill') : W.color('err')) : W.color('rule');
          sim += `<rect x="${x0 + 60 + j * cw}" y="${y}" width="${cw - 3}" height="17" rx="2" fill="${fill}" opacity="${j > n ? 0.6 : 0.9}"/>`;
        }
        total += n + 1;
        sim += `<text x="${x0 + 60 + (st.k + 1) * cw + 6}" y="${y + 13}" font-size="11" fill="${W.color('ink-2')}">+${n + 1}</text>`;
      });
      sim += '</svg>';
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">每轮期望产出</div><div class="lw-stat-value">${E.toFixed(2)} 个 token</div><div class="lw-stat-sub">(1 − α^(K+1)) / (1 − α)</div></div>
          <div class="lw-stat"><div class="lw-stat-label">每轮耗时</div><div class="lw-stat-value">${(1 + st.k * st.c).toFixed(2)}</div><div class="lw-stat-sub">1 + K·c</div></div>
          <div class="lw-stat ${S > 1 ? 'ok' : 'warn'}"><div class="lw-stat-label">加速比</div><div class="lw-stat-value">${S.toFixed(2)} 倍</div><div class="lw-stat-sub">最优 K = ${best}</div></div>
        </div>
        <div class="lw-legend">
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('ok')}"></i>接受的草稿</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('err')}"></i>被拒绝，由目标模型重新采样</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('prefill')}"></i>全部接受时额外得到的 token</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('rule')}"></i>被丢弃的草稿</span>
        </div>
        ${sim}
        <div class="lw-callout">模拟的这 12 轮共产出 ${total} 个 token，不用投机解码需要目标模型前向 ${total} 次，现在只用了 12 次。${S < 1 ? '当前设置下草稿的开销超过了收益，投机解码反而更慢。' : st.k > best ? `K 超过了最优值：后面的草稿 token 被接受的概率是 α 的幂，越往后越低，却要付出同样的草稿开销。` : ''}</div>`;
    }

    $('[data-a]').addEventListener('input', (e) => { st.a = +e.target.value; update(); });
    $('[data-k]').addEventListener('input', (e) => { st.k = +e.target.value; update(); });
    $('[data-c]').addEventListener('input', (e) => { st.c = +e.target.value; update(); });
    $('[data-sim]').addEventListener('click', () => { st.seed++; update(); });
    update();
  });

  // ------------------------------------------------------------------ 3.5 并发与投机解码
  W.register('spec-batch', function (container) {
    const GP = W.GPUS || {}, MD = W.MODELS || {};
    if (!W.modelParams) { container.innerHTML = '<div class="lw-card">需要先加载 basics.js。</div>'; return; }
    const st = { gpu: GP.h100 ? 'h100' : Object.keys(GP)[0], model: MD['qwen3-8b'] ? 'qwen3-8b' : Object.keys(MD)[0], a: 0.6, k: 3, logb: 2, ctx: 512 };
    const C_DRAFT = 0.05;

    function times(g, m, b) {
      const p = W.modelParams(m);
      const P = p.total - p.emb;
      const kvTok = 2 * m.L * m.nkv * m.dh * 2;
      const bw = g.bw * 1e12 * 0.8, fl = g.bf16 * 1e12 * 0.6;
      const attnFlops = 4 * st.ctx * m.nq * m.dh * m.L;   // 每个查询 token 的注意力计算量
      const fwd = (nTok) => Math.max((2 * P) / bw, (2 * P * nTok) / fl) + Math.max((b * st.ctx * kvTok) / bw, (nTok * attnFlops) / fl);
      const E = (1 - st.a ** (st.k + 1)) / (1 - st.a);
      const t0 = fwd(b);
      const tr = fwd(b * (st.k + 1)) + st.k * C_DRAFT * t0;
      const bmax = Math.floor((g.mem * 1e9 * 0.9 - 2 * p.total) / (st.ctx * kvTok));
      return { t0, tr, E, base: b / t0, spec: (b * E) / tr, bmax };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">并发增大时，投机解码还划算吗</h4></div>
        <p class="lw-subtitle">按 Roofline 估算一步 decode 的耗时：读权重和计算取较大者，再加上注意力的时间（读 KV Cache 和计算取较大者）。投机解码每轮要验证 batch ×（K+1）个 token，草稿开销按目标的 5% 计（类似 EAGLE 这种很小的草稿头）。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>GPU</label><select data-gpu>${Object.keys(GP).map((k) => `<option value="${k}" ${k === st.gpu ? 'selected' : ''}>${GP[k].name}</option>`).join('')}</select></div>
          <div class="lw-control"><label>模型</label><select data-model>${Object.keys(MD).map((k) => `<option value="${k}" ${k === st.model ? 'selected' : ''}>${MD[k].name}</option>`).join('')}</select></div>
          <div class="lw-control"><label><span>接受率 α</span><span class="lw-val" data-av></span></label><input type="range" min="0.3" max="0.9" step="0.05" value="${st.a}" data-a /></div>
          <div class="lw-control"><label><span>草稿长度 K</span><span class="lw-val" data-kv></span></label><input type="range" min="1" max="8" step="1" value="${st.k}" data-k /></div>
          <div class="lw-control"><label><span>并发请求数</span><span class="lw-val" data-bv></span></label><input type="range" min="0" max="9" step="1" value="${st.logb}" data-b /></div>
          <div class="lw-control"><label>每个请求的上下文</label><select data-ctx>${[512, 1024, 4096, 16384].map((c) => `<option value="${c}" ${c === st.ctx ? 'selected' : ''}>${c}</option>`).join('')}</select></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      const g = GP[st.gpu], m = MD[st.model];
      const b = 2 ** st.logb;
      $('[data-av]').textContent = st.a.toFixed(2);
      $('[data-kv]').textContent = st.k;
      $('[data-bv]').textContent = b;
      const bs = Array.from({ length: 10 }, (_, i) => 2 ** i);
      const ratio = bs.map((x) => { const t = times(g, m, x); return t.spec / t.base; });
      const Wd = 640, H = 190, L = 44, B = 150, ymax = Math.max(1.5, ...ratio) * 1.1;
      const X = (i) => L + (i / 9) * (Wd - L - 16);
      const Y = (v) => B - (v / ymax) * (B - 12);
      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="投机解码的吞吐加速比随并发的变化">`;
      for (let v = 0.5; v <= ymax; v += 0.5) svg += `<line x1="${L}" x2="${Wd - 16}" y1="${Y(v)}" y2="${Y(v)}" stroke="${W.color(v === 1 ? 'ink-3' : 'rule')}" stroke-dasharray="${v === 1 ? '4 3' : ''}"/><text x="${L - 6}" y="${Y(v) + 4}" font-size="10" fill="${W.color('ink-3')}" text-anchor="end">${v.toFixed(1)}×</text>`;
      bs.forEach((x, i) => { svg += `<text x="${X(i)}" y="${B + 15}" font-size="10" fill="${W.color('ink-3')}" text-anchor="middle">${x}</text>`; });
      const bmax = times(g, m, 1).bmax;
      const fit = bs.filter((x) => x <= bmax).length;
      svg += `<polyline points="${ratio.slice(0, fit).map((v, i) => `${X(i)},${Y(v)}`).join(' ')}" fill="none" stroke="${W.color('prefill')}" stroke-width="2"/>`;
      if (fit < bs.length) {
        svg += `<polyline points="${ratio.slice(Math.max(0, fit - 1)).map((v, i) => `${X(i + Math.max(0, fit - 1))},${Y(v)}`).join(' ')}" fill="none" stroke="${W.color('rule-strong')}" stroke-width="2" stroke-dasharray="4 3"/>`;
        svg += `<text x="${X(Math.max(0, fit - 1)) + 6}" y="${Y(ymax) + 12}" font-size="10" fill="${W.color('ink-3')}">右侧：KV Cache 放不下</text>`;
      }
      svg += `<circle cx="${X(st.logb)}" cy="${Y(ratio[st.logb])}" r="5" fill="${W.color('decode')}"/>`;
      svg += `<text x="${(L + Wd) / 2}" y="${B + 32}" font-size="11" fill="${W.color('ink-3')}" text-anchor="middle">并发请求数与吞吐的加速比（低于虚线表示投机解码反而降低了吞吐）</text></svg>`;
      const t = times(g, m, b);
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">不用投机解码的 ITL</div><div class="lw-stat-value">${(t.t0 * 1e3).toFixed(1)} ms</div><div class="lw-stat-sub">吞吐 ${Math.round(t.base)} token/s</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">投机解码的平均 ITL</div><div class="lw-stat-value">${((t.tr / t.E) * 1e3).toFixed(1)} ms</div><div class="lw-stat-sub">吞吐 ${Math.round(t.spec)} token/s</div></div>
          <div class="lw-stat ${t.spec > t.base ? 'ok' : 'warn'}"><div class="lw-stat-label">吞吐加速比</div><div class="lw-stat-value">${(t.spec / t.base).toFixed(2)} 倍</div></div>
        </div>
        <div class="lw-callout">${b > t.bmax ? `这张卡放不下 ${b} 个上下文 ${st.ctx} 的请求（最多约 ${t.bmax} 个），这里的数字只是外推。` : ''}${t.spec > t.base * 1.05
          ? (b <= 16 ? '并发较低时，验证多个 token 几乎不增加耗时，投机解码同时改善了延迟和吞吐。'
            : '并发虽高，但每个请求的上下文较长，读 KV Cache 的时间占了大头。验证 K+1 个查询读的是同一份 KV，这部分开销没有增加，所以投机解码仍有收益。')
          : t.spec >= t.base * 0.95 ? '在这个并发下，验证的额外计算基本抵消了收益。'
          : '并发高、上下文短时，线性层已经是计算瓶颈，验证被拒绝的草稿 token 是白白消耗算力，吞吐下降。单个请求的延迟可能还略好，但整体能服务的请求变少了。'}</div>`;
    }

    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    $('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    $('[data-a]').addEventListener('input', (e) => { st.a = +e.target.value; update(); });
    $('[data-k]').addEventListener('input', (e) => { st.k = +e.target.value; update(); });
    $('[data-b]').addEventListener('input', (e) => { st.logb = +e.target.value; update(); });
    $('[data-ctx]').addEventListener('change', (e) => { st.ctx = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 3.6 时间线
  W.register('launch-timeline', function (container) {
    const GP = W.GPUS || {}, MD = W.MODELS || {};
    if (!W.modelParams) { container.innerHTML = '<div class="lw-card">需要先加载 basics.js。</div>'; return; }
    const st = { model: MD['qwen3-0.6b'] ? 'qwen3-0.6b' : Object.keys(MD)[0], gpu: GP.h100 ? 'h100' : Object.keys(GP)[0], graph: false };
    const CPU_EAGER = 12e-6, GAP = 0.5e-6, MIN_K = 2.5e-6, GRAPH_LAUNCH = 30e-6;

    // 一层 decode（batch 1）中的 kernel：名称与读的权重字节数（BF16）
    function layerKernels(m) {
      const qkv = m.h * (m.nq + 2 * m.nkv) * m.dh * 2, o = m.nq * m.dh * m.h * 2;
      const gu = 2 * m.h * m.inter * 2, dn = m.inter * m.h * 2;
      return [['RMSNorm', 0], ['QKV 投影', qkv], ['q/k norm', 0], ['RoPE', 0], ['写入 KV', 0], ['注意力', 2 * 1024 * m.nkv * m.dh * 2],
        ['O 投影', o], ['残差', 0], ['RMSNorm', 0], ['gate/up 投影', gu], ['SwiGLU', 0], ['down 投影', dn], ['残差', 0]];
    }

    function simulate() {
      const m = MD[st.model], g = GP[st.gpu];
      const bw = g.bw * 1e12 * 0.75;
      const ks = [];
      for (let l = 0; l < m.L; l++) layerKernels(m).forEach(([n, b]) => ks.push({ n, d: Math.max(MIN_K, b / bw), layer: l }));
      ks.push({ n: '输出头', d: Math.max(MIN_K, (m.V * m.h * 2) / bw), layer: m.L });
      let cpu = 0, gpu = 0;
      const out = [];
      if (st.graph) {
        cpu = GRAPH_LAUNCH;
        gpu = GRAPH_LAUNCH;
        ks.forEach((k) => { const s0 = gpu; gpu = s0 + k.d + GAP; out.push({ ...k, s: s0, cpuAt: null }); });
      } else {
        ks.forEach((k) => {
          cpu += CPU_EAGER;                         // Python 与 PyTorch 分派、发射这个 kernel
          const s0 = Math.max(cpu, gpu);
          gpu = s0 + k.d;
          out.push({ ...k, s: s0, cpuAt: cpu });
        });
      }
      const busy = ks.reduce((a, k) => a + k.d, 0);
      return { ks: out, total: Math.max(cpu, gpu), busy, n: ks.length, cpu };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">一步 decode 的时间线</h4></div>
        <p class="lw-subtitle">batch 为 1、上下文 1024 的一步 decode。每个 kernel 的 GPU 耗时按读的字节数估算（最短 2.5 微秒）。逐个发射时，CPU 每发射一个 kernel 约需 12 微秒（Python、PyTorch 分派和 CUDA 调用）；CUDA Graph 一次发射整张图。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label><select data-model>${Object.keys(MD).map((k) => `<option value="${k}" ${k === st.model ? 'selected' : ''}>${MD[k].name}</option>`).join('')}</select></div>
          <div class="lw-control"><label>GPU</label><select data-gpu>${Object.keys(GP).map((k) => `<option value="${k}" ${k === st.gpu ? 'selected' : ''}>${GP[k].name}</option>`).join('')}</select></div>
        </div>
        <div class="lw-tabs">
          <button type="button" class="lw-tab" data-g="0">逐个发射（eager）</button>
          <button type="button" class="lw-tab" data-g="1">CUDA Graph</button>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      container.querySelectorAll('[data-g]').forEach((b) => b.classList.toggle('active', b.dataset.g === (st.graph ? '1' : '0')));
      const r = simulate();
      const Wd = 640, L = 52, R = 10;
      // 只画前两层，展示细节
      const shown = r.ks.filter((k) => k.layer < 2);
      const tEnd = Math.max(...shown.map((k) => k.s + k.d), st.graph ? 0 : shown[shown.length - 1].cpuAt) * 1.02;
      const X = (t) => L + (t / tEnd) * (Wd - L - R);
      let svg = `<svg viewBox="0 0 ${Wd} 120" width="100%" role="img" aria-label="前两层的 CPU 与 GPU 时间线">`;
      svg += `<text x="0" y="34" font-size="11" fill="${W.color('ink-2')}">CPU</text><text x="0" y="74" font-size="11" fill="${W.color('ink-2')}">GPU</text>`;
      svg += `<line x1="${L}" x2="${Wd - R}" y1="48" y2="48" stroke="${W.color('rule')}"/>`;
      if (st.graph) {
        svg += `<rect x="${X(0)}" y="22" width="${Math.max(2, X(GRAPH_LAUNCH) - X(0))}" height="18" fill="${W.color('decode')}" opacity="0.8"><title>发射整张 CUDA Graph</title></rect>`;
        svg += `<text x="${X(GRAPH_LAUNCH) + 6}" y="35" font-size="10" fill="${W.color('ink-3')}">一次发射之后，CPU 就空闲了</text>`;
      } else {
        shown.forEach((k) => { svg += `<rect x="${X(k.cpuAt - CPU_EAGER)}" y="22" width="${Math.max(1, X(k.cpuAt) - X(k.cpuAt - CPU_EAGER) - 1)}" height="18" fill="${W.color('decode')}" opacity="0.7"><title>发射 ${k.n}</title></rect>`; });
      }
      shown.forEach((k) => {
        const lin = /投影|注意力/.test(k.n);
        svg += `<rect x="${X(k.s)}" y="62" width="${Math.max(1, X(k.s + k.d) - X(k.s) - 0.5)}" height="18" fill="${lin ? W.color('prefill') : W.color('ok')}" opacity="0.85"><title>${k.n}：${(k.d * 1e6).toFixed(1)} 微秒</title></rect>`;
      });
      svg += `<text x="${L}" y="104" font-size="10" fill="${W.color('ink-3')}">只画了前两层（共 ${MD[st.model].L} 层）。蓝色：矩阵乘和注意力；绿色：小算子；空白：GPU 在等 CPU。横轴总长 ${(tEnd * 1e6).toFixed(0)} 微秒</text>`;
      svg += '</svg>';
      const util = r.busy / r.total;
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">一步的 kernel 数</div><div class="lw-stat-value">${r.n}</div></div>
          <div class="lw-stat ${util < 0.6 ? 'warn' : 'ok'}"><div class="lw-stat-label">一步的耗时</div><div class="lw-stat-value">${(r.total * 1e3).toFixed(2)} ms</div><div class="lw-stat-sub">GPU 忙碌 ${(util * 100).toFixed(0)}%</div></div>
          <div class="lw-stat"><div class="lw-stat-label">GPU 实际计算</div><div class="lw-stat-value">${(r.busy * 1e3).toFixed(2)} ms</div><div class="lw-stat-sub">不考虑 CPU 时的下限</div></div>
        </div>
        <div class="lw-callout">${st.graph
          ? 'CUDA Graph 把整步的 kernel 录制下来，之后一次调用重放，CPU 的开销只剩一次发射，kernel 之间几乎没有空隙。代价是输入的形状必须固定，所以 vLLM 会为若干种 batch 大小分别录制。'
          : util < 0.6 ? 'GPU 一大半时间在等 CPU 发射下一个 kernel。这就是原书说的"GPU 利用率看起来不低，kernel 之间却有大量空隙"。模型越小、GPU 越快，这个问题越严重。'
          : '模型足够大时，每个 kernel 的 GPU 耗时超过了发射开销，CPU 能跑在 GPU 前面，空隙就消失了。'}</div>`;
    }

    $('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    $('[data-gpu]').addEventListener('change', (e) => { st.gpu = e.target.value; update(); });
    container.querySelectorAll('[data-g]').forEach((b) => b.addEventListener('click', () => { st.graph = b.dataset.g === '1'; update(); }));
    update();
  });
})();
