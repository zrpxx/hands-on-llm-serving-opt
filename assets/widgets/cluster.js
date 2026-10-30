/**
 * 第 4 部分的交互组件
 * - tp-scaling          4.1 张量并行：每张卡的显存、计算与通信
 * - moe-ep              4.2 专家并行的负载与 all-to-all 通信
 * - pd-transfer         4.3 PD 分离的 KV 传输带宽
 * - kv-tier             4.4 重算还是从哪一层存储加载 KV
 * - cold-start          4.5 模型冷启动的耗时
 * - cache-router        4.6 路由策略与前缀缓存命中率
 * - lora-cost           4.7 LoRA 适配器的显存与计算开销
 *
 * 依赖 basics.js 中的 W.MODELS、W.GPUS、W.modelParams。
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  }
  const GB = 1e9;
  const sel = (attr, obj, cur, label = (v) => v.name) => `<select ${attr}>${Object.keys(obj).map((k) => `<option value="${k}" ${k === cur ? 'selected' : ''}>${label(obj[k])}</option>`).join('')}</select>`;
  const need = (container) => {
    if (!W.MODELS || !W.GPUS || !W.modelParams) { container.innerHTML = '<div class="lw-card">需要先加载 basics.js。</div>'; return false; }
    return true;
  };
  const kvPerTok = (m) => 2 * m.L * m.nkv * m.dh * 2;

  // ------------------------------------------------------------------ 4.1 张量并行
  const LINKS = {
    nvlink4: { name: 'NVLink（H100，单向 450 GB/s）', bw: 450, lat: 8e-6 },
    pcie5: { name: 'PCIe 5.0 x16（约 50 GB/s）', bw: 50, lat: 15e-6 },
    pcie4: { name: 'PCIe 4.0 x16（约 25 GB/s）', bw: 25, lat: 15e-6 },
  };

  W.register('tp-scaling', function (container) {
    if (!need(container)) return;
    const MD = W.MODELS, GP = W.GPUS;
    const st = { model: MD['llama3-70b'] ? 'llama3-70b' : Object.keys(MD)[0], gpu: GP.h100 ? 'h100' : Object.keys(GP)[0], link: 'nvlink4', logTok: 5 };

    function stepTime(tp) {
      const m = MD[st.model], g = GP[st.gpu], l = LINKS[st.link];
      const P = W.modelParams(m).total;
      const tokens = 2 ** st.logTok;
      const wBytes = (P * 2) / tp;
      const bw = g.bw * 1e12 * 0.8, fl = g.bf16 * 1e12 * 0.6;
      const tLin = Math.max(wBytes / bw, (2 * P * tokens) / tp / fl);
      // 每层两次 all-reduce（注意力输出、FFN 输出），每次 [tokens, hidden] 的 BF16
      const msg = tokens * m.h * 2;
      const tAr = tp === 1 ? 0 : 2 * m.L * ((2 * (tp - 1)) / tp * msg / (l.bw * GB) + l.lat);
      const fits = wBytes < g.mem * GB * 0.85;
      return { tLin, tAr, total: tLin + tAr, wBytes, fits };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">张量并行：几张卡、怎样连接</h4></div>
        <p class="lw-subtitle">一步前向的估算：每张卡读 1/TP 的权重、做 1/TP 的计算；每层有两次 all-reduce，每次传 [token 数 × hidden] 的 BF16 激活，按 ring 算法计算耗时，每次另有固定延迟。不含注意力读 KV 的时间。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${sel('data-model', MD, st.model)}</div>
          <div class="lw-control"><label>GPU</label>${sel('data-gpu', GP, st.gpu)}</div>
          <div class="lw-control"><label>卡间连接</label>${sel('data-link', LINKS, st.link)}</div>
          <div class="lw-control"><label><span>这一步的 token 数</span><span class="lw-val" data-tv></span></label><input type="range" min="0" max="13" step="1" value="${st.logTok}" data-tok /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      $('[data-tv]').textContent = 2 ** st.logTok;
      const tps = [1, 2, 4, 8];
      const res = tps.map(stepTime);
      const tmax = Math.max(...res.map((r) => r.total));
      const Wd = 640, rowH = 34, L = 70, R = 150;
      let svg = `<svg viewBox="0 0 ${Wd} ${tps.length * rowH + 10}" width="100%" role="img" aria-label="不同 TP 下一步的耗时">`;
      res.forEach((r, i) => {
        const y = i * rowH + 6;
        const w1 = (r.tLin / tmax) * (Wd - L - R), w2 = (r.tAr / tmax) * (Wd - L - R);
        svg += `<text x="0" y="${y + 17}" font-size="12" fill="${W.color('ink')}">TP = ${tps[i]}</text>`;
        svg += `<rect x="${L}" y="${y + 4}" width="${Math.max(1, w1)}" height="18" fill="${W.color('prefill')}" opacity="${r.fits ? 0.85 : 0.3}"/>`;
        svg += `<rect x="${L + w1}" y="${y + 4}" width="${Math.max(0, w2)}" height="18" fill="${W.color('decode')}" opacity="${r.fits ? 0.85 : 0.3}"/>`;
        svg += `<text x="${L + w1 + w2 + 6}" y="${y + 17}" font-size="11" fill="${W.color('ink-2')}">${r.fits ? `${(r.total * 1e3).toFixed(2)} ms，通信占 ${((r.tAr / r.total) * 100).toFixed(0)}%` : `放不下（每卡 ${(r.wBytes / GB).toFixed(0)} GB）`}</text>`;
      });
      svg += '</svg>';
      const ok = res.map((r, i) => ({ r, tp: tps[i] })).filter((x) => x.r.fits);
      const best = ok.length ? ok.reduce((a, b) => (b.r.total < a.r.total ? b : a)) : null;
      const firstFit = ok.length ? ok[0] : null;
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-legend">
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('prefill')}"></i>读权重或计算（取较大者）</span>
          <span class="lw-legend-item"><i class="lw-dot" style="background:${W.color('decode')}"></i>all-reduce 通信</span>
        </div>
        <div class="lw-callout">${!best ? '8 张卡也放不下，需要更多的卡、流水线并行或者量化。'
          : `最少需要 TP = ${firstFit.tp} 才能放下权重；这一步最快的是 TP = ${best.tp}。${st.link !== 'nvlink4' ? ' 没有 NVLink 时，all-reduce 很快成为主要开销，TP 越大越不划算，原书建议这时尽量少用 TP，或者改用流水线并行。' : ' 有 NVLink 时通信占比不高，增大 TP 能明显降低每一步的延迟，这是单机多卡最常用的配置。'}`}</div>`;
    }
    ['model', 'gpu', 'link'].forEach((k) => $(`[data-${k}]`).addEventListener('change', (e) => { st[k] = e.target.value; update(); }));
    $('[data-tok]').addEventListener('input', (e) => { st.logTok = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 4.2 专家并行
  const MOE = {
    dsv3: { name: 'DeepSeek-V3（256 个专家，每 token 8 个）', E: 256, k: 8, h: 7168 },
    qwen235: { name: 'Qwen3-235B-A22B（128 个专家，每 token 8 个）', E: 128, k: 8, h: 4096 },
    gptoss: { name: 'gpt-oss-120b（128 个专家，每 token 4 个）', E: 128, k: 4, h: 2880 },
    mixtral: { name: 'Mixtral 8x7B（8 个专家，每 token 2 个）', E: 8, k: 2, h: 4096 },
  };

  W.register('moe-ep', function (container) {
    const st = { m: 'dsv3', ep: 8, logTok: 12, skew: 0.6, seed: 1 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">专家并行：每张卡分到多少 token</h4>
          <div class="lw-btn-row" style="margin:0"><button type="button" class="lw-btn" data-re>换一批 token</button></div>
        </div>
        <p class="lw-subtitle">专家平均分到 EP 张卡上。每个 token 按路由器的结果发往它选中的 k 个专家所在的卡（dispatch），算完再送回（combine）。"热度不均"控制专家受欢迎程度的差异：0 表示每个专家被选中的概率相同。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${sel('data-m', MOE, st.m)}</div>
          <div class="lw-control"><label>EP 大小</label><select data-ep>${[2, 4, 8, 16, 32, 64].map((v) => `<option ${v === st.ep ? 'selected' : ''}>${v}</option>`).join('')}</select></div>
          <div class="lw-control"><label><span>这一步的 token 数</span><span class="lw-val" data-tv></span></label><input type="range" min="6" max="15" step="1" value="${st.logTok}" data-tok /></div>
          <div class="lw-control"><label><span>专家热度不均</span><span class="lw-val" data-sv></span></label><input type="range" min="0" max="1.5" step="0.1" value="${st.skew}" data-skew /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);

    function update() {
      const m = MOE[st.m];
      const ep = Math.min(st.ep, m.E);
      const T = 2 ** st.logTok;
      $('[data-tv]').textContent = T;
      $('[data-sv]').textContent = st.skew.toFixed(1);
      const r = rng(st.seed * 31 + st.logTok);
      // 专家热度：Zipf 分布，随机打乱顺序，再按顺序放到各张卡上
      const order = Array.from({ length: m.E }, (_, i) => i).sort(() => r() - 0.5);
      const w = order.map((rank) => 1 / (rank + 1) ** st.skew);
      const tot = w.reduce((a, b) => a + b, 0);
      const cdf = []; w.reduce((a, b, i) => (cdf[i] = a + b / tot), 0);
      const perExpert = new Array(m.E).fill(0);
      for (let t = 0; t < T; t++) {
        const chosen = new Set();
        while (chosen.size < m.k) {
          const u = r();
          let lo = 0, hi = m.E - 1;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
          chosen.add(lo);
        }
        chosen.forEach((e) => { perExpert[e]++; });
      }
      const perGpu = Array.from({ length: ep }, (_, g) => perExpert.slice((g * m.E) / ep, ((g + 1) * m.E) / ep).reduce((a, b) => a + b, 0));
      const mean = (T * m.k) / ep, mx = Math.max(...perGpu);
      const a2a = 2 * T * m.k * m.h * 2 * ((ep - 1) / ep);   // dispatch + combine，BF16
      const Wd = 640, H = 150, L = 36, B = 120;
      const bw = (Wd - L - 10) / ep;
      const vmax = Math.max(mx, mean * 1.2);
      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="每张卡收到的 token 数">`;
      const ym = B - (mean / vmax) * (B - 10);
      perGpu.forEach((v, i) => {
        const h = (v / vmax) * (B - 10);
        svg += `<rect x="${L + i * bw + (ep > 16 ? 0.5 : 3)}" y="${B - h}" width="${Math.max(1, bw - (ep > 16 ? 1 : 6))}" height="${h}" fill="${v === mx ? W.color('decode') : W.color('prefill')}" opacity="0.85"><title>GPU ${i}：${v} 个 token-专家对</title></rect>`;
      });
      svg += `<line x1="${L}" x2="${Wd - 10}" y1="${ym}" y2="${ym}" stroke="${W.color('ink-3')}" stroke-dasharray="4 3"/><text x="${L - 4}" y="${ym + 4}" font-size="10" fill="${W.color('ink-3')}" text-anchor="end">平均</text>`;
      svg += `<text x="${(L + Wd) / 2}" y="${B + 22}" font-size="11" fill="${W.color('ink-3')}" text-anchor="middle">每张卡收到的 token 数（琥珀色：最忙的卡）</text></svg>`;
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">每张卡的专家数</div><div class="lw-stat-value">${m.E / ep}</div></div>
          <div class="lw-stat ${mx / mean > 1.3 ? 'warn' : 'ok'}"><div class="lw-stat-label">最忙 / 平均</div><div class="lw-stat-value">${(mx / mean).toFixed(2)} 倍</div><div class="lw-stat-sub">MoE 层的耗时由最忙的卡决定</div></div>
          <div class="lw-stat"><div class="lw-stat-label">all-to-all 通信量</div><div class="lw-stat-value">${(a2a / 2 ** 20).toFixed(0)} MiB</div><div class="lw-stat-sub">每层，dispatch + combine</div></div>
        </div>
        <div class="lw-callout">${mx / mean > 1.3
          ? `热门专家集中的卡要处理 ${(mx / mean).toFixed(1)} 倍于平均的 token，其他卡要等它算完。这就是 EPLB 要解决的问题：复制热门专家，重新分配专家的位置（6.3 节）。`
          : 'token 数足够多、专家热度差异不大时，各卡的负载比较均衡。token 少或者热度不均时，差距会迅速拉大。'}${st.logTok <= 8 ? ' token 很少时（decode），每张卡分到的 token 不多，专家的矩阵乘法很小，通信延迟占比更高。' : ''}</div>`;
    }
    $('[data-m]').addEventListener('change', (e) => { st.m = e.target.value; update(); });
    $('[data-ep]').addEventListener('change', (e) => { st.ep = +e.target.value; update(); });
    $('[data-tok]').addEventListener('input', (e) => { st.logTok = +e.target.value; update(); });
    $('[data-skew]').addEventListener('input', (e) => { st.skew = +e.target.value; update(); });
    $('[data-re]').addEventListener('click', () => { st.seed++; update(); });
    update();
  });

  // ------------------------------------------------------------------ 4.3 PD 分离的 KV 传输
  const NETS = {
    nvlink: { name: '同机 NVLink（单向 450 GB/s）', bw: 450 },
    ib8: { name: '跨机 RDMA，8 张 400 Gb/s 网卡（约 400 GB/s）', bw: 400 },
    ib1: { name: '跨机 RDMA，1 张 400 Gb/s 网卡（约 50 GB/s）', bw: 50 },
    tcp: { name: '跨机 TCP，无 RDMA（约 10 GB/s）', bw: 10 },
  };

  W.register('pd-transfer', function (container) {
    if (!need(container)) return;
    const MD = W.MODELS;
    const st = { model: MD['qwen3-8b'] ? 'qwen3-8b' : Object.keys(MD)[0], net: 'ib1', logIn: 13, rate: 16 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">PD 分离要传多少 KV</h4></div>
        <p class="lw-subtitle">每个请求 prefill 完成后，它的全部 KV 要从 prefill 实例传到 decode 实例（BF16）。原书的估算：8B 模型 1024 个输入 token 的 KV 约 0.1～0.15 GB。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${sel('data-model', MD, st.model)}</div>
          <div class="lw-control"><label>传输链路</label>${sel('data-net', NETS, st.net)}</div>
          <div class="lw-control"><label><span>每个请求的输入长度</span><span class="lw-val" data-iv></span></label><input type="range" min="9" max="16" step="1" value="${st.logIn}" data-in /></div>
          <div class="lw-control"><label><span>每秒请求数（这一组 prefill 实例）</span><span class="lw-val" data-rv></span></label><input type="range" min="1" max="64" step="1" value="${st.rate}" data-rate /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      const m = MD[st.model], n = NETS[st.net];
      const T = 2 ** st.logIn;
      $('[data-iv]').textContent = T >= 1024 ? `${T / 1024}K` : T;
      $('[data-rv]').textContent = st.rate;
      const bytes = T * kvPerTok(m);
      const need = (bytes * st.rate) / GB;
      const t = bytes / (n.bw * GB);
      const util = need / n.bw;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">每个请求的 KV</div><div class="lw-stat-value">${(bytes / GB).toFixed(2)} GB</div></div>
          <div class="lw-stat ${util > 0.7 ? 'warn' : ''}"><div class="lw-stat-label">需要的带宽</div><div class="lw-stat-value">${need.toFixed(1)} GB/s</div><div class="lw-stat-sub">链路占用 ${(util * 100).toFixed(0)}%</div></div>
          <div class="lw-stat"><div class="lw-stat-label">整体传输一个请求</div><div class="lw-stat-value">${(t * 1e3).toFixed(1)} ms</div><div class="lw-stat-sub">逐层传输时暴露约 ${((t / m.L) * 1e3).toFixed(2)} ms</div></div>
        </div>
        <div class="lw-callout">${util > 1 ? '需要的带宽超过了链路的能力，传输会越积越多，decode 实例拿不到 KV。要么把 prefill 和 decode 放在同一台机器上，要么换更快的网络，或者压缩 KV（2.5 节）。'
          : util > 0.7 ? '链路接近饱和，传输要排队，第二个 token 的延迟会明显变长。项目 4 的模拟器可以观察到这一点。'
          : '链路有余量。逐层传输时，大部分 KV 在 prefill 计算期间就传完了，几乎不增加延迟。'}</div>`;
    }
    ['model', 'net'].forEach((k) => $(`[data-${k}]`).addEventListener('change', (e) => { st[k] = e.target.value; update(); }));
    $('[data-in]').addEventListener('input', (e) => { st.logIn = +e.target.value; update(); });
    $('[data-rate]').addEventListener('input', (e) => { st.rate = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 4.4 KV 分层存储
  const TIERS = [
    { key: 'gpu', name: 'GPU 显存（命中）', bw: null },
    { key: 'cpu', name: 'CPU 内存，经 PCIe 5.0', bw: 50 },
    { key: 'rdma', name: '远端内存，经 RDMA', bw: 25 },
    { key: 'ssd', name: '本地 NVMe SSD', bw: 6 },
    { key: 'net', name: '远端存储，经 TCP', bw: 2 },
  ];

  W.register('kv-tier', function (container) {
    if (!need(container)) return;
    const MD = W.MODELS, GP = W.GPUS;
    const st = { model: MD['qwen3-8b'] ? 'qwen3-8b' : Object.keys(MD)[0], gpu: GP.h100 ? 'h100' : Object.keys(GP)[0], logT: 14 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">重算，还是加载</h4></div>
        <p class="lw-subtitle">一个长度为 T 的前缀，重新 prefill 的耗时，与从各层存储把它的 KV 读回 GPU 的耗时比较。重算按线性层与注意力的计算量、峰值算力的 60% 估算；加载按各层的典型带宽估算，不含查找等开销。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${sel('data-model', MD, st.model)}</div>
          <div class="lw-control"><label>GPU</label>${sel('data-gpu', GP, st.gpu)}</div>
          <div class="lw-control"><label><span>前缀长度</span><span class="lw-val" data-tv></span></label><input type="range" min="10" max="17" step="1" value="${st.logT}" data-t /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      const m = MD[st.model], g = GP[st.gpu];
      const T = 2 ** st.logT;
      $('[data-tv]').textContent = `${T / 1024}K token`;
      const P = W.modelParams(m).total;
      const fl = g.bf16 * 1e12 * 0.6;
      const recompute = (2 * P * T + 2 * m.L * m.nq * m.dh * T * T) / fl;   // 线性层 + 因果注意力
      const bytes = T * kvPerTok(m);
      const rows = [{ name: '重新 prefill', t: recompute, c: W.color('decode') }].concat(
        TIERS.filter((x) => x.bw).map((x) => ({ name: x.name, t: bytes / (x.bw * GB), c: W.color('prefill') })));
      const tmax = Math.max(...rows.map((x) => x.t));
      const Wd = 640, rowH = 28, L = 168, R = 90;
      let svg = `<svg viewBox="0 0 ${Wd} ${rows.length * rowH + 8}" width="100%" role="img" aria-label="重算与加载的耗时比较">`;
      rows.forEach((x, i) => {
        const y = i * rowH + 4;
        const w = Math.max(2, (x.t / tmax) * (Wd - L - R));
        const faster = i > 0 && x.t < recompute;
        svg += `<text x="0" y="${y + 15}" font-size="12" fill="${W.color('ink')}">${x.name}</text>`;
        svg += `<rect x="${L}" y="${y + 3}" width="${w}" height="16" fill="${i === 0 ? x.c : faster ? W.color('ok') : W.color('rule-strong')}" opacity="0.85"/>`;
        svg += `<text x="${L + w + 6}" y="${y + 15}" font-size="11" fill="${W.color('ink-2')}">${x.t >= 1 ? `${x.t.toFixed(2)} s` : `${(x.t * 1e3).toFixed(0)} ms`}</text>`;
      });
      svg += '</svg>';
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">这个前缀的 KV</div><div class="lw-stat-value">${(bytes / GB).toFixed(2)} GB</div></div>
          <div class="lw-stat"><div class="lw-stat-label">重算的耗时</div><div class="lw-stat-value">${recompute >= 1 ? `${recompute.toFixed(2)} s` : `${(recompute * 1e3).toFixed(0)} ms`}</div><div class="lw-stat-sub">注意力部分随长度平方增长</div></div>
        </div>
        <div class="lw-callout">绿色的层比重算快。前缀越长，重算的注意力开销按平方增长，而加载的时间只按线性增长，所以长前缀更值得缓存到下一层存储。KV 越小（GQA、MLA、量化），加载越划算。</div>`;
    }
    ['model', 'gpu'].forEach((k) => $(`[data-${k}]`).addEventListener('change', (e) => { st[k] = e.target.value; update(); }));
    $('[data-t]').addEventListener('input', (e) => { st.logT = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 4.5 冷启动
  const SOURCES = {
    hub: { name: '从 Hugging Face 下载（约 0.5 GB/s）', bw: 0.5 },
    s3: { name: '从对象存储并行读取（约 2 GB/s）', bw: 2 },
    nvme: { name: '本地 NVMe（约 5 GB/s）', bw: 5 },
    ram: { name: '已在内存的页缓存中（约 20 GB/s）', bw: 20 },
  };

  W.register('cold-start', function (container) {
    const st = { size: 16, src: 's3', compile: true, graph: true, sleep: false };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">一个模型从零启动要多久</h4></div>
        <p class="lw-subtitle">粗略的量级估计：读权重的时间由来源的带宽决定；进程与 CUDA 初始化约 10 秒；torch.compile 首次编译约 30 秒（有缓存时约 5 秒）；录制 CUDA Graph 约 10 秒。实际数字取决于版本、硬件和模型。</p>
        <div class="lw-controls">
          <div class="lw-control"><label><span>权重大小</span><span class="lw-val" data-sv></span></label><input type="range" min="1" max="200" step="1" value="${st.size}" data-size /></div>
          <div class="lw-control"><label>权重来源</label>${sel('data-src', SOURCES, st.src)}</div>
        </div>
        <div class="lw-tabs">
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:0.84rem"><input type="checkbox" data-c checked /> 首次 torch.compile</label>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:0.84rem;margin-left:12px"><input type="checkbox" data-g checked /> 录制 CUDA Graph</label>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:0.84rem;margin-left:12px"><input type="checkbox" data-sl /> 从睡眠模式唤醒</label>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      $('[data-sv]').textContent = `${st.size} GB`;
      let parts;
      if (st.sleep) {
        parts = [['权重从 CPU 内存拷回 GPU（约 25 GB/s）', st.size / 25], ['恢复 KV Cache 等状态', 0.5]];
      } else {
        parts = [['进程与 CUDA 初始化', 10], ['读取权重', st.size / SOURCES[st.src].bw], ['torch.compile', st.compile ? 30 : 5]];
        if (st.graph) parts.push(['录制 CUDA Graph', 10]);
      }
      const total = parts.reduce((a, [, t]) => a + t, 0);
      const Wd = 640;
      let x = 0;
      const colors = [W.color('ink-3'), W.color('prefill'), W.color('decode'), W.color('ok')];
      let svg = `<svg viewBox="0 0 ${Wd} 46" width="100%" role="img" aria-label="启动耗时的组成">`;
      parts.forEach(([n, t], i) => {
        const w = (t / total) * Wd;
        svg += `<rect x="${x}" y="6" width="${Math.max(1, w - 1)}" height="22" fill="${colors[i % colors.length]}" opacity="0.85"><title>${n}：${t.toFixed(1)} 秒</title></rect>`;
        x += w;
      });
      svg += '</svg>';
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-legend">${parts.map(([n, t], i) => `<span class="lw-legend-item"><i class="lw-dot" style="background:${colors[i % colors.length]}"></i>${n} ${t.toFixed(1)} 秒</span>`).join('')}</div>
        <div class="lw-stats"><div class="lw-stat ${total > 60 ? 'warn' : 'ok'}"><div class="lw-stat-label">合计</div><div class="lw-stat-value">${total.toFixed(0)} 秒</div></div></div>
        <div class="lw-callout">${st.sleep ? 'vLLM 的睡眠模式把权重暂存到 CPU 内存，释放显存给其他模型或训练任务，唤醒时只需要拷回权重，比重新启动快一个数量级。'
          : total > 60 ? '冷启动要一分钟以上。对延迟敏感的服务，不能等请求来了再加载模型：要么常驻，要么预测流量提前扩容，要么缩短各个环节（本地缓存权重、编译缓存、睡眠模式）。'
          : '各环节都已经比较快。注意 torch.compile 和 CUDA Graph 是一次性的开销，换来的是运行时更快的速度。'}</div>`;
    }
    $('[data-size]').addEventListener('input', (e) => { st.size = +e.target.value; update(); });
    $('[data-src]').addEventListener('change', (e) => { st.src = e.target.value; update(); });
    $('[data-c]').addEventListener('change', (e) => { st.compile = e.target.checked; update(); });
    $('[data-g]').addEventListener('change', (e) => { st.graph = e.target.checked; update(); });
    $('[data-sl]').addEventListener('change', (e) => { st.sleep = e.target.checked; update(); });
    update();
  });

  // ------------------------------------------------------------------ 4.6 缓存感知路由
  W.register('cache-router', function (container) {
    const N_INST = 4, N_REQ = 3000;
    const st = { tenants: 64, cap: 24, policy: 'cache', skew: 1.0 };
    const POL = { rr: '轮询', least: '最少请求', hash: '一致性哈希', cache: '缓存感知' };

    function run(policy) {
      const r = rng(7);
      const w = Array.from({ length: st.tenants }, (_, k) => 1 / (k + 1) ** st.skew);
      const tot = w.reduce((a, b) => a + b, 0);
      const caches = Array.from({ length: N_INST }, () => new Map());   // 租户 -> 最近使用时间（LRU）
      const load = new Array(N_INST).fill(0);                          // 最近一段时间收到的请求数（指数衰减）
      const count = new Array(N_INST).fill(0);
      let hits = 0, rrI = 0;
      const hashOf = (k) => { let h = 2166136261; for (const c of `t${k}`) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return (h >>> 0) % N_INST; };
      for (let i = 0; i < N_REQ; i++) {
        let u = r() * tot, t = 0;
        while (u > w[t] && t < st.tenants - 1) { u -= w[t]; t++; }
        let k;
        if (policy === 'rr') k = rrI++ % N_INST;
        else if (policy === 'least') k = load.indexOf(Math.min(...load));
        else if (policy === 'hash') k = hashOf(t);
        else {
          const lo = Math.min(...load), hi = Math.max(...load);
          const holders = caches.map((c, j) => (c.has(t) ? j : -1)).filter((j) => j >= 0);
          if (hi - lo > 6 && hi > 1.5 * lo) k = load.indexOf(lo);
          else if (holders.length) k = holders.reduce((a, b) => (load[b] < load[a] ? b : a));
          else k = load.indexOf(lo);
        }
        const c = caches[k];
        if (c.has(t)) hits++;
        c.delete(t); c.set(t, i);
        if (c.size > st.cap) c.delete(c.keys().next().value);
        for (let j = 0; j < N_INST; j++) load[j] *= 0.97;
        load[k] += 1;
        count[k]++;
      }
      return { hit: hits / N_REQ, count };
    }

    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">路由策略与前缀缓存</h4></div>
        <p class="lw-subtitle">4 个实例，每个最多缓存若干个租户的长前缀（LRU）。3000 个请求按 Zipf 分布来自各个租户。只统计缓存命中率和每个实例收到的请求数，不模拟时间；项目 4 的模拟器会给出延迟。</p>
        <div class="lw-tabs">${Object.entries(POL).map(([k, n]) => `<button type="button" class="lw-tab" data-p="${k}">${n}</button>`).join('')}</div>
        <div class="lw-controls">
          <div class="lw-control"><label><span>租户数</span><span class="lw-val" data-tv></span></label><input type="range" min="8" max="256" step="8" value="${st.tenants}" data-t /></div>
          <div class="lw-control"><label><span>每个实例能缓存的前缀数</span><span class="lw-val" data-cv></span></label><input type="range" min="4" max="128" step="4" value="${st.cap}" data-c /></div>
          <div class="lw-control"><label><span>租户流量不均</span><span class="lw-val" data-kv></span></label><input type="range" min="0" max="2" step="0.1" value="${st.skew}" data-k /></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      container.querySelectorAll('[data-p]').forEach((b) => b.classList.toggle('active', b.dataset.p === st.policy));
      $('[data-tv]').textContent = st.tenants;
      $('[data-cv]').textContent = st.cap;
      $('[data-kv]').textContent = st.skew.toFixed(1);
      const all = Object.keys(POL).map((p) => ({ p, ...run(p) }));
      const cur = all.find((x) => x.p === st.policy);
      const imb = Math.max(...cur.count) / (N_REQ / N_INST);
      $('[data-out]').innerHTML = `
        <div class="lw-stats">${all.map((x) => `<div class="lw-stat ${x.p === st.policy ? 'info' : ''}"><div class="lw-stat-label">${POL[x.p]}</div><div class="lw-stat-value">${(x.hit * 100).toFixed(0)}%</div><div class="lw-stat-sub">命中率；最忙 / 平均 ${(Math.max(...x.count) / (N_REQ / N_INST)).toFixed(2)}</div></div>`).join('')}</div>
        <div class="lw-callout">${POL[st.policy]}：各实例收到 ${cur.count.join('、')} 个请求，最忙的是平均的 ${imb.toFixed(2)} 倍。${st.tenants <= st.cap ? '每个实例都放得下所有租户的前缀，怎么路由都能命中，路由策略不重要。'
          : st.policy === 'hash' && imb > 1.3 ? '一致性哈希的命中率高，但热门租户固定落在同一个实例上，负载不均。'
          : st.policy === 'cache' ? '缓存感知路由在负载均衡时追随缓存，失衡时退回到按负载分配，兼顾了命中率和均衡。'
          : '轮询和最少请求不看前缀，同一个租户的请求被分散到所有实例，每个实例都要缓存所有租户，有效容量只有四分之一。'}</div>`;
    }
    container.querySelectorAll('[data-p]').forEach((b) => b.addEventListener('click', () => { st.policy = b.dataset.p; update(); }));
    $('[data-t]').addEventListener('input', (e) => { st.tenants = +e.target.value; update(); });
    $('[data-c]').addEventListener('input', (e) => { st.cap = +e.target.value; update(); });
    $('[data-k]').addEventListener('input', (e) => { st.skew = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 4.7 LoRA
  W.register('lora-cost', function (container) {
    if (!need(container)) return;
    const MD = W.MODELS;
    const st = { model: MD['qwen3-8b'] ? 'qwen3-8b' : Object.keys(MD)[0], rank: 16, all: true, budget: 8 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">一个 LoRA 适配器有多大</h4></div>
        <p class="lw-subtitle">LoRA 给每个目标线性层 $W$（形状 $d_{out} \\times d_{in}$）加上 $BA$，$A$ 是 $r \\times d_{in}$，$B$ 是 $d_{out} \\times r$，BF16 存储。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>基座模型</label>${sel('data-model', MD, st.model)}</div>
          <div class="lw-control"><label><span>秩 r</span><span class="lw-val" data-rv></span></label><input type="range" min="3" max="8" step="1" value="${Math.log2(st.rank)}" data-r /></div>
          <div class="lw-control"><label><span>留给适配器的显存</span><span class="lw-val" data-bv></span></label><input type="range" min="1" max="32" step="1" value="${st.budget}" data-b /></div>
        </div>
        <div class="lw-tabs">
          <button type="button" class="lw-tab" data-a="1">所有线性层</button>
          <button type="button" class="lw-tab" data-a="0">只有注意力的 q、k、v、o</button>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      container.querySelectorAll('[data-a]').forEach((b) => b.classList.toggle('active', (b.dataset.a === '1') === st.all));
      $('[data-rv]').textContent = st.rank;
      $('[data-bv]').textContent = `${st.budget} GB`;
      const m = MD[st.model];
      const q = m.nq * m.dh, kv = m.nkv * m.dh;
      const mods = [[m.h, q], [m.h, kv], [m.h, kv], [q, m.h]];
      if (st.all) mods.push([m.h, m.inter], [m.h, m.inter], [m.inter, m.h]);
      const loraParams = mods.reduce((a, [i, o]) => a + st.rank * (i + o), 0) * m.L;
      const baseParams = mods.reduce((a, [i, o]) => a + i * o, 0) * m.L;
      const bytes = loraParams * 2;
      const fit = Math.floor((st.budget * GB) / bytes);
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">一个适配器</div><div class="lw-stat-value">${(bytes / 2 ** 20).toFixed(0)} MiB</div><div class="lw-stat-sub">${(loraParams / 1e6).toFixed(1)}M 参数</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">${st.budget} GB 能放下</div><div class="lw-stat-value">${fit} 个</div></div>
          <div class="lw-stat"><div class="lw-stat-label">每个 token 增加的计算</div><div class="lw-stat-value">${((loraParams / baseParams) * 100).toFixed(2)}%</div><div class="lw-stat-sub">相对这些线性层本身</div></div>
        </div>
        <div class="lw-callout">适配器很小，计算量的增加也只有百分之一左右，难点在于同一个 batch 中的请求用的是不同的适配器：每个请求的 $x$ 要乘以它自己的 $BA$。Punica 的 SGMV kernel 把这些小矩阵乘法合并成一次调用，否则逐个适配器计算会产生大量的小 kernel（3.3 节）。</div>`;
    }
    $('[data-model]').addEventListener('change', (e) => { st.model = e.target.value; update(); });
    $('[data-r]').addEventListener('input', (e) => { st.rank = 2 ** +e.target.value; update(); });
    $('[data-b]').addEventListener('input', (e) => { st.budget = +e.target.value; update(); });
    container.querySelectorAll('[data-a]').forEach((b) => b.addEventListener('click', () => { st.all = b.dataset.a === '1'; update(); }));
    update();
  });
})();
