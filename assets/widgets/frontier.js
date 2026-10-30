/**
 * 第 5 部分的交互组件
 * - littles-law         5.1 并发上限、延迟与吞吐（Little 定律）
 * - scale-tradeoff      5.2 N 张卡：N 个副本还是一个 TP = N 的实例
 * - semantic-threshold  5.3 语义缓存的相似度阈值
 * - vlm-tokens          5.4 图片分辨率与视觉 token 数
 * - edge-fit            5.5 端侧设备能跑多大的模型、多快
 * - rl-rollout          5.6 RL 采样的长尾与同步、异步
 * - agent-turns         5.7 Agent 多轮调用中的前缀缓存
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
  const sel = (attr, obj, cur, label = (v) => v.name) => `<select ${attr}>${Object.keys(obj).map((k) => `<option value="${k}" ${k === cur ? 'selected' : ''}>${label(obj[k])}</option>`).join('')}</select>`;
  const slider = (label, attr, min, max, step, val) => `<div class="lw-control"><label><span>${label}</span><span class="lw-val" data-v-${attr}></span></label><input type="range" min="${min}" max="${max}" step="${step}" value="${val}" data-${attr} /></div>`;

  // ------------------------------------------------------------------ 5.1 Little 定律
  W.register('littles-law', function (container) {
    const st = { conc: 10, inLen: 223, outLen: 206, ttftA: 104, itlA: 43, ttftB: 59, itlB: 16 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">并发上限下的吞吐</h4></div>
        <p class="lw-subtitle">压测工具限制了同时在途的请求数（--max-concurrency）。到达速率足够高时，在途请求数始终等于这个上限，于是吞吐 = 并发 ÷ 每个请求的耗时（Little 定律）。默认值取自原书第 9 章：L40S 上的 Qwen3-14B 与 Qwen3-14B-AWQ，ShareGPT 负载，并发 10。AWQ 的 ITL 是按原书吞吐反推的估计值。</p>
        <div class="lw-controls">
          ${slider('并发上限', 'conc', 1, 64, 1, st.conc)}
          ${slider('平均输入 token', 'inLen', 50, 2000, 1, st.inLen)}
          ${slider('平均输出 token', 'outLen', 20, 2000, 1, st.outLen)}
          ${slider('配置 A 的 ITL（ms）', 'itlA', 5, 100, 1, st.itlA)}
          ${slider('配置 B 的 ITL（ms）', 'itlB', 5, 100, 1, st.itlB)}
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      ['conc', 'inLen', 'outLen', 'itlA', 'itlB'].forEach((k) => { $(`[data-v-${k}]`).textContent = st[k]; });
      const lat = (ttft, itl) => (ttft + itl * (st.outLen - 1)) / 1000;
      const a = lat(st.ttftA, st.itlA), b = lat(st.ttftB, st.itlB);
      const tp = (l) => (st.conc / l) * (st.inLen + st.outLen);
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">配置 A：每个请求耗时</div><div class="lw-stat-value">${a.toFixed(1)} s</div><div class="lw-stat-sub">总吞吐约 ${Math.round(tp(a))} token/s</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">配置 B：每个请求耗时</div><div class="lw-stat-value">${b.toFixed(1)} s</div><div class="lw-stat-sub">总吞吐约 ${Math.round(tp(b))} token/s</div></div>
          <div class="lw-stat"><div class="lw-stat-label">吞吐之比</div><div class="lw-stat-value">${(tp(b) / tp(a)).toFixed(2)} 倍</div><div class="lw-stat-sub">恰好等于延迟之比的倒数</div></div>
        </div>
        <div class="lw-callout">并发被固定时，吞吐的差别完全来自单个请求的延迟，而延迟主要由 ITL 决定。这时 KV Cache 能放多少 token 并不起作用：${st.conc} 个请求、每个约 ${st.inLen + st.outLen} token，只需要约 ${((st.conc * (st.inLen + st.outLen)) / 1000).toFixed(0)}K token 的 KV。要测出"更大的 KV Cache 带来更大的 batch"的效果，必须放开并发上限，或者提高并发直到 KV Cache 成为瓶颈。</div>`;
    }
    ['conc', 'inLen', 'outLen', 'itlA', 'itlB'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 5.2 横向还是纵向扩展
  W.register('scale-tradeoff', function (container) {
    if (!W.MODELS || !W.GPUS || !W.modelParams) { container.innerHTML = '<div class="lw-card">需要先加载 basics.js。</div>'; return; }
    const MD = W.MODELS, GP = W.GPUS;
    const st = { model: MD['qwen3-8b'] ? 'qwen3-8b' : Object.keys(MD)[0], gpu: GP.a100 ? 'a100' : Object.keys(GP)[0], n: 4, nvlink: true, conc: 64 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">N 张卡：N 个副本，还是一个 TP = N 的实例</h4></div>
        <p class="lw-subtitle">Roofline 估算 decode 的一步：每张卡读 1/TP 的权重，算 1/TP 的计算；TP 每层两次 all-reduce。总并发平均分给各个副本。只看 decode，不含 prefill。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>模型</label>${sel('data-model', MD, st.model)}</div>
          <div class="lw-control"><label>GPU</label>${sel('data-gpu', GP, st.gpu)}</div>
          <div class="lw-control"><label>卡数 N</label><select data-n>${[2, 4, 8].map((v) => `<option ${v === st.n ? 'selected' : ''}>${v}</option>`).join('')}</select></div>
          ${slider('总并发', 'conc', 1, 512, 1, st.conc)}
        </div>
        <div class="lw-tabs">
          <button type="button" class="lw-tab" data-l="1">卡间有 NVLink</button>
          <button type="button" class="lw-tab" data-l="0">只有 PCIe</button>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function step(m, g, tp, batch) {
      const P = W.modelParams(m).total;
      const bw = g.bw * 1e12 * 0.8, fl = g.bf16 * 1e12 * 0.6;
      const kv = batch * 1024 * 2 * m.L * m.nkv * m.dh * 2 / tp;
      const t = Math.max((P * 2) / tp / bw, (2 * P * batch) / tp / fl) + kv / bw;
      const link = st.nvlink ? 450e9 : 25e9, lat = st.nvlink ? 8e-6 : 15e-6;
      const ar = tp === 1 ? 0 : 2 * m.L * (((2 * (tp - 1)) / tp) * batch * m.h * 2 / link + lat);
      return t + ar + 1e-3;
    }
    function update() {
      container.querySelectorAll('[data-l]').forEach((b) => b.classList.toggle('active', (b.dataset.l === '1') === st.nvlink));
      $('[data-v-conc]').textContent = st.conc;
      const m = MD[st.model], g = GP[st.gpu];
      const fits1 = W.modelParams(m).total * 2 < g.mem * 1e9 * 0.85;
      const per = Math.max(1, Math.ceil(st.conc / st.n));
      const tR = step(m, g, 1, per), tT = step(m, g, st.n, st.conc);
      const thrR = (st.n * per) / tR, thrT = st.conc / tT;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat ${fits1 ? '' : 'warn'}"><div class="lw-stat-label">${st.n} 个单卡副本</div><div class="lw-stat-value">${fits1 ? `${(tR * 1e3).toFixed(1)} ms` : '放不下'}</div><div class="lw-stat-sub">${fits1 ? `ITL；总吞吐约 ${Math.round(thrR)} token/s` : '单卡放不下权重'}</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">一个 TP = ${st.n} 的实例</div><div class="lw-stat-value">${(tT * 1e3).toFixed(1)} ms</div><div class="lw-stat-sub">ITL；总吞吐约 ${Math.round(thrT)} token/s</div></div>
        </div>
        <div class="lw-callout">${!fits1 ? '单卡放不下，只能纵向扩展。'
          : thrR > thrT && tT < tR ? `副本的吞吐更高（${(thrR / thrT).toFixed(1)} 倍），TP 的延迟更低（ITL 为副本的 ${((tT / tR) * 100).toFixed(0)}%）。这就是原书第 9 章的结论：横向扩展换吞吐，纵向扩展换延迟。`
          : thrR > thrT ? '副本在吞吐和延迟上都更好：没有 NVLink 时，TP 的通信开销太大，原书在 L40S 上观察到的正是这种情况。'
          : '总并发相同时，每个副本只分到四分之一的请求，各自都受限于读权重，4 份权重各读一遍；TP 实例只读一份、做一份 batch 更大的计算，吞吐和延迟反而都更好。只有通信代价高（试试"只有 PCIe"）时，副本才占优。'}</div>`;
    }
    ['model', 'gpu'].forEach((k) => $(`[data-${k}]`).addEventListener('change', (e) => { st[k] = e.target.value; update(); }));
    $('[data-n]').addEventListener('change', (e) => { st.n = +e.target.value; update(); });
    $('[data-conc]').addEventListener('input', (e) => { st.conc = +e.target.value; update(); });
    container.querySelectorAll('[data-l]').forEach((b) => b.addEventListener('click', () => { st.nvlink = b.dataset.l === '1'; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 5.3 语义缓存的阈值
  W.register('semantic-threshold', function (container) {
    // 模拟 2000 个"新请求与缓存中最相似条目"的配对：一部分是同义改写（答案可以复用），
    // 一部分是看起来相似、但答案不同的请求（比如只差一个地名或数字）。
    const r = rng(5);
    const pairs = [];
    for (let i = 0; i < 2000; i++) {
      const kind = r();
      if (kind < 0.3) pairs.push({ same: true, sim: Math.min(0.995, 0.9 + gauss(r) * 0.035) });
      else if (kind < 0.55) pairs.push({ same: false, sim: Math.min(0.99, 0.86 + gauss(r) * 0.045) });   // 近似但不同
      else pairs.push({ same: false, sim: 0.55 + gauss(r) * 0.12 });
    }
    const st = { th: 0.9 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">语义缓存：阈值定在哪里</h4></div>
        <p class="lw-subtitle">2000 个新请求，各自与缓存中最相似的条目比较余弦相似度，超过阈值就直接返回缓存的答案。其中约 30% 确实是同一个问题的改写；约 25% 看起来很像，但答案不同，例如"北京到上海的航班多久"和"北京到广州的航班多久"。数据是模拟的。</p>
        <div class="lw-controls" style="grid-template-columns:1fr">${slider('相似度阈值', 'th', 0.7, 0.99, 0.01, st.th)}</div>
        <div data-plot></div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      $('[data-v-th]').textContent = st.th.toFixed(2);
      const bins = 60, lo = 0.2, hi = 1.0;
      const hs = new Array(bins).fill(0), hd = new Array(bins).fill(0);
      pairs.forEach((p) => { const k = Math.max(0, Math.min(bins - 1, Math.floor(((p.sim - lo) / (hi - lo)) * bins))); (p.same ? hs : hd)[k]++; });
      const mx = Math.max(...hs, ...hd);
      const Wd = 640, H = 150, L = 10, B = 120;
      const X = (v) => L + ((v - lo) / (hi - lo)) * (Wd - 2 * L);
      const bw = (Wd - 2 * L) / bins;
      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="相似度分布与阈值">`;
      for (let k = 0; k < bins; k++) {
        const x = L + k * bw;
        svg += `<rect x="${x}" y="${B - (hd[k] / mx) * 100}" width="${bw - 1}" height="${(hd[k] / mx) * 100}" fill="${W.color('err')}" opacity="0.55"/>`;
        svg += `<rect x="${x}" y="${B - (hs[k] / mx) * 100}" width="${bw - 1}" height="${(hs[k] / mx) * 100}" fill="${W.color('ok')}" opacity="0.7"/>`;
      }
      svg += `<line x1="${X(st.th)}" x2="${X(st.th)}" y1="10" y2="${B}" stroke="${W.color('ink')}" stroke-width="2"/>`;
      [0.2, 0.4, 0.6, 0.8, 1.0].forEach((v) => { svg += `<text x="${X(v)}" y="${B + 14}" font-size="10" fill="${W.color('ink-3')}" text-anchor="middle">${v.toFixed(1)}</text>`; });
      svg += `<text x="${Wd / 2}" y="${B + 28}" font-size="11" fill="${W.color('ink-3')}" text-anchor="middle">相似度（绿色：答案可复用；红色：答案不同）</text></svg>`;
      $('[data-plot]').innerHTML = svg;
      const hit = pairs.filter((p) => p.sim >= st.th);
      const good = hit.filter((p) => p.same).length, bad = hit.length - good;
      const reusable = pairs.filter((p) => p.same).length;
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">命中率</div><div class="lw-stat-value">${((hit.length / pairs.length) * 100).toFixed(1)}%</div><div class="lw-stat-sub">省下的 LLM 调用</div></div>
          <div class="lw-stat ${bad / Math.max(1, hit.length) > 0.05 ? 'warn' : 'ok'}"><div class="lw-stat-label">命中中答错的比例</div><div class="lw-stat-value">${((bad / Math.max(1, hit.length)) * 100).toFixed(1)}%</div><div class="lw-stat-sub">返回了别人的答案</div></div>
          <div class="lw-stat"><div class="lw-stat-label">可复用的请求被命中</div><div class="lw-stat-value">${((good / reusable) * 100).toFixed(0)}%</div></div>
        </div>
        <div class="lw-callout">两类请求的相似度分布有重叠，任何阈值都不能同时做到"命中多"和"不出错"。阈值低，省得多，但会把"到上海"的答案返回给问"到广州"的用户；阈值高，几乎不出错，但大部分改写也命中不了。这类错误对用户来说是直接的事实错误，所以实际中阈值通常设得很高，并且只对答案稳定、不依赖细节的问题启用语义缓存。</div>`;
    }
    $('[data-th]').addEventListener('input', (e) => { st.th = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 5.4 视觉 token
  W.register('vlm-tokens', function (container) {
    const st = { w: 1280, h: 720, n: 1, maxPix: 12845056 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">一张图片变成多少个 token</h4></div>
        <p class="lw-subtitle">按 Qwen2.5-VL 的规则：图片的长宽各自取整到最接近的 28 的倍数，每 14×14 像素是一个 patch，每 2×2 个 patch 合并成一个视觉 token，即每 28×28 像素一个 token。总像素数超过上限时等比例缩小。预填充耗时按 Qwen2.5-VL-7B（约 76 亿参数）、H100 峰值算力的 60% 估算，不含视觉编码器。</p>
        <div class="lw-controls">
          ${slider('宽（像素）', 'w', 224, 4096, 28, st.w)}
          ${slider('高（像素）', 'h', 224, 4096, 28, st.h)}
          ${slider('每个请求的图片数', 'n', 1, 16, 1, st.n)}
          <div class="lw-control"><label>像素上限</label><select data-max><option value="12845056" selected>默认（约 1280 万像素）</option><option value="1003520">约 100 万像素</option><option value="401408">约 40 万像素</option></select></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      ['w', 'h', 'n'].forEach((k) => { $(`[data-v-${k}]`).textContent = st[k]; });
      let w = st.w, h = st.h;
      if (w * h > st.maxPix) { const f = Math.sqrt(st.maxPix / (w * h)); w *= f; h *= f; }
      const W28 = Math.max(1, Math.round(w / 28)), H28 = Math.max(1, Math.round(h / 28));
      const tok = W28 * H28;
      const total = tok * st.n;
      const prefill = (2 * 7.6e9 * (total + 50)) / (989e12 * 0.6);
      $('[data-out]').innerHTML = `
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">实际处理的分辨率</div><div class="lw-stat-value">${W28 * 28}×${H28 * 28}</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">每张图片</div><div class="lw-stat-value">${tok} token</div></div>
          <div class="lw-stat"><div class="lw-stat-label">这个请求</div><div class="lw-stat-value">${total} token</div><div class="lw-stat-sub">LLM 部分预填充约 ${(prefill * 1e3).toFixed(0)} ms</div></div>
        </div>
        <div class="lw-callout">${total > 4000 ? '一个请求的视觉 token 已经相当于一篇长文档，TTFT 和 KV Cache 都主要由图片决定。限制像素上限（vLLM 中通过处理器参数设置）是控制成本最直接的手段，代价是细节的损失。' : '视觉 token 的数量与像素数成正比。分辨率加倍，token 数和预填充的计算量约变为 4 倍。'}原书的例子中，一张图片在提示词里占了 644 个占位 token。</div>`;
    }
    ['w', 'h', 'n'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    $('[data-max]').addEventListener('change', (e) => { st.maxPix = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 5.5 端侧
  const DEVICES = {
    phone: { name: '旗舰手机（约 8 GB 内存，约 60 GB/s）', mem: 8, bw: 60, usable: 0.4 },
    m4: { name: 'MacBook Air M4（16 GB，120 GB/s）', mem: 16, bw: 120, usable: 0.6 },
    m4max: { name: 'MacBook Pro M4 Max（128 GB，546 GB/s）', mem: 128, bw: 546, usable: 0.75 },
    orin: { name: 'Jetson AGX Orin（64 GB，204.8 GB/s）', mem: 64, bw: 204.8, usable: 0.75 },
    rtx4090: { name: 'RTX 4090 台式机（24 GB 显存，1008 GB/s）', mem: 24, bw: 1008, usable: 0.9 },
  };
  W.register('edge-fit', function (container) {
    const st = { dev: 'm4', bits: 4 };
    const SIZES = [0.6, 1.7, 4, 8, 14, 32, 70];
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">这台设备能跑多大的模型</h4></div>
        <p class="lw-subtitle">decode 时每生成一个 token 要把权重读一遍，速度上限约为"内存带宽 ÷ 权重大小"，按 70% 的带宽效率估算。能用的内存按设备类型打折（手机上系统和其他应用要占很大一部分）。不含 KV Cache。</p>
        <div class="lw-controls">
          <div class="lw-control"><label>设备</label>${sel('data-dev', DEVICES, st.dev)}</div>
          <div class="lw-control"><label>权重精度</label><select data-bits><option value="16">BF16</option><option value="8">8 位</option><option value="4" selected>4 位（如 GGUF Q4）</option></select></div>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      const d = DEVICES[st.dev];
      const rows = SIZES.map((b) => {
        const gb = (b * st.bits) / 8 * 1.07;   // 加上 scale 等开销
        const fits = gb < d.mem * d.usable;
        return { b, gb, fits, tps: (d.bw * 0.7) / gb };
      });
      $('[data-out]').innerHTML = `
        <div class="lw-table-wrap"><table class="lw-table">
          <thead><tr><th>参数量</th><th>权重</th><th>放得下吗</th><th>decode 速度上限</th></tr></thead>
          <tbody>${rows.map((x) => `<tr${x.fits ? '' : ' style="opacity:0.45"'}><td>${x.b}B</td><td>${x.gb.toFixed(1)} GB</td><td>${x.fits ? '是' : '否'}</td><td>${x.fits ? `约 ${x.tps.toFixed(0)} token/s` : '-'}</td></tr>`).join('')}</tbody>
        </table></div>
        <div class="lw-callout">端侧几乎总是 batch 为 1 的 decode，完全受限于内存带宽，所以 4 位量化同时决定了"放不放得下"和"有多快"。每秒 10 个 token 左右是交互可以接受的下限。prefill 则受限于算力，端侧芯片的算力有限，长输入的 TTFT 会比较明显。</div>`;
    }
    $('[data-dev]').addEventListener('change', (e) => { st.dev = e.target.value; update(); });
    $('[data-bits]').addEventListener('change', (e) => { st.bits = +e.target.value; update(); });
    update();
  });

  // ------------------------------------------------------------------ 5.6 RL 采样
  W.register('rl-rollout', function (container) {
    const st = { n: 64, tail: 1.2, seed: 3, async: false };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header">
          <h4 class="lw-title">RL 采样的长尾</h4>
          <div class="lw-btn-row" style="margin:0"><button type="button" class="lw-btn" data-re>重新采样</button></div>
        </div>
        <p class="lw-subtitle">一个训练步需要一批回答。回答长度服从长尾分布：大部分几百到几千 token，少数推理很长。同步模式下，这一批要等最长的那个回答生成完，才能开始训练。下图每一行是一个回答的生成时间（按 decode 速度恒定计）。</p>
        <div class="lw-controls">
          ${slider('每批回答数', 'n', 16, 256, 16, st.n)}
          ${slider('长度分布的长尾程度', 'tail', 0.5, 2, 0.1, st.tail)}
        </div>
        <div class="lw-tabs">
          <button type="button" class="lw-tab" data-a="0">同步：等全部完成</button>
          <button type="button" class="lw-tab" data-a="1">部分异步：完成 90% 就开始训练</button>
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      $('[data-v-n]').textContent = st.n;
      $('[data-v-tail]').textContent = st.tail.toFixed(1);
      container.querySelectorAll('[data-a]').forEach((b) => b.classList.toggle('active', (b.dataset.a === '1') === st.async));
      const r = rng(st.seed);
      const lens = Array.from({ length: st.n }, () => Math.min(32768, Math.round(800 * Math.exp(gauss(r) * st.tail))));
      const sorted = lens.slice().sort((a, b) => a - b);
      const cut = st.async ? sorted[Math.floor(st.n * 0.9) - 1] : sorted[st.n - 1];
      const busy = lens.reduce((a, l) => a + Math.min(l, cut), 0) / (st.n * cut);
      const Wd = 640, rowH = Math.max(1.5, Math.min(6, 300 / st.n)), H = st.n * rowH + 30;
      const X = (v) => 10 + (v / sorted[st.n - 1]) * (Wd - 20);
      let svg = `<svg viewBox="0 0 ${Wd} ${H}" width="100%" role="img" aria-label="每个回答的生成时间">`;
      sorted.forEach((l, i) => {
        svg += `<rect x="10" y="${i * rowH}" width="${X(Math.min(l, cut)) - 10}" height="${Math.max(1, rowH - 0.5)}" fill="${W.color('prefill')}" opacity="0.8"/>`;
        if (l > cut) svg += `<rect x="${X(cut)}" y="${i * rowH}" width="${X(l) - X(cut)}" height="${Math.max(1, rowH - 0.5)}" fill="${W.color('rule-strong')}" opacity="0.8"/>`;
      });
      svg += `<line x1="${X(cut)}" x2="${X(cut)}" y1="0" y2="${st.n * rowH}" stroke="${W.color('decode')}" stroke-width="2"/>`;
      svg += `<text x="${X(cut) - 4}" y="${st.n * rowH + 16}" font-size="11" fill="${W.color('decode')}" text-anchor="end">开始训练</text></svg>`;
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">长度中位数 / 最长</div><div class="lw-stat-value">${sorted[Math.floor(st.n / 2)]} / ${sorted[st.n - 1]}</div></div>
          <div class="lw-stat ${busy < 0.5 ? 'warn' : 'ok'}"><div class="lw-stat-label">采样阶段的有效利用率</div><div class="lw-stat-value">${(busy * 100).toFixed(0)}%</div><div class="lw-stat-sub">批次越到后面，同时在跑的回答越少</div></div>
        </div>
        <div class="lw-callout">${st.async ? '不等最长的那 10%，把它们留到下一轮继续生成（或者丢弃）。利用率明显提高，代价是这些回答会由更旧的策略生成，训练时要处理这种"离策略"的偏差。' : '同步模式下，最后一段时间只有少数几个回答还在生成，batch 很小，GPU 大部分算力闲置。'}</div>`;
    }
    $('[data-n]').addEventListener('input', (e) => { st.n = +e.target.value; update(); });
    $('[data-tail]').addEventListener('input', (e) => { st.tail = +e.target.value; update(); });
    $('[data-re]').addEventListener('click', () => { st.seed++; update(); });
    container.querySelectorAll('[data-a]').forEach((b) => b.addEventListener('click', () => { st.async = b.dataset.a === '1'; update(); }));
    update();
  });

  // ------------------------------------------------------------------ 5.7 Agent 多轮
  W.register('agent-turns', function (container) {
    const st = { sys: 6000, turns: 20, tool: 800, outp: 300 };
    container.innerHTML = `
      <div class="lw-card">
        <div class="lw-header"><h4 class="lw-title">一个 Agent 任务要算多少 token</h4></div>
        <p class="lw-subtitle">Agent 每一轮把全部历史（系统提示词、工具定义、之前的所有调用和结果）发给模型，生成下一个工具调用。每轮追加一次工具结果和一次模型输出。比较"每轮都完整 prefill"与"前缀缓存命中历史"两种情况。</p>
        <div class="lw-controls">
          ${slider('系统提示词与工具定义（token）', 'sys', 500, 30000, 500, st.sys)}
          ${slider('轮数', 'turns', 1, 100, 1, st.turns)}
          ${slider('每轮的工具结果（token）', 'tool', 50, 5000, 50, st.tool)}
          ${slider('每轮的模型输出（token）', 'outp', 20, 2000, 20, st.outp)}
        </div>
        <div data-out></div>
      </div>`;
    const $ = (s) => container.querySelector(s);
    function update() {
      ['sys', 'turns', 'tool', 'outp'].forEach((k) => { $(`[data-v-${k}]`).textContent = st[k]; });
      let ctx = st.sys, full = 0, cached = 0, outTot = 0;
      const per = [];
      for (let t = 0; t < st.turns; t++) {
        const newTok = t === 0 ? st.sys : st.tool;      // 第一轮是系统提示词，之后每轮新增工具结果
        if (t > 0) ctx += st.tool;
        full += ctx;
        cached += newTok;                                 // 上一轮输出的 KV 在 decode 时已经算好，也在缓存中
        outTot += st.outp;
        per.push(ctx);
        ctx += st.outp;
      }
      const Wd = 640, H = 120;
      const mx = Math.max(...per);
      const bw = (Wd - 20) / st.turns;
      let svg = `<svg viewBox="0 0 ${Wd} ${H + 20}" width="100%" role="img" aria-label="每一轮的上下文长度">`;
      per.forEach((c, i) => { svg += `<rect x="${10 + i * bw}" y="${H - (c / mx) * (H - 10)}" width="${Math.max(1, bw - 1)}" height="${(c / mx) * (H - 10)}" fill="${W.color('prefill')}" opacity="0.8"><title>第 ${i + 1} 轮：上下文 ${c} token</title></rect>`; });
      svg += `<text x="${Wd / 2}" y="${H + 16}" font-size="11" fill="${W.color('ink-3')}" text-anchor="middle">每一轮的上下文长度（最后一轮 ${per[per.length - 1]} token）</text></svg>`;
      $('[data-out]').innerHTML = `${svg}
        <div class="lw-stats">
          <div class="lw-stat"><div class="lw-stat-label">没有前缀缓存时 prefill 的 token</div><div class="lw-stat-value">${(full / 1000).toFixed(0)}K</div></div>
          <div class="lw-stat info"><div class="lw-stat-label">历史全部命中缓存时</div><div class="lw-stat-value">${(cached / 1000).toFixed(1)}K</div><div class="lw-stat-sub">少算 ${(100 * (1 - cached / full)).toFixed(0)}%</div></div>
          <div class="lw-stat"><div class="lw-stat-label">输入与输出之比</div><div class="lw-stat-value">${Math.round(full / outTot)} : 1</div><div class="lw-stat-sub">按每轮的完整输入计</div></div>
        </div>
        <div class="lw-callout">没有缓存时，总的 prefill 量随轮数平方增长。Agent 负载的输入远多于输出，而且每一轮的前缀都是上一轮的完整内容，正是前缀缓存最理想的场景。但命中有前提：同一个会话的请求要落到同一个实例上（4.6 节），两轮之间的间隔里缓存不能被淘汰，而且历史不能被改写。</div>`;
    }
    ['sys', 'turns', 'tool', 'outp'].forEach((k) => $(`[data-${k}]`).addEventListener('input', (e) => { st[k] = +e.target.value; update(); }));
    update();
  });
})();
