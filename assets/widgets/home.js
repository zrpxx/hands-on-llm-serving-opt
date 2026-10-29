/**
 * 主页组件
 * - kv-pool：分页 KV 块池的动画。每个方格是一个 KV 块；请求到达时 prefill 一次写入若干块（钴蓝），
 *   decode 逐块追加（琥珀），请求结束后释放。物理块从空闲列表中取出，所以同一个请求的块并不连续。
 * - course-map：读取 _sidebar.md 生成课程地图，并结合本地学习进度给出"继续学习"的位置。
 */
(function () {
  const W = window.LLMWidgets;
  if (!W) return;

  // ------------------------------------------------------------------ KV 块池
  W.register('kv-pool', function (container) {
    const COLS = 16;
    const ROWS = 8;
    const N = COLS * ROWS;
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    container.innerHTML = `
      <div class="kv-pool" role="img" aria-label="分页 KV Cache 块池示意：请求的 prefill 一次写入多个块，decode 逐块追加，请求结束后块被释放">
        <div class="kv-pool-grid">${'<i class="kv-cell"></i>'.repeat(N)}</div>
        <div class="kv-pool-caption">
          <span><i class="lw-dot" style="background:var(--prefill)"></i>prefill 写入</span>
          <span><i class="lw-dot" style="background:var(--decode)"></i>decode 追加</span>
          <span class="kv-pool-stat" data-stat></span>
        </div>
      </div>`;
    const cells = Array.from(container.querySelectorAll('.kv-cell'));
    const stat = container.querySelector('[data-stat]');

    let seed = 7;
    const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    const free = [];
    for (let i = 0; i < N; i++) free.push(i);
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [free[i], free[j]] = [free[j], free[i]]; }
    const owner = new Array(N).fill(null);
    const reqs = [];
    let tick = 0;

    function paint() {
      cells.forEach((c, i) => {
        const o = owner[i];
        c.className = 'kv-cell' + (o ? (o.kind === 'p' ? ' is-prefill' : ' is-decode') : '');
      });
      const used = N - free.length;
      stat.textContent = `${reqs.length} 个请求，占用 ${used} / ${N} 块`;
    }

    function step() {
      tick += 1;
      // 新请求到达：prefill 一次分配若干块
      if ((tick % 3 === 1 || reqs.length < 3) && reqs.length < 9) {
        const need = 2 + Math.floor(rand() * 6);
        if (free.length > need + 6) {
          const r = { blocks: [], target: need + 3 + Math.floor(rand() * 10) };
          for (let k = 0; k < need; k++) { const b = free.shift(); owner[b] = { r, kind: 'p' }; r.blocks.push(b); }
          reqs.push(r);
        }
      }
      // 运行中的请求 decode：每次追加一块（一块是 16 个 token，这里压缩了时间）
      reqs.forEach((r) => {
        if (rand() < 0.55 && free.length && r.blocks.length < r.target) {
          const b = free.shift(); owner[b] = { r, kind: 'd' }; r.blocks.push(b);
        }
      });
      // 生成完的请求释放全部块，块回到空闲列表尾部
      for (let i = reqs.length - 1; i >= 0; i--) {
        const r = reqs[i];
        if (r.blocks.length >= r.target || (free.length < 2 && i === reqs.length - 1)) {
          r.blocks.forEach((b) => { owner[b] = null; free.push(b); });
          reqs.splice(i, 1);
        }
      }
      paint();
    }

    if (reduce) {
      for (let i = 0; i < 24; i++) step();
      return;
    }
    for (let i = 0; i < 6; i++) step();
    let timer = null;
    let visible = true;
    const gen = container.dataset.gen;
    const start = () => { if (!timer) timer = setInterval(() => {
      if (!document.body.contains(container) || container.dataset.gen !== gen) { clearInterval(timer); return; }
      if (visible && !document.hidden) step();
    }, 650); };
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(container);
    }
    start();
  });

  // ------------------------------------------------------------------ 课程地图
  const PART_INFO = {
    0: { desc: '模型怎样逐个生成 token，KV Cache 为什么必要；再从 GPU 规格出发，估算显存、判断瓶颈，定义衡量服务的指标。', written: true },
    1: { desc: '把模型变成在线服务：异步流式接口、连续批处理、分页 KV Cache、前缀缓存，以及主流框架的取舍。', written: true },
    2: { desc: '量化的原理与 GPTQ、AWQ、SmoothQuant、FP8、KV Cache 量化，低比特算子，以及怎样选择、评估方案；蒸馏与剪枝。', written: true },
    3: { desc: '注意力变体、FlashAttention、用 Triton 写算子、投机解码。', written: false },
    4: { desc: '张量 / 流水线 / 专家并行，Prefill-Decode 分离，KV 分层存储与缓存感知路由。', written: false },
    5: { desc: '书的最后两章：端到端优化流程、语义缓存、多模态、端侧与强化学习中的推理。', written: false },
    6: { desc: '对照 DeepInfer、NVFP4、Wide EP、Attention-FFN 分离等 2026 年的生产实践。', written: false },
  };

  function parseSidebar(md) {
    const parts = [];
    let cur = null;
    md.split('\n').forEach((line) => {
      const head = line.match(/^\*\s+\*\*第\s*(\d+)\s*部分[\s　]*(.+?)\*\*/);
      if (head) {
        cur = { num: Number(head[1]), title: head[2].replace(/（书外）/, '').trim(), pages: [] };
        parts.push(cur);
        return;
      }
      const link = line.match(/^\s+\*\s+\[(.+?)\]\((.+?)\.md\)/);
      if (link && cur) cur.pages.push({ title: link[1], route: link[2] });
      if (/^---/.test(line) && cur && cur.num === 6) cur = null;
    });
    return parts;
  }

  W.register('course-map', function (container) {
    container.innerHTML = '<div class="lw-subtitle">正在读取目录…</div>';
    fetch('_sidebar.md')
      .then((r) => r.text())
      .then((md) => {
        const parts = parseSidebar(md);
        const prog = (window.LLMProgress && window.LLMProgress.load()) || {};
        let next = null;
        parts.forEach((p) => p.pages.forEach((pg) => {
          if (!next && PART_INFO[p.num] && PART_INFO[p.num].written && !prog[pg.route]) next = pg;
        }));
        const anyDone = Object.keys(prog).length > 0;
        container.innerHTML = `
          <div class="course-map">
            ${parts.map((p) => {
              const done = p.pages.filter((pg) => prog[pg.route]).length;
              const info = PART_INFO[p.num] || {};
              const isDone = done === p.pages.length && p.pages.length > 0;
              const isCurrent = next && p.pages.includes(next);
              const projects = p.pages.filter((pg) => /boss/.test(pg.route));
              return `
                <div class="cm-part ${isDone ? 'is-done' : ''} ${isCurrent ? 'is-current' : ''}">
                  <div class="cm-num">${p.num}</div>
                  <div class="cm-title"><a href="#/${p.pages[0].route}">${W.escapeHtml(p.title)}</a></div>
                  <div class="cm-meta">${info.written ? `${done} / ${p.pages.length} 节` : '提纲，待编写'}</div>
                  <div class="cm-desc">${info.desc || ''}
                    ${projects.length ? `<div class="cm-project">${projects.map((pg) => `<a href="#/${pg.route}">${W.escapeHtml(pg.title.replace(/　/g, ' '))}</a>`).join('，')}</div>` : ''}
                  </div>
                  ${info.written ? `<div class="cm-bar">${p.pages.map((pg) => `<span class="${prog[pg.route] ? 'is-done' : ''}" title="${W.escapeHtml(pg.title)}"></span>`).join('')}</div>` : ''}
                </div>`;
            }).join('')}
          </div>
          ${next ? `<p class="cm-resume">${anyDone ? '上次学到这里：' : '从这里开始：'}<a href="#/${next.route}">${W.escapeHtml(next.title)}</a></p>` : ''}`;
        const resume = document.querySelector('[data-resume]');
        if (resume && next) {
          resume.setAttribute('href', '#/' + next.route);
          resume.textContent = anyDone ? '继续：' + next.title : '从 0.1 开始';
        }
      })
      .catch(() => {
        container.innerHTML = '<div class="lw-error">目录读取失败。请通过本地服务器（docsify serve 或 python -m http.server）打开站点。</div>';
      });
  });
})();
