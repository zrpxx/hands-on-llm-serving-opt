/**
 * 主页组件
 * - kv-pool：分页 KV 块池的实时示意与交互检视。每个方格代表一个物理 KV 块；
 *   请求到达时 prefill 一次分配若干块（钴蓝），decode 逐块追加（琥珀），完成后整批释放回空闲队列。
 *   支持悬停高亮同一序列的全部非连续物理块，以及暂停/单步检视。
 * - course-map：读取 _sidebar.md 生成全书课程地图与章节索引，结合本地学习进度提供续读入口。
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
      <div class="kv-pool" role="region" aria-label="分页 KV Cache 块池实时示意">
        <div class="kv-pool-head">
          <div class="kv-pool-title">
            <span class="kv-pool-fig">图 0.1</span>
            <span>分页 KV Cache 块池</span>
          </div>
          <div class="kv-pool-ctrls">
            <button type="button" class="kv-pool-btn" data-toggle>${reduce ? '继续' : '暂停'}</button>
            <button type="button" class="kv-pool-btn" data-step>单步</button>
          </div>
        </div>
        <div class="kv-pool-grid">${Array.from({ length: N }, (_, i) => `<i class="kv-cell" data-idx="${i}"></i>`).join('')}</div>
        <div class="kv-pool-inspect" data-inspect>悬停任一色块，查看同一请求在物理池中的非连续块表</div>
        <div class="kv-pool-caption">
          <div class="kv-pool-legend">
            <span><i class="lw-dot" style="background:var(--prefill)"></i>prefill 写入</span>
            <span><i class="lw-dot" style="background:var(--decode)"></i>decode 追加</span>
            <span><i class="lw-dot" style="background:var(--paper-3)"></i>空闲</span>
          </div>
          <span class="kv-pool-stat" data-stat></span>
        </div>
      </div>`;

    const gridEl = container.querySelector('.kv-pool-grid');
    const cells = Array.from(container.querySelectorAll('.kv-cell'));
    const stat = container.querySelector('[data-stat]');
    const inspectEl = container.querySelector('[data-inspect]');
    const toggleBtn = container.querySelector('[data-toggle]');
    const stepBtn = container.querySelector('[data-step]');

    let seed = 7;
    const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    const free = [];
    for (let i = 0; i < N; i++) free.push(i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [free[i], free[j]] = [free[j], free[i]];
    }
    const owner = new Array(N).fill(null);
    const reqs = [];
    let tick = 0;
    let seqCounter = 0;
    let hoveredIdx = null;
    let paused = !!reduce;

    function updateInspect() {
      if (hoveredIdx === null || !owner[hoveredIdx]) {
        gridEl.classList.remove('has-hover');
        cells.forEach((c) => c.classList.remove('is-hi'));
        inspectEl.textContent = '悬停任一色块，查看同一请求在物理池中的非连续块表';
        return;
      }
      const targetReq = owner[hoveredIdx].r;
      gridEl.classList.add('has-hover');
      cells.forEach((c, i) => {
        c.classList.toggle('is-hi', !!(owner[i] && owner[i].r === targetReq));
      });
      const logIdx = targetReq.blocks.indexOf(hoveredIdx);
      const preview = targetReq.blocks.slice(0, 8).join(', ') + (targetReq.blocks.length > 8 ? ', …' : '');
      inspectEl.innerHTML =
        `请求 <strong>#${targetReq.id}</strong> · 逻辑块 ${logIdx} → 物理块 ${hoveredIdx} · ` +
        `块表 [${preview}]（共 ${targetReq.blocks.length} 块，预分配 0 碎片）`;
    }

    function paint() {
      cells.forEach((c, i) => {
        const o = owner[i];
        c.className = 'kv-cell' + (o ? (o.kind === 'p' ? ' is-prefill' : ' is-decode') : '');
      });
      const used = N - free.length;
      const pct = Math.round((used / N) * 100);
      stat.textContent = `${reqs.length} 在途 · ${used} / ${N} 块 (${pct}%)`;
      if (hoveredIdx !== null) updateInspect();
    }

    function step() {
      tick += 1;
      if ((tick % 3 === 1 || reqs.length < 3) && reqs.length < 9) {
        const need = 2 + Math.floor(rand() * 6);
        if (free.length > need + 6) {
          seqCounter += 1;
          const r = { id: seqCounter, blocks: [], target: need + 3 + Math.floor(rand() * 10) };
          for (let k = 0; k < need; k++) {
            const b = free.shift();
            owner[b] = { r, kind: 'p' };
            r.blocks.push(b);
          }
          reqs.push(r);
        }
      }
      reqs.forEach((r) => {
        if (rand() < 0.55 && free.length && r.blocks.length < r.target) {
          const b = free.shift();
          owner[b] = { r, kind: 'd' };
          r.blocks.push(b);
        }
      });
      for (let i = reqs.length - 1; i >= 0; i--) {
        const r = reqs[i];
        if (r.blocks.length >= r.target || (free.length < 2 && i === reqs.length - 1)) {
          r.blocks.forEach((b) => {
            owner[b] = null;
            free.push(b);
          });
          reqs.splice(i, 1);
        }
      }
      paint();
    }

    gridEl.addEventListener('mouseover', (e) => {
      const cell = e.target.closest('.kv-cell');
      if (!cell) return;
      hoveredIdx = Number(cell.dataset.idx);
      updateInspect();
    });
    gridEl.addEventListener('mouseleave', () => {
      hoveredIdx = null;
      updateInspect();
    });

    toggleBtn.addEventListener('click', () => {
      paused = !paused;
      toggleBtn.textContent = paused ? '继续' : '暂停';
    });
    stepBtn.addEventListener('click', () => {
      paused = true;
      toggleBtn.textContent = '继续';
      step();
    });

    for (let i = 0; i < (reduce ? 24 : 6); i++) step();
    if (reduce) return;

    let visible = true;
    const gen = container.dataset.gen;
    const timer = setInterval(() => {
      if (!document.body.contains(container) || container.dataset.gen !== gen) {
        clearInterval(timer);
        return;
      }
      if (!paused && visible && !document.hidden && hoveredIdx === null) step();
    }, 680);
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((es) => {
        visible = es[0].isIntersecting;
      }).observe(container);
    }
  });

  // ------------------------------------------------------------------ 课程地图
  const PART_INFO = {
    0: {
      desc: '模型怎样逐个生成 token，KV Cache 为什么必要；再从 GPU 规格出发，估算显存、判断瓶颈，定义衡量服务的指标。',
      written: true,
    },
    1: {
      desc: '把模型变成在线服务：异步流式接口、连续批处理、分页 KV Cache、前缀缓存，以及主流框架的取舍。',
      written: true,
    },
    2: {
      desc: '量化的原理与 GPTQ、AWQ、SmoothQuant、FP8、KV Cache 量化，低比特算子，以及怎样选择、评估方案；蒸馏与剪枝。',
      written: true,
    },
    3: {
      desc: '注意力变体与 MLA、FlashAttention 与在线 softmax、算子融合与 Triton、投机解码的原理与调参、性能剖析。',
      written: true,
    },
    4: {
      desc: '数据、张量、流水线与专家并行，Prefill-Decode 分离，KV 分层存储，多模型与 Multi-LoRA，缓存感知路由，企业级架构。',
      written: true,
    },
    5: {
      desc: '端到端优化流程与取舍、语义缓存与语义路由、多模态、端侧、强化学习与 Agent 场景下的推理服务。',
      written: true,
    },
    6: {
      desc: '以 DeepInfer 为线索：NVFP4 与 MXFP4、Wide EP 与 EPLB、Attention-FFN 分离、MTP、长上下文、批不变推理、异步调度、算子 DSL。',
      written: true,
    },
  };

  function parseSidebar(md) {
    const parts = [];
    let cur = null;
    md.split('\n').forEach((line) => {
      const head = line.match(/^\*\s+\*\*第\s*(\d+)\s*部分[\s　]*(.+?)\*\*/);
      if (head) {
        cur = { num: Number(head[1]), title: head[2].replace(/（书外）/, '').trim(), isExtra: /书外/.test(head[2]), pages: [] };
        parts.push(cur);
        return;
      }
      const link = line.match(/^\s+\*\s+\[(.+?)\]\((.+?)\.md\)/);
      if (link && cur) cur.pages.push({ title: link[1], route: link[2] });
      if (/^---/.test(line) && cur && cur.num === 6) cur = null;
    });
    return parts;
  }

  function splitChapterTitle(raw) {
    const clean = raw.replace(/　/g, ' ').trim();
    const m = clean.match(/^(\d+\.\d+)\s+(.+)$/);
    if (m) return { code: m[1], name: m[2], isProject: false };
    const p = clean.match(/^(项目\s*\d+(?:\s*分步指南)?|毕业项目)\s*(.*)$/);
    if (p) return { code: p[1], name: p[2] || p[1], isProject: true };
    return { code: '', name: clean, isProject: false };
  }

  W.register('course-map', function (container) {
    container.innerHTML = '<div class="lw-subtitle">正在读取目录…</div>';
    let expandAll = false;
    const manualOpen = new Set();

    fetch('_sidebar.md')
      .then((r) => r.text())
      .then((md) => {
        const parts = parseSidebar(md);

        function renderMap() {
          const prog = (window.LLMProgress && window.LLMProgress.load()) || {};
          let next = null;
          let currentPartNum = 0;
          let totalPages = 0;
          let totalDone = 0;
          parts.forEach((p) =>
            p.pages.forEach((pg) => {
              totalPages += 1;
              if (prog[pg.route]) totalDone += 1;
              if (!next && PART_INFO[p.num] && PART_INFO[p.num].written && !prog[pg.route]) {
                next = pg;
                currentPartNum = p.num;
              }
            })
          );
          if (manualOpen.size === 0) manualOpen.add(currentPartNum);
          const anyDone = totalDone > 0;
          const totalPct = totalPages ? Math.round((totalDone / totalPages) * 100) : 0;

          container.innerHTML = `
            <div class="cm-toolbar">
              <div class="cm-summary">
                <span class="cm-summary-stat">已读 <strong>${totalDone}</strong> / ${totalPages} 节（${totalPct}%）</span>
                ${
                  next
                    ? `<span class="cm-summary-next">${anyDone ? '续读：' : '起点：'}<a href="#/${next.route}">${W.escapeHtml(next.title.replace(/　/g, ' '))}</a></span>`
                    : `<span class="cm-summary-next">已读完全部章节</span>`
                }
              </div>
              <button type="button" class="cm-toggle-btn" data-cm-toggle>${expandAll ? '只展开当前部分' : '展开全部小节'}</button>
            </div>
            <div class="course-map">
              ${parts
                .map((p) => {
                  const done = p.pages.filter((pg) => prog[pg.route]).length;
                  const info = PART_INFO[p.num] || {};
                  const isDone = done === p.pages.length && p.pages.length > 0;
                  const isCurrent = next && p.pages.includes(next);
                  const isOpen = expandAll || manualOpen.has(p.num);
                  const numStr = String(p.num).padStart(2, '0');
                  return `
                    <div class="cm-part ${isDone ? 'is-done' : ''} ${isCurrent ? 'is-current' : ''} ${isOpen ? 'is-open' : ''}">
                      <div class="cm-num-col">
                        <div class="cm-num">${numStr}</div>
                      </div>
                      <div class="cm-body">
                        <div class="cm-head-row">
                          <div class="cm-title">
                            <a href="#/${p.pages[0].route}">${W.escapeHtml(p.title)}</a>
                            ${p.isExtra ? '<span class="cm-extra-badge">书外扩展</span>' : ''}
                          </div>
                          <div class="cm-head-right">
                            <span class="cm-meta">${done} / ${p.pages.length} 节</span>
                            <button type="button" class="cm-part-btn" data-part-toggle="${p.num}">${isOpen ? '收起' : '小节'}</button>
                          </div>
                        </div>
                        <div class="cm-desc">${info.desc || ''}</div>
                        <div class="cm-bar">${p.pages
                          .map(
                            (pg) =>
                              `<a href="#/${pg.route}" class="cm-bar-seg ${prog[pg.route] ? 'is-done' : ''}" title="${W.escapeHtml(pg.title.replace(/　/g, ' '))}" aria-label="${W.escapeHtml(pg.title)}"></a>`
                          )
                          .join('')}</div>
                        ${
                          isOpen
                            ? `<div class="cm-chapters">${p.pages
                                .map((pg) => {
                                  const sp = splitChapterTitle(pg.title);
                                  const cleared = !!prog[pg.route];
                                  return `<a href="#/${pg.route}" class="cm-chap ${cleared ? 'is-done' : ''} ${sp.isProject ? 'is-proj' : ''}">
                                    <span class="cm-chap-code">${W.escapeHtml(sp.code)}</span>
                                    <span class="cm-chap-name">${W.escapeHtml(sp.name !== sp.code ? sp.name : '')}</span>
                                  </a>`;
                                })
                                .join('')}</div>`
                            : ''
                        }
                      </div>
                    </div>`;
                })
                .join('')}
            </div>`;

          const toggle = container.querySelector('[data-cm-toggle]');
          if (toggle) {
            toggle.addEventListener('click', () => {
              expandAll = !expandAll;
              if (!expandAll) {
                manualOpen.clear();
                manualOpen.add(currentPartNum);
              }
              renderMap();
            });
          }

          container.querySelectorAll('[data-part-toggle]').forEach((btn) => {
            btn.addEventListener('click', () => {
              const n = Number(btn.dataset.partToggle);
              if (expandAll) {
                expandAll = false;
                manualOpen.clear();
                parts.forEach((p) => {
                  if (p.num !== n) manualOpen.add(p.num);
                });
              } else if (manualOpen.has(n)) {
                manualOpen.delete(n);
              } else {
                manualOpen.add(n);
              }
              renderMap();
            });
          });

          const resume = document.querySelector('[data-resume]');
          if (resume && next) {
            resume.setAttribute('href', '#/' + next.route);
            resume.textContent = anyDone ? '继续：' + next.title.replace(/　/g, ' ') : '从 0.1 开始';
          }
        }

        renderMap();
      })
      .catch(() => {
        container.innerHTML = '<div class="lw-error">目录读取失败。请通过本地服务器（docsify serve 或 python -m http.server）打开站点。</div>';
      });
  });
})();
