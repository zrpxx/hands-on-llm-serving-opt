/**
 * 交互组件的公共部分。
 *
 * 用法：在 Markdown 中写 <div class="llm-widget" data-widget="名称" ...></div>，
 * 页面渲染完成后由 index.html 调用 LLMWidgets.mountAll() 挂载。
 *
 * 组件内的文本可以写 $...$ 公式，挂载和每次重绘后都会自动用 KaTeX 渲染。
 */
(function () {
  const registry = {};

  function register(name, renderFn) {
    registry[name] = renderFn;
  }

  function formatNum(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(n)) return '-';
    if (Math.abs(n) >= 1000) return Number(n.toFixed(digits)).toLocaleString('en-US');
    return Number(n.toFixed(digits)).toString();
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /** 读取 CSS 设计令牌，例如 color('prefill') -> '#2d50c4' */
  function color(name) {
    return getComputedStyle(document.documentElement).getPropertyValue('--' + name).trim();
  }

  // ------------------------------------------------------------------ 组件内公式
  const MATH_RE = /\$([^$\n]+?)\$/g;

  function renderMath(root) {
    if (!window.katex) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || node.nodeValue.indexOf('$') === -1) return NodeFilter.FILTER_REJECT;
        const p = node.parentNode;
        if (!p || p.closest('code, pre, .katex, script, style, textarea, input')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      const text = node.nodeValue;
      MATH_RE.lastIndex = 0;
      if (!MATH_RE.test(text)) return;
      MATH_RE.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0;
      let m;
      while ((m = MATH_RE.exec(text))) {
        if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const span = document.createElement('span');
        try {
          window.katex.render(m[1], span, { throwOnError: false });
        } catch (e) {
          span.textContent = m[0];
        }
        frag.appendChild(span);
        last = m.index + m[0].length;
      }
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  function watchMath(node) {
    renderMath(node);
    let pending = false;
    const obs = new MutationObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        obs.disconnect();
        renderMath(node);
        obs.observe(node, { childList: true, subtree: true, characterData: true });
      });
    });
    obs.observe(node, { childList: true, subtree: true, characterData: true });
  }

  let retried = false;
  function loadMissing() {
    if (retried) return;
    retried = true;
    const files = ['basics.js', 'serving.js', 'compression.js', 'kernels.js', 'cluster.js', 'frontier.js', 'production.js'];
    let left = files.length;
    files.forEach((f) => {
      if (document.querySelector(`script[src^="assets/widgets/${f}"]`)) { if (--left === 0) remountAll(); return; }
      const s = document.createElement('script');
      s.src = `assets/widgets/${f}?v=${Date.now()}`;
      s.onload = s.onerror = () => { if (--left === 0) remountAll(); };
      document.body.appendChild(s);
    });
  }

  function mountAll() {
    document.querySelectorAll('.llm-widget[data-widget]').forEach((node) => {
      if (node.dataset.mounted === 'true') return;
      const name = node.dataset.widget;
      const renderFn = registry[name];
      if (!renderFn) {
        // 常见原因：浏览器用了缓存里的旧 index.html，没有引用新的组件脚本。尝试补加载一次。
        loadMissing();
        node.innerHTML = `<div class="lw-error">组件 <code>${escapeHtml(name)}</code> 没有加载。如果刷新后仍然出现，请强制刷新页面（macOS：Cmd + Shift + R；Windows：Ctrl + F5）以清除缓存。</div>`;
        return;
      }
      try {
        node.dataset.mounted = 'true';
        renderFn(node);
        watchMath(node);
      } catch (err) {
        console.error('[LLMWidgets]', name, err);
        node.innerHTML = `<div class="lw-error">组件 ${escapeHtml(name)} 渲染出错：${escapeHtml(err.message)}</div>`;
      }
    });
  }

  // ------------------------------------------------------------------ 自测题
  // <div class="llm-widget" data-widget="quiz" data-title="..." data-questions='[{"q","options","answer","explain"}]'></div>
  // 选项按题目文本做确定性打乱：同一道题每次显示的顺序相同，但正确答案不会总在同一个位置。
  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function shuffledOrder(n, seed) {
    const order = Array.from({ length: n }, (_, i) => i);
    let s = seed || 1;
    for (let i = n - 1; i > 0; i--) {
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      const j = s % (i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    return order;
  }

  register('quiz', function (container) {
    let questions = [];
    try {
      questions = JSON.parse(container.getAttribute('data-questions') || '[]');
    } catch (e) {
      container.innerHTML = '<div class="lw-error">自测题数据格式有误：data-questions 不是合法 JSON。题目中的 $ 需要写成 &amp;#36;，否则会被公式插件提前替换。</div>';
      return;
    }
    const title = container.getAttribute('data-title') || '自测';
    const orders = questions.map((q) => shuffledOrder(q.options.length, hash(q.q)));
    const picked = {};

    function render() {
      const correct = questions.filter((q, i) => picked[i] === q.answer).length;
      const answered = Object.keys(picked).length;
      container.innerHTML = `
        <div class="lw-card lw-quiz-card">
          <div class="lw-header">
            <h4 class="lw-title">${escapeHtml(title)}</h4>
            <div class="lw-quiz-score">${answered ? `答对 <strong>${correct}</strong> / ${questions.length}` : `共 ${questions.length} 题`}</div>
          </div>
          <div class="lw-quiz-list">
            ${questions.map((q, qi) => {
              const has = picked[qi] !== undefined;
              const right = picked[qi] === q.answer;
              return `
                <div class="lw-quiz-item">
                  <div class="lw-quiz-q"><strong>${qi + 1}.</strong>${q.q}</div>
                  <div class="lw-quiz-opts" role="group">
                    ${orders[qi].map((oi, pos) => {
                      let cls = 'lw-quiz-opt';
                      if (has) {
                        if (oi === q.answer) cls += ' opt-correct';
                        else if (oi === picked[qi]) cls += ' opt-wrong';
                        else cls += ' opt-dim';
                      }
                      return `<button type="button" class="${cls}" data-q="${qi}" data-o="${oi}" ${has ? 'disabled' : ''}>
                        <span class="opt-letter">${String.fromCharCode(65 + pos)}</span><span class="opt-text">${q.options[oi]}</span>
                      </button>`;
                    }).join('')}
                  </div>
                  ${has ? `
                    <div class="lw-quiz-explain ${right ? 'exp-ok' : 'exp-err'}">
                      <strong>${right ? '正确。' : '不对。'}</strong>${q.explain}
                    </div>
                    ${right ? '' : `<button type="button" class="lw-quiz-retry" data-retry="${qi}">重新作答</button>`}` : ''}
                </div>`;
            }).join('')}
          </div>
        </div>`;

      container.querySelectorAll('.lw-quiz-opt').forEach((btn) => {
        btn.addEventListener('click', () => {
          picked[Number(btn.dataset.q)] = Number(btn.dataset.o);
          render();
        });
      });
      container.querySelectorAll('[data-retry]').forEach((btn) => {
        btn.addEventListener('click', () => {
          delete picked[Number(btn.dataset.retry)];
          render();
        });
      });
    }

    render();
  });

  /** 配色切换后重绘全部组件。组件里的定时器通过 data-gen 判断自己是否已过期。 */
  function remountAll() {
    document.querySelectorAll('.llm-widget[data-widget]').forEach((node) => {
      node.dataset.gen = String(Number(node.dataset.gen || 0) + 1);
      node.dataset.mounted = 'false';
      node.innerHTML = '';
    });
    mountAll();
  }

  window.LLMWidgets = { register, mountAll, remountAll, formatNum, escapeHtml, color, renderMath };
})();
