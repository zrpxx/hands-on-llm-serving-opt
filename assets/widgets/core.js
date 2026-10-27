/**
 * Core Interactive Widget Engine for Hands-On LLM Serving & Optimization Tutorial
 * Provides declarative mounting (<div class="llm-widget" data-widget="..."></div>),
 * reusable UI controls, and the interactive self-check quiz widget.
 */
(function () {
  const registry = {};

  function register(name, renderFn) {
    registry[name] = renderFn;
  }

  function formatNum(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(n)) return '-';
    if (Math.abs(n) >= 1000) {
      return Number(n.toFixed(digits)).toLocaleString('en-US');
    }
    return Number(n.toFixed(digits)).toString();
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function mountAll() {
    const nodes = document.querySelectorAll('.llm-widget[data-widget]');
    nodes.forEach((node) => {
      if (node.dataset.mounted === 'true') return;
      const name = node.dataset.widget;
      const renderFn = registry[name];
      if (!renderFn) {
        node.innerHTML = `<div class="lw-error">交互组件 <code>${escapeHtml(name)}</code> 尚未加载</div>`;
        return;
      }
      try {
        node.dataset.mounted = 'true';
        renderFn(node);
      } catch (err) {
        console.error('[LLMWidgets] Failed to mount widget:', name, err);
        node.innerHTML = `<div class="lw-error">组件渲染异常：${escapeHtml(err.message)}</div>`;
      }
    });
  }

  // Built-in interactive quiz widget
  // Usage: <div class="llm-widget" data-widget="quiz" data-title="..." data-questions='[...]'></div>
  register('quiz', function (container) {
    let questions = [];
    try {
      const raw = container.getAttribute('data-questions') || '[]';
      questions = JSON.parse(raw);
    } catch (e) {
      container.innerHTML = '<div class="lw-error">自测题数据解析失败</div>';
      return;
    }
    const title = container.getAttribute('data-title') || '闯关自测：检验你的直觉';
    const state = { answers: {} };

    function render() {
      const total = questions.length;
      const correctCount = questions.filter((q, idx) => state.answers[idx] === q.answer).length;

      container.innerHTML = `
        <div class="lw-card lw-quiz-card">
          <div class="lw-header">
            <div class="lw-title-group">
              <span class="lw-badge">Interactive Check</span>
              <h4 class="lw-title">${escapeHtml(title)}</h4>
            </div>
            <div class="lw-quiz-score">
              已答对 <strong>${correctCount}</strong> / ${total} 题
            </div>
          </div>
          <div class="lw-quiz-list">
            ${questions
              .map((q, qIdx) => {
                const picked = state.answers[qIdx];
                const answered = picked !== undefined;
                const isRight = picked === q.answer;
                return `
                  <div class="lw-quiz-item ${answered ? (isRight ? 'is-correct' : 'is-wrong') : ''}">
                    <div class="lw-quiz-q"><strong>Q${qIdx + 1}.</strong> ${q.q}</div>
                    <div class="lw-quiz-opts">
                      ${q.options
                        .map((opt, oIdx) => {
                          let cls = 'lw-quiz-opt';
                          if (answered) {
                            if (oIdx === q.answer) cls += ' opt-correct';
                            else if (oIdx === picked) cls += ' opt-wrong';
                          }
                          return `<button type="button" class="${cls}" data-q="${qIdx}" data-o="${oIdx}">
                            <span class="opt-letter">${String.fromCharCode(65 + oIdx)}</span>
                            <span class="opt-text">${opt}</span>
                          </button>`;
                        })
                        .join('')}
                    </div>
                    ${
                      answered
                        ? `<div class="lw-quiz-explain ${isRight ? 'exp-ok' : 'exp-err'}">
                            <strong>${isRight ? '回答正确！' : '再想一想：'}</strong> ${q.explain}
                          </div>`
                        : ''
                    }
                  </div>
                `;
              })
              .join('')}
          </div>
        </div>
      `;

      container.querySelectorAll('.lw-quiz-opt').forEach((btn) => {
        btn.addEventListener('click', () => {
          const qIdx = Number(btn.getAttribute('data-q'));
          const oIdx = Number(btn.getAttribute('data-o'));
          state.answers[qIdx] = oIdx;
          render();
        });
      });
    }

    render();
  });

  window.LLMWidgets = {
    register,
    mountAll,
    formatNum,
    escapeHtml,
  };
})();
