// PatternRAG Dashboard Client Logic
// Pure Vanilla JavaScript — Zero external runtime dependencies

document.addEventListener("DOMContentLoaded", () => {
  const data = window.BENCHMARK_DATA;
  if (!data) {
    console.error("Benchmark data not loaded.");
    return;
  }

  initTabs();
  renderSummaryCards(data);
  renderRetrievalMetrics(data);
  renderRagasMetrics(data);
  renderLatencyMetrics(data);
  renderObserverTelemetry(data);
  renderArchitecturalCards(data);
  renderQueryExplorer(data);
  renderFiguresGallery();
  initModal();
});

// -------------------------------------------------------------
// 1. Tab Navigation
// -------------------------------------------------------------
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

// -------------------------------------------------------------
// 2. Summary Cards
// -------------------------------------------------------------
function renderSummaryCards(data) {
  const container = document.getElementById("summary-grid");
  if (!container) return;

  container.innerHTML = data.summaryCards.map(c => `
    <div class="summary-card ${c.type}">
      <div class="card-top">
        <span class="card-title">${escapeHtml(c.title)}</span>
        <span class="card-badge">${escapeHtml(c.badge)}</span>
      </div>
      <div class="card-value">${escapeHtml(c.value)}</div>
      <div class="card-subtext">${escapeHtml(c.subtext)}</div>
      <div class="card-desc">${escapeHtml(c.description)}</div>
    </div>
  `).join("");
}

// -------------------------------------------------------------
// 3. Retrieval Metrics Table & Interactive SVG Chart
// -------------------------------------------------------------
function renderRetrievalMetrics(data) {
  const tbody = document.getElementById("retrieval-table-body");
  if (!tbody) return;

  const rows = data.retrievalMetrics.aggregate;
  tbody.innerHTML = rows.map(r => {
    const monoPct = (r.monolithic * 100).toFixed(1);
    const pragPct = (r.patternrag * 100).toFixed(1);
    const diffClass = r.diff === 0 ? "zero" : (r.diff > 0 ? "positive" : "negative");
    const diffSign = r.diff > 0 ? "+" : "";

    return `
      <tr>
        <td><strong>${escapeHtml(r.metric)}</strong> <span style="color:var(--text-dim);font-size:11px;">(${escapeHtml(r.label)})</span></td>
        <td class="num-cell">${r.monolithic.toFixed(4)}</td>
        <td class="num-cell" style="color:var(--accent-blue);font-weight:700;">${r.patternrag.toFixed(4)}</td>
        <td>
          <span class="diff-badge ${diffClass}">
            ${diffSign}${r.pctDiff}
          </span>
        </td>
        <td class="bar-cell">
          <div class="comparison-meter" title="Mono: ${monoPct}% | PatternRAG: ${pragPct}%">
            <div class="meter-fill prag" style="width: ${pragPct}%;"></div>
          </div>
        </td>
      </tr>
    `;
  }).join("");

  renderRetrievalChart(rows);
}

function renderRetrievalChart(rows) {
  const container = document.getElementById("retrieval-chart-container");
  if (!container) return;

  const width = 800;
  const height = 300;
  const padding = { top: 30, right: 30, bottom: 50, left: 60 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const metrics = rows.map(r => r.metric);
  const barGroupWidth = chartWidth / metrics.length;
  const barWidth = 14;

  let svg = `
    <svg class="custom-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
      <!-- Grid lines -->
      <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
      <line x1="${padding.left}" y1="${padding.top + chartHeight * 0.5}" x2="${width - padding.right}" y2="${padding.top + chartHeight * 0.5}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
      <line x1="${padding.left}" y1="${padding.top + chartHeight}" x2="${width - padding.right}" y2="${padding.top + chartHeight}" stroke="rgba(255,255,255,0.15)"/>

      <!-- Y Axis Labels -->
      <text x="${padding.left - 12}" y="${padding.top + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">1.00</text>
      <text x="${padding.left - 12}" y="${padding.top + chartHeight * 0.5 + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">0.50</text>
      <text x="${padding.left - 12}" y="${padding.top + chartHeight + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">0.00</text>
  `;

  rows.forEach((r, idx) => {
    const xCenter = padding.left + (idx + 0.5) * barGroupWidth;
    const monoHeight = r.monolithic * chartHeight;
    const pragHeight = r.patternrag * chartHeight;

    const monoY = padding.top + chartHeight - monoHeight;
    const pragY = padding.top + chartHeight - pragHeight;

    svg += `
      <!-- Monolithic Bar -->
      <rect x="${xCenter - barWidth - 2}" y="${monoY}" width="${barWidth}" height="${monoHeight}" fill="#64748b" rx="2">
        <title>Monolithic ${r.metric}: ${r.monolithic.toFixed(4)}</title>
      </rect>

      <!-- PatternRAG Bar -->
      <rect x="${xCenter + 2}" y="${pragY}" width="${barWidth}" height="${pragHeight}" fill="#3b82f6" rx="2">
        <title>PatternRAG ${r.metric}: ${r.patternrag.toFixed(4)}</title>
      </rect>

      <!-- Label -->
      <text x="${xCenter}" y="${height - 20}" fill="#94a3b8" font-size="11" text-anchor="middle" font-weight="600" font-family="sans-serif">
        ${r.metric}
      </text>
    `;
  });

  // Legend
  svg += `
    <g transform="translate(${width - 240}, 10)">
      <rect x="0" y="0" width="12" height="12" fill="#64748b" rx="2"/>
      <text x="18" y="10" fill="#94a3b8" font-size="11" font-family="sans-serif">Monolithic</text>
      <rect x="90" y="0" width="12" height="12" fill="#3b82f6" rx="2"/>
      <text x="108" y="10" fill="#60a5fa" font-size="11" font-family="sans-serif">PatternRAG</text>
    </g>
    </svg>
  `;

  container.innerHTML = svg;
}

// -------------------------------------------------------------
// 4. RAGAS Answer Quality
// -------------------------------------------------------------
function renderRagasMetrics(data) {
  const tbody = document.getElementById("ragas-table-body");
  if (!tbody) return;

  const rows = data.ragasMetrics.aggregate;
  tbody.innerHTML = rows.map(r => {
    const diffSign = r.diff > 0 ? "+" : "";
    const diffClass = r.diff === 0 ? "zero" : (r.diff > 0 ? "positive" : "negative");

    return `
      <tr>
        <td><strong>${escapeHtml(r.label)}</strong></td>
        <td class="num-cell">${r.monolithic.toFixed(4)}</td>
        <td class="num-cell" style="color:var(--accent-blue);font-weight:700;">${r.patternrag.toFixed(4)}</td>
        <td>
          <span class="diff-badge ${diffClass}">
            ${diffSign}${r.pctDiff}
          </span>
        </td>
        <td style="color:var(--text-muted);font-size:12px;">${escapeHtml(r.interpretation)}</td>
      </tr>
    `;
  }).join("");

  renderRagasChart(rows);
}

function renderRagasChart(rows) {
  const container = document.getElementById("ragas-chart-container");
  if (!container) return;

  const width = 600;
  const height = 240;
  const padding = { top: 30, right: 30, bottom: 45, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const barGroupWidth = chartWidth / rows.length;
  const barWidth = 20;

  let svg = `
    <svg class="custom-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
      <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
      <line x1="${padding.left}" y1="${padding.top + chartHeight * 0.5}" x2="${width - padding.right}" y2="${padding.top + chartHeight * 0.5}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4"/>
      <line x1="${padding.left}" y1="${padding.top + chartHeight}" x2="${width - padding.right}" y2="${padding.top + chartHeight}" stroke="rgba(255,255,255,0.15)"/>

      <text x="${padding.left - 10}" y="${padding.top + 4}" fill="#64748b" font-size="10" text-anchor="end" font-family="monospace">1.0</text>
      <text x="${padding.left - 10}" y="${padding.top + chartHeight * 0.5 + 4}" fill="#64748b" font-size="10" text-anchor="end" font-family="monospace">0.5</text>
      <text x="${padding.left - 10}" y="${padding.top + chartHeight + 4}" fill="#64748b" font-size="10" text-anchor="end" font-family="monospace">0.0</text>
  `;

  rows.forEach((r, idx) => {
    const xCenter = padding.left + (idx + 0.5) * barGroupWidth;
    const monoH = r.monolithic * chartHeight;
    const pragH = r.patternrag * chartHeight;

    svg += `
      <rect x="${xCenter - barWidth - 2}" y="${padding.top + chartHeight - monoH}" width="${barWidth}" height="${monoH}" fill="#64748b" rx="2"/>
      <rect x="${xCenter + 2}" y="${padding.top + chartHeight - pragH}" width="${barWidth}" height="${pragH}" fill="#10b981" rx="2"/>
      <text x="${xCenter}" y="${height - 18}" fill="#cbd5e1" font-size="11" text-anchor="middle" font-weight="600">${r.label}</text>
    `;
  });

  svg += `
    <g transform="translate(${width - 230}, 8)">
      <rect x="0" y="0" width="10" height="10" fill="#64748b" rx="2"/>
      <text x="16" y="9" fill="#94a3b8" font-size="10">Monolithic</text>
      <rect x="85" y="0" width="10" height="10" fill="#10b981" rx="2"/>
      <text x="101" y="9" fill="#34d399" font-size="10">PatternRAG</text>
    </g>
    </svg>
  `;

  container.innerHTML = svg;
}

// -------------------------------------------------------------
// 5. Latency Profile
// -------------------------------------------------------------
function renderLatencyMetrics(data) {
  const tbody = document.getElementById("latency-table-body");
  if (!tbody) return;

  const lat = data.latencyMetrics;
  const stages = [
    { name: "Retrieval", stats: lat.retrieval },
    { name: "Generation (LLM)", stats: lat.generation },
    { name: "Total Pipeline", stats: lat.total }
  ];

  let html = "";
  stages.forEach(st => {
    ["mean_ms", "median_ms", "p95_ms"].forEach(statKey => {
      const row = st.stats[statKey];
      const label = statKey.replace("_ms", "").toUpperCase();
      const diffSign = row.diff_ms > 0 ? "+" : "";

      html += `
        <tr>
          <td><strong>${escapeHtml(st.name)}</strong></td>
          <td style="color:var(--text-dim);font-size:12px;">${label}</td>
          <td class="num-cell">${row.monolithic.toFixed(2)} ms</td>
          <td class="num-cell" style="color:var(--accent-blue);font-weight:700;">${row.patternrag.toFixed(2)} ms</td>
          <td class="num-cell" style="color:${row.diff_ms > 0 ? 'var(--text-muted)' : 'var(--accent-emerald)'};">
            ${diffSign}${row.diff_ms.toFixed(2)} ms
          </td>
        </tr>
      `;
    });
  });

  tbody.innerHTML = html;
}

// -------------------------------------------------------------
// 6. Observer Telemetry
// -------------------------------------------------------------
function renderObserverTelemetry(data) {
  const container = document.getElementById("observer-telemetry-container");
  if (!container) return;

  const obs = data.observerTelemetry;
  container.innerHTML = `
    <div class="pattern-card-grid">
      <div class="pattern-card">
        <div class="pattern-header">
          <span class="pattern-name">LatencyLogger Observer</span>
          <span class="pattern-role">Observer Pattern</span>
        </div>
        <p class="pattern-desc">Passively hooks into pipeline dispatch events without modifying execution code.</p>
        <div class="pattern-components">
          <div>Queries Tracked: <strong>${obs.latencyLogger.queries_tracked}</strong></div>
          <div>Avg Retrieval: <strong>${obs.latencyLogger.retrieval_mean_ms} ms</strong></div>
          <div>Avg Generation: <strong>${obs.latencyLogger.generation_mean_ms} ms</strong></div>
        </div>
      </div>

      <div class="pattern-card">
        <div class="pattern-header">
          <span class="pattern-name">MetricsCollector Observer</span>
          <span class="pattern-role">Observer Pattern</span>
        </div>
        <p class="pattern-desc">Gathers retrieval counts, rank distributions, and scoring telemetry.</p>
        <div class="pattern-components">
          <div>Tracked Queries: <strong>${obs.metricsCollector.total_queries}</strong></div>
          <div>Mean Docs Retrieved: <strong>${obs.metricsCollector.mean_docs_retrieved}</strong></div>
          <div>Configured Top-K: <strong>${obs.metricsCollector.top_k_configured}</strong></div>
        </div>
      </div>

      <div class="pattern-card">
        <div class="pattern-header">
          <span class="pattern-name">CostMonitor Observer</span>
          <span class="pattern-role">Observer Pattern</span>
        </div>
        <p class="pattern-desc">Calculates token consumption and estimated API expenditure per query.</p>
        <div class="pattern-components">
          <div>Prompt Tokens: <strong>${obs.costMonitor.total_prompt_tokens.toLocaleString()}</strong></div>
          <div>Completion Tokens: <strong>${obs.costMonitor.total_completion_tokens.toLocaleString()}</strong></div>
          <div>Total Tokens: <strong>${obs.costMonitor.total_tokens.toLocaleString()}</strong></div>
          <div>Total Est Cost: <strong>$${obs.costMonitor.estimated_cost_usd} USD</strong></div>
        </div>
      </div>
    </div>
  `;
}

// -------------------------------------------------------------
// 7. Architectural Extensibility Cards (M1-M3)
// -------------------------------------------------------------
function renderArchitecturalCards(data) {
  const container = document.getElementById("architectural-container");
  if (!container) return;

  container.innerHTML = data.architecturalExperiments.map(exp => `
    <div class="experiment-card">
      <div class="exp-header">
        <div>
          <span class="exp-badge">${escapeHtml(exp.id)} · ${escapeHtml(exp.pattern)}</span>
          <h3 style="font-size:16px;font-weight:700;margin-top:6px;color:#f8fafc;">${escapeHtml(exp.name)}</h3>
        </div>
        <span style="font-size:12px;font-family:monospace;color:var(--accent-emerald);">${escapeHtml(exp.testCoverage)}</span>
      </div>
      <p style="font-size:13px;color:var(--text-muted);margin-bottom:12px;">${escapeHtml(exp.goal)}</p>
      <div class="exp-diff-compare">
        <div class="diff-box mono">
          <strong>Monolithic Baseline: Invasive Changes</strong>
          ${escapeHtml(exp.monolithicImpact)}
        </div>
        <div class="diff-box prag">
          <strong>PatternRAG: Clean Extension</strong>
          ${escapeHtml(exp.patternRagImpact)}
        </div>
      </div>
    </div>
  `).join("");
}

// -------------------------------------------------------------
// 8. 30-Query Explorer
// -------------------------------------------------------------
let allQueries = [];

function renderQueryExplorer(data) {
  allQueries = data.queries;
  const listContainer = document.getElementById("query-list-container");
  const searchInput = document.getElementById("query-search-input");
  const filterSelect = document.getElementById("query-filter-select");

  function updateList() {
    const term = (searchInput.value || "").toLowerCase().trim();
    const filter = filterSelect.value;

    const filtered = allQueries.filter(q => {
      const matchSearch = 
        q.question.toLowerCase().includes(term) ||
        q.gold_answer.toLowerCase().includes(term) ||
        q.monolithic.predicted_answer.toLowerCase().includes(term) ||
        q.patternrag.predicted_answer.toLowerCase().includes(term) ||
        q.id.toLowerCase().includes(term);

      if (!matchSearch) return false;

      if (filter === "agreement") {
        return q.monolithic.predicted_answer.trim() === q.patternrag.predicted_answer.trim();
      } else if (filter === "disagreement") {
        return q.monolithic.predicted_answer.trim() !== q.patternrag.predicted_answer.trim();
      } else if (filter === "high-faithfulness") {
        return q.patternrag.scores.faithfulness >= 0.8;
      }
      return true;
    });

    renderQueryItems(filtered, listContainer);
  }

  if (searchInput) searchInput.addEventListener("input", updateList);
  if (filterSelect) filterSelect.addEventListener("change", updateList);

  updateList();
}

function renderQueryItems(queries, container) {
  if (!container) return;

  if (queries.length === 0) {
    container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-dim);">No matching queries found.</div>`;
    return;
  }

  container.innerHTML = queries.map(q => {
    const isAgree = q.monolithic.predicted_answer.trim() === q.patternrag.predicted_answer.trim();
    const agreePill = isAgree 
      ? `<span class="diff-badge zero" style="font-size:10px;">Identical Answer</span>`
      : `<span class="diff-badge positive" style="font-size:10px;">Variation Found</span>`;

    return `
      <div class="query-card" id="q-card-${q.index}">
        <div class="q-top-row">
          <span class="q-index">#${q.index} · ${q.id.slice(0, 8)}...</span>
          <div class="q-metrics-pills">
            ${agreePill}
            <span class="q-pill">Faith: ${q.patternrag.scores.faithfulness.toFixed(1)}</span>
            <span class="q-pill">Relevancy: ${q.patternrag.scores.answer_relevancy.toFixed(2)}</span>
            <span class="q-pill">Retrieval: ${q.patternrag.retrieval_ms.toFixed(0)}ms</span>
          </div>
        </div>

        <div class="q-question">${escapeHtml(q.question)}</div>

        <div class="q-answer-compare">
          <div class="q-ans-box">
            <div class="label">Monolithic Predicted:</div>
            <div class="val">${escapeHtml(q.monolithic.predicted_answer) || '<em style="color:#64748b;">(No answer generated)</em>'}</div>
          </div>
          <div class="q-ans-box">
            <div class="label" style="color:var(--accent-blue);">PatternRAG Predicted:</div>
            <div class="val" style="color:#93c5fd;">${escapeHtml(q.patternrag.predicted_answer) || '<em style="color:#64748b;">(No answer generated)</em>'}</div>
          </div>
        </div>

        <div class="q-gold">
          <span>Target Ground Truth:</span>
          <strong>${escapeHtml(q.gold_answer)}</strong>
        </div>

        <button class="q-docs-toggle" onclick="toggleDocList('${q.index}')">
          <span>📄 View Top-10 Retrieved Documents (${q.retrieved_doc_ids.length} docs, 100% Equivalence)</span>
        </button>

        <div class="q-docs-list" id="docs-list-${q.index}">
          <div style="font-size:11px;font-weight:700;color:var(--text-muted);margin-bottom:6px;">Ranked Document IDs (Identical across both architectures):</div>
          ${q.retrieved_doc_ids.map((docId, dIdx) => `
            <div class="q-doc-item">
              <span style="color:var(--accent-blue);">${dIdx + 1}.</span> ${escapeHtml(docId)}
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }).join("");
}

window.toggleDocList = function(index) {
  const el = document.getElementById(`docs-list-${index}`);
  if (el) {
    el.classList.toggle("show");
  }
};

// -------------------------------------------------------------
// 9. Figures Lightbox
// -------------------------------------------------------------
function renderFiguresGallery() {
  const cards = document.querySelectorAll(".figure-card");
  cards.forEach(card => {
    card.addEventListener("click", () => {
      const img = card.querySelector("img");
      const caption = card.querySelector(".figure-caption");
      if (img) {
        openModal(img.src, caption ? caption.innerText : "Figure Inspection");
      }
    });
  });
}

function initModal() {
  const overlay = document.getElementById("modal-overlay");
  const closeBtn = document.getElementById("modal-close-btn");

  if (!overlay || !closeBtn) return;

  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

function openModal(imgSrc, title) {
  const overlay = document.getElementById("modal-overlay");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");

  if (overlay && modalImg) {
    modalImg.src = imgSrc;
    if (modalTitle) modalTitle.innerText = title;
    overlay.classList.add("open");
  }
}

function closeModal() {
  const overlay = document.getElementById("modal-overlay");
  if (overlay) overlay.classList.remove("open");
}

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
