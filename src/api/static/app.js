// Enterprise Risk Intelligence — dashboard logic.
// Pulls from the FastAPI endpoints under the same origin and renders Plotly charts.

const API = '';
const RATING_COLORS = {
  AAA:'#00AA00', AA:'#44CC44', A:'#88DD88', BBB:'#FFDD00', BB:'#FFAA00',
  B:'#FF7700', CCC:'#FF5500', CC:'#FF3300', C:'#CC0000', D:'#990000',
};
const RATING_ORDER = ['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'];

const PL_LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
          color: '#cbd5e1', size: 12 },
  margin: { t: 28, r: 18, b: 48, l: 56 },
  xaxis: { gridcolor: '#252a37', zerolinecolor: '#252a37' },
  yaxis: { gridcolor: '#252a37', zerolinecolor: '#252a37' },
  legend: { font: { color: '#cbd5e1' } },
  hoverlabel: { bgcolor: '#141822', bordercolor: '#252a37', font: { color: '#e2e8f0' } },
};
const PL_CONFIG = { displayModeBar: false, responsive: true };

// ---------- helpers ----------
const $  = (s, c=document) => c.querySelector(s);
const $$ = (s, c=document) => Array.from(c.querySelectorAll(s));

const fmt = {
  pct:   (v, d=1) => v == null || isNaN(v) ? '—' : (v*100).toFixed(d) + '%',
  num:   (v, d=2) => v == null || isNaN(v) ? '—' : Number(v).toLocaleString(undefined, {maximumFractionDigits:d}),
  money: v => {
    if (v == null || isNaN(v)) return '—';
    const a = Math.abs(v);
    if (a >= 1e12) return '$' + (v/1e12).toFixed(2) + 'T';
    if (a >= 1e9)  return '$' + (v/1e9).toFixed(2) + 'B';
    if (a >= 1e6)  return '$' + (v/1e6).toFixed(2) + 'M';
    if (a >= 1e3)  return '$' + (v/1e3).toFixed(1) + 'K';
    return '$' + v.toFixed(0);
  },
  date:  v => v ? String(v).slice(0,10) : '—',
};

async function fetchJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res.json();
}

function showLoading(text='Loading…') { $('#loading-text').textContent = text; $('#loading').classList.remove('hidden'); }
function hideLoading() { $('#loading').classList.add('hidden'); }
function ratingPill(r) { return r ? `<span class="pill pill-${r}">${r}</span>` : '<span class="muted">—</span>'; }

function clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// The dataset holds one row per company per as_of_date; for "top N" views we
// want each company once, at its riskiest observation.
function dedupeByCompany(companies) {
  const best = new Map();
  companies.forEach(c => {
    const key = c.symbol || c.company_name || c.company_id;
    const prev = best.get(key);
    if (!prev || (c.aggregated_risk||0) > (prev.aggregated_risk||0)) best.set(key, c);
  });
  return [...best.values()];
}
function quantile(arr, q) {
  const s = [...arr].sort((a,b) => a-b);
  const i = q * (s.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
}

// ---------- state ----------
const state = { companies: [], industries: [], filtered: [] };

document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupRefresh();
  setupFilters();
  try {
    showLoading('Loading dashboard…');
    await loadAll();
  } catch (e) {
    console.error(e);
    alert('Failed to load data: ' + e.message);
  } finally {
    hideLoading();
  }
});

function setupTabs() {
  $$('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.tab').forEach(t => t.classList.remove('tab-active'));
      btn.classList.add('tab-active');
      const target = btn.dataset.tab;
      $$('[data-panel]').forEach(p => p.classList.toggle('hidden', p.dataset.panel !== target));
      // Plotly needs a kick after un-hiding its container
      window.dispatchEvent(new Event('resize'));
    });
  });
}

function setupRefresh() {
  $('#refresh-btn').addEventListener('click', async () => {
    try {
      showLoading('Rebuilding model cache…');
      await fetch(API + '/refresh', { method: 'POST' });
      await loadAll();
    } catch (e) {
      alert('Refresh failed: ' + e.message);
    } finally {
      hideLoading();
    }
  });
}

function setupFilters() {
  $('#search-input').addEventListener('input', renderCompaniesTable);
  $('#industry-filter').addEventListener('change', renderCompaniesTable);
  $('#rating-filter').addEventListener('change', renderCompaniesTable);
}

async function loadAll() {
  const [health, companies, dist, industries, trend, scenarios, corr, returns, modelMetrics] = await Promise.all([
    fetchJSON('/health'),
    fetchJSON('/companies'),
    fetchJSON('/risk-distribution'),
    fetchJSON('/industry-benchmarks'),
    fetchJSON('/time-series'),
    fetchJSON('/scenarios'),
    fetchJSON('/correlations'),
    fetchJSON('/returns'),
    fetchJSON('/model/metrics'),
  ]);

  state.companies = companies;
  state.industries = [...new Set(companies.map(c => c.industry).filter(Boolean))].sort();

  renderHeader(health);
  renderKPIs(companies, dist);
  renderRatingDonut(dist);
  renderProbaHist(companies);
  renderSectorTreemap(companies);
  renderTopRisky(companies);
  renderWatchlist(companies);

  populateIndustryFilter();
  renderCompaniesTable();

  renderIndustriesChart(industries);
  renderIndustriesCount(industries);
  renderIndustriesTable(industries);

  renderScatterLevLiq(companies);
  renderScatterProfit(companies);
  renderCorrelation(corr);
  renderAltmanHist(companies);
  renderCoverageHist(companies);

  renderTrend(trend);
  renderTrendByIndustry(companies);

  renderScenarioCards(scenarios);
  renderScenariosChart(scenarios);
  renderScenariosTable(scenarios);

  await renderVarPanel(returns);

  renderModelPanel(modelMetrics);
}

// ============ Header ============
function renderHeader(health) {
  const built = health.last_built ? new Date(health.last_built).toLocaleString() : '—';
  $('#dataset-info').innerHTML = `<strong>${health.rows.toLocaleString()}</strong> rows<br>built ${built}`;
}

// ============ KPIs ============
function renderKPIs(companies, dist) {
  const n = companies.length;
  const avgRisk = companies.reduce((s,c) => s + (c.aggregated_risk || 0), 0) / Math.max(1, n);
  const avgProba = companies.reduce((s,c) => s + (c.predicted_proba || 0), 0) / Math.max(1, n);
  const flagged = companies.filter(c => c.risk_flag === 1).length;
  const distressed = dist.filter(r => ['CCC','CC','C','D'].includes(r.risk_rating))
                          .reduce((s,r) => s + r.count, 0);
  const investmentGrade = dist.filter(r => ['AAA','AA','A','BBB'].includes(r.risk_rating))
                              .reduce((s,r) => s + r.count, 0);

  const kpis = [
    { lbl:'Companies analysed', val: n.toLocaleString(), sub:`${state.industries.length} industries` },
    { lbl:'Mean default prob.', val: fmt.pct(avgProba), sub:`Aggregated risk ${fmt.num(avgRisk, 3)}`, cls: avgProba > 0.5 ? 'bad' : avgProba > 0.2 ? 'warn' : 'good' },
    { lbl:'Investment grade',   val: investmentGrade.toLocaleString(), sub:`${fmt.pct(investmentGrade/Math.max(1,n))} of portfolio · AAA–BBB`, cls:'good' },
    { lbl:'Distressed (CCC–D)', val: distressed.toLocaleString(),      sub:`${fmt.pct(distressed/Math.max(1,n))} of portfolio · flagged ${flagged}`, cls:'bad' },
  ];
  $('#kpi-grid').innerHTML = kpis.map(k => `
    <div class="kpi ${k.cls||''}">
      <div class="lbl">${k.lbl}</div>
      <div class="val">${k.val}</div>
      <div class="sub">${k.sub}</div>
    </div>`).join('');
}

// ============ Rating donut ============
function renderRatingDonut(dist) {
  const filt = RATING_ORDER.map(r => dist.find(d => d.risk_rating === r)).filter(d => d && d.count > 0);
  Plotly.newPlot('chart-rating', [{
    type:'pie', hole:0.6,
    labels: filt.map(d => d.risk_rating),
    values: filt.map(d => d.count),
    marker: { colors: filt.map(d => RATING_COLORS[d.risk_rating]), line: { color:'#141822', width: 2 } },
    textinfo:'label+percent', textfont:{color:'#fff', size:12},
    hovertemplate: '<b>%{label}</b><br>%{value} companies<br>%{percent}<extra></extra>',
  }], { ...PL_LAYOUT, showlegend: false, margin:{t:8,b:8,l:8,r:8} }, PL_CONFIG);
}

// ============ Default-prob histogram ============
function renderProbaHist(companies) {
  const vals = companies.map(c => c.predicted_proba).filter(v => v != null);
  const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
  Plotly.newPlot('chart-proba-hist', [{
    type:'histogram', x: vals, nbinsx: 30,
    marker: { color:'#7c8cff', line: { color:'#141822', width: 1 } },
    hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Predicted default probability', tickformat:'.0%' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Companies' },
    shapes: [{ type:'line', x0: mean, x1: mean, y0:0, y1:1, yref:'paper',
               line: { color:'#ef4444', width:2, dash:'dash' } }],
    annotations: [{ x: mean, y:1, yref:'paper', text:`mean ${fmt.pct(mean)}`,
                    showarrow:false, font:{color:'#ef4444'}, xanchor:'left' }],
  }, PL_CONFIG);
}

// ============ Sector treemap ============
function renderSectorTreemap(companies) {
  const groups = {};
  companies.forEach(c => {
    const sec = c.sector || c.industry || 'Unknown';
    const ind = c.industry || 'Unknown';
    if (!groups[sec]) groups[sec] = {};
    groups[sec][ind] = (groups[sec][ind] || 0) + (c.revenue || 0);
  });
  const ids = [], labels = [], parents = [], values = [];
  Object.entries(groups).forEach(([sec, inds]) => {
    // branchvalues:'total' requires parent value >= sum of children
    const secTotal = Object.values(inds).reduce((s,v) => s+v, 0);
    ids.push(sec); labels.push(sec); parents.push(''); values.push(secTotal);
    Object.entries(inds).forEach(([ind, rev]) => {
      ids.push(`${sec}/${ind}`); labels.push(ind); parents.push(sec); values.push(rev);
    });
  });
  Plotly.newPlot('chart-sector-tree', [{
    type:'treemap', ids, labels, parents, values, branchvalues:'total',
    marker: { colorscale:[[0,'#1e3a8a'],[0.5,'#7c8cff'],[1,'#9f7aea']], line:{color:'#141822', width:2} },
    textfont: { color:'#fff' },
    hovertemplate: '<b>%{label}</b><br>Revenue: $%{value:,.0f}<extra></extra>',
  }], { ...PL_LAYOUT, margin:{t:8,b:8,l:8,r:8} }, PL_CONFIG);
}

// ============ Top-risky bar ============
function renderTopRisky(companies) {
  const top = dedupeByCompany(companies)
    .sort((a,b) => (b.aggregated_risk||0) - (a.aggregated_risk||0))
    .slice(0, 15)
    .reverse();
  Plotly.newPlot('chart-top-risky', [{
    type:'bar', orientation:'h',
    x: top.map(c => c.aggregated_risk),
    y: top.map(c => `${c.company_name||c.company_id} (${c.symbol||''})`),
    marker: { color: top.map(c => RATING_COLORS[c.risk_rating] || '#7c8cff') },
    text: top.map(c => c.risk_rating),
    textposition:'outside',
    hovertemplate: '<b>%{y}</b><br>Risk: %{x:.3f}<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Aggregated risk score' },
    yaxis: { ...PL_LAYOUT.yaxis, automargin: true },
    margin: { t: 8, r: 40, b: 50, l: 10 },
  }, PL_CONFIG);
}

// ============ Watchlist mini-table ============
function renderWatchlist(companies) {
  const top = dedupeByCompany(companies)
    .sort((a,b) => (b.aggregated_risk||0) - (a.aggregated_risk||0))
    .slice(0, 8);
  $('#watchlist-tbody').innerHTML = top.map(c => `
    <tr onclick='openCompany(${JSON.stringify(c.company_id)})'>
      <td>
        <div>${c.company_name || c.company_id}</div>
        <div class="secondary">${c.industry || ''}</div>
      </td>
      <td class="num mono">${fmt.num(c.aggregated_risk, 3)}</td>
      <td>${ratingPill(c.risk_rating)}</td>
    </tr>
  `).join('');
}

// ============ Companies table ============
function populateIndustryFilter() {
  $('#industry-filter').innerHTML = '<option value="">All industries</option>' +
    state.industries.map(i => `<option value="${i}">${i}</option>`).join('');
}

function renderCompaniesTable() {
  const q = $('#search-input').value.trim().toLowerCase();
  const ind = $('#industry-filter').value;
  const rat = $('#rating-filter').value;

  let rows = state.companies;
  if (q) rows = rows.filter(c => `${c.company_name||''} ${c.symbol||''} ${c.company_id||''}`.toLowerCase().includes(q));
  if (ind) rows = rows.filter(c => c.industry === ind);
  if (rat) rows = rows.filter(c => c.risk_rating === rat);
  rows = [...rows].sort((a,b) => (b.aggregated_risk||0) - (a.aggregated_risk||0));
  state.filtered = rows;

  $('#companies-tbody').innerHTML = rows.slice(0, 300).map(c => `
    <tr onclick='openCompany(${JSON.stringify(c.company_id)})'>
      <td>
        <div>${c.company_name || c.company_id}</div>
        <div class="secondary">${c.symbol || c.company_id} · ${fmt.date(c.as_of_date)}</div>
      </td>
      <td>${c.industry || '—'}</td>
      <td class="num">${fmt.money(c.revenue)}</td>
      <td class="num">${fmt.pct(c.ebitda_margin)}</td>
      <td class="num">${fmt.num(c.leverage_ratio)}</td>
      <td class="num">${fmt.num(c.current_ratio)}</td>
      <td class="num">${fmt.pct(c.predicted_proba)}</td>
      <td>${ratingPill(c.risk_rating)}</td>
      <td class="num mono">${fmt.num(c.aggregated_risk, 3)}</td>
      <td class="muted small">▸</td>
    </tr>
  `).join('');
  const total = state.companies.length, shown = Math.min(rows.length, 300);
  $('#companies-footer').textContent =
    rows.length === total
      ? `Showing ${shown} of ${total} companies${rows.length > 300 ? ' (first 300)' : ''}`
      : `Filter: ${rows.length} match · showing first ${shown}`;
}

// ============ Industries ============
function renderIndustriesChart(rows) {
  const sorted = [...rows].sort((a,b) => (a.aggregated_risk||0) - (b.aggregated_risk||0));
  Plotly.newPlot('chart-industries', [{
    type:'bar', orientation:'h',
    x: sorted.map(r => r.aggregated_risk),
    y: sorted.map(r => r.industry || '—'),
    marker: {
      color: sorted.map(r => r.aggregated_risk),
      colorscale: [[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']],
      showscale: false,
    },
    text: sorted.map(r => r.company_count),
    hovertemplate: '<b>%{y}</b><br>Mean risk: %{x:.3f}<br>%{text} companies<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Mean aggregated risk' },
    yaxis: { ...PL_LAYOUT.yaxis, automargin: true },
    margin: { t: 8, r: 18, b: 48, l: 10 },
  }, PL_CONFIG);
}

function renderIndustriesCount(rows) {
  const sorted = [...rows].sort((a,b) => (b.company_count||0) - (a.company_count||0));
  Plotly.newPlot('chart-industries-count', [{
    type:'bar', orientation:'h',
    x: sorted.map(r => r.company_count),
    y: sorted.map(r => r.industry || '—'),
    marker: { color:'#7c8cff' },
    hovertemplate: '<b>%{y}</b><br>%{x} companies<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Companies' },
    yaxis: { ...PL_LAYOUT.yaxis, automargin: true },
    margin: { t: 8, r: 18, b: 48, l: 10 },
  }, PL_CONFIG);
}

function renderIndustriesTable(rows) {
  const sorted = [...rows].sort((a,b) => (b.aggregated_risk||0) - (a.aggregated_risk||0));
  $('#industries-tbody').innerHTML = sorted.map(r => `
    <tr>
      <td>${r.industry || '—'}</td>
      <td class="num">${r.company_count}</td>
      <td class="num">${fmt.pct(r.predicted_proba)}</td>
      <td class="num mono">${fmt.num(r.aggregated_risk, 3)}</td>
      <td class="num">${fmt.num(r.leverage_ratio)}</td>
      <td class="num">${fmt.num(r.current_ratio)}</td>
      <td class="num">${fmt.pct(r.ebitda_margin)}</td>
      <td class="num">${fmt.num(r.altman_z_score)}</td>
    </tr>
  `).join('');
}

// ============ Ratios scatter & friends ============
function renderScatterLevLiq(companies) {
  const pts = companies.filter(c => c.leverage_ratio != null && c.current_ratio != null);
  // clip to reasonable view range
  const xs = pts.map(c => clip(c.leverage_ratio, 0, 15));
  const ys = pts.map(c => clip(c.current_ratio, 0, 8));
  Plotly.newPlot('chart-scatter-lev-liq', [{
    type:'scatter', mode:'markers',
    x: xs, y: ys,
    marker: {
      size: pts.map(c => 6 + (c.predicted_proba||0) * 12),
      color: pts.map(c => RATING_COLORS[c.risk_rating] || '#7c8cff'),
      opacity: 0.75,
      line: { color: '#141822', width: 0.5 },
    },
    text: pts.map(c => `${c.company_name||c.company_id} (${c.risk_rating})`),
    hovertemplate: '<b>%{text}</b><br>Leverage: %{x:.2f}<br>Current ratio: %{y:.2f}<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Leverage ratio (debt / EBITDA)', range:[0, 15] },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Current ratio (current assets / liabilities)', range:[0, 8] },
  }, PL_CONFIG);
}

function renderScatterProfit(companies) {
  const pts = companies.filter(c => c.ebitda_margin != null && c.predicted_proba != null);
  Plotly.newPlot('chart-scatter-prof', [{
    type:'scatter', mode:'markers',
    x: pts.map(c => clip(c.ebitda_margin, -0.5, 1)),
    y: pts.map(c => c.predicted_proba),
    marker: {
      size: 7,
      color: pts.map(c => RATING_COLORS[c.risk_rating] || '#7c8cff'),
      opacity: 0.75,
    },
    text: pts.map(c => c.company_name || c.company_id),
    hovertemplate: '<b>%{text}</b><br>EBITDA margin: %{x:.1%}<br>Default prob: %{y:.1%}<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'EBITDA margin', tickformat:'.0%' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Predicted default probability', tickformat:'.0%' },
  }, PL_CONFIG);
}

function renderCorrelation(corr) {
  const labels = corr.fields;
  Plotly.newPlot('chart-corr', [{
    type:'heatmap',
    z: corr.matrix, x: labels, y: labels,
    colorscale: [[0,'#ef4444'],[0.5,'#141822'],[1,'#10b981']],
    zmin: -1, zmax: 1,
    text: corr.matrix.map(row => row.map(v => v.toFixed(2))),
    texttemplate: '%{text}',
    textfont: { color:'#fff', size: 10 },
    hovertemplate: '%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>',
    colorbar: { tickfont: { color:'#cbd5e1' } },
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, tickangle: -35, automargin: true },
    yaxis: { ...PL_LAYOUT.yaxis, automargin: true },
    margin: { t: 8, r: 18, b: 100, l: 130 },
  }, PL_CONFIG);
}

function renderAltmanHist(companies) {
  const vals = companies.map(c => c.altman_z_score).filter(v => v != null && isFinite(v));
  const clipped = vals.map(v => clip(v, -5, 15));
  Plotly.newPlot('chart-altman', [{
    type:'histogram', x: clipped, nbinsx: 30,
    marker: { color:'#9f7aea' },
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Altman Z-score (clipped)' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Companies' },
    shapes: [
      { type:'line', x0:1.81, x1:1.81, y0:0, y1:1, yref:'paper', line:{color:'#ef4444', dash:'dash'} },
      { type:'line', x0:2.99, x1:2.99, y0:0, y1:1, yref:'paper', line:{color:'#10b981', dash:'dash'} },
    ],
    annotations: [
      { x: 1.81, y: 1, yref:'paper', text:'distress', font:{color:'#ef4444'}, showarrow:false, xanchor:'right' },
      { x: 2.99, y: 1, yref:'paper', text:'safe',     font:{color:'#10b981'}, showarrow:false, xanchor:'left'  },
    ],
  }, PL_CONFIG);
}

function renderCoverageHist(companies) {
  const vals = companies.map(c => c.interest_coverage).filter(v => v != null && isFinite(v));
  const clipped = vals.map(v => clip(v, -5, 60));
  Plotly.newPlot('chart-coverage', [{
    type:'histogram', x: clipped, nbinsx: 30,
    marker: { color:'#7c8cff' },
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Interest coverage (EBITDA / interest)' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Companies' },
  }, PL_CONFIG);
}

// ============ Trends ============
function renderTrend(rows) {
  const x = rows.map(r => r.as_of_date);
  Plotly.newPlot('chart-trend', [
    { type:'scatter', mode:'lines+markers', name:'Mean aggregated risk',
      x, y: rows.map(r => r.avg_risk), line:{color:'#7c8cff', width:3}, marker:{size:8} },
    { type:'scatter', mode:'lines+markers', name:'Mean default prob.',
      x, y: rows.map(r => r.avg_proba), line:{color:'#9f7aea', width:3}, marker:{size:8} },
    { type:'scatter', mode:'lines+markers', name:'Mean leverage', yaxis:'y2',
      x, y: rows.map(r => r.avg_leverage), line:{color:'#f59e0b', width:2, dash:'dot'} },
  ], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'As-of date' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Risk / probability', rangemode:'tozero' },
    yaxis2: { title:'Leverage', overlaying:'y', side:'right', gridcolor:'transparent' },
    legend: { ...PL_LAYOUT.legend, orientation:'h', y:-0.2 },
  }, PL_CONFIG);
}

function renderTrendByIndustry(companies) {
  // group by (industry, date) -> mean predicted_proba
  const groups = {};
  companies.forEach(c => {
    const ind = c.industry || 'Unknown';
    const date = (c.as_of_date || '').slice(0,10);
    if (!date) return;
    groups[ind] = groups[ind] || {};
    groups[ind][date] = groups[ind][date] || [];
    groups[ind][date].push(c.predicted_proba);
  });
  // pick top 10 industries by company count
  const topInds = Object.entries(groups)
    .map(([k,v]) => [k, Object.values(v).reduce((s,arr) => s + arr.length, 0)])
    .sort((a,b) => b[1] - a[1]).slice(0, 10).map(t => t[0]);

  const palette = ['#7c8cff','#9f7aea','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#a3e635','#fb923c','#60a5fa'];
  const traces = topInds.map((ind, i) => {
    const dates = Object.keys(groups[ind]).sort();
    return {
      type:'scatter', mode:'lines+markers', name: ind,
      x: dates,
      y: dates.map(d => {
        const a = groups[ind][d]; return a.reduce((s,v) => s+v, 0) / a.length;
      }),
      line: { color: palette[i % palette.length], width: 2 },
      marker: { size: 6 },
    };
  });
  Plotly.newPlot('chart-trend-industry', traces, {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'As-of date' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Mean default probability', tickformat:'.0%' },
    legend: { ...PL_LAYOUT.legend, orientation:'h', y:-0.25 },
  }, PL_CONFIG);
}

// ============ Scenarios ============
function renderScenarioCards(rows) {
  const sorted = [...rows].sort((a,b) => (b.mean_default_proba||0) - (a.mean_default_proba||0));
  $('#scenario-cards').innerHTML = sorted.map(s => {
    const cls = s.mean_default_proba > 0.5 ? 'bad' : s.mean_default_proba > 0.25 ? 'warn' : 'good';
    return `
      <div class="kpi ${cls}">
        <div class="lbl">${s.scenario.replace(/_/g, ' ')}</div>
        <div class="val">${fmt.pct(s.mean_default_proba)}</div>
        <div class="sub">P95: ${fmt.pct(s.p95_default_proba)} · ${s.companies_at_risk}/${s.company_count} at risk</div>
      </div>`;
  }).join('');
}

function renderScenariosChart(rows) {
  const sorted = [...rows].sort((a,b) => (a.mean_default_proba||0) - (b.mean_default_proba||0));
  Plotly.newPlot('chart-scenarios', [
    { type:'bar', orientation:'h', name:'Mean',
      x: sorted.map(r => r.mean_default_proba),
      y: sorted.map(r => r.scenario.replace(/_/g, ' ')),
      marker: { color: sorted.map(r => r.mean_default_proba),
                colorscale: [[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']], showscale:false },
      hovertemplate: '<b>%{y}</b><br>Mean: %{x:.2%}<extra></extra>',
    },
    { type:'scatter', mode:'markers', name:'P95',
      x: sorted.map(r => r.p95_default_proba),
      y: sorted.map(r => r.scenario.replace(/_/g, ' ')),
      marker: { color:'#ef4444', symbol:'diamond', size: 10, line: { color:'#fff', width: 1 } },
      hovertemplate: '<b>%{y}</b><br>P95: %{x:.2%}<extra></extra>',
    },
  ], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Default probability', tickformat:'.0%' },
    yaxis: { ...PL_LAYOUT.yaxis, automargin: true },
    legend: { ...PL_LAYOUT.legend, orientation:'h', y:-0.18 },
    margin: { t: 8, r: 18, b: 60, l: 10 },
  }, PL_CONFIG);
}

function renderScenariosTable(rows) {
  const sorted = [...rows].sort((a,b) => (b.mean_default_proba||0) - (a.mean_default_proba||0));
  $('#scenarios-tbody').innerHTML = sorted.map(r => `
    <tr>
      <td>${r.scenario.replace(/_/g, ' ')}</td>
      <td class="muted small">${r.description || ''}</td>
      <td class="num">${fmt.pct(r.mean_default_proba)}</td>
      <td class="num">${fmt.pct(r.median_default_proba)}</td>
      <td class="num">${fmt.pct(r.p95_default_proba)}</td>
      <td class="num">${r.companies_at_risk} / ${r.company_count}</td>
    </tr>
  `).join('');
}

// ============ VaR ============
async function renderVarPanel(returns) {
  const methods = ['historical','parametric','monte_carlo'];
  const confs   = [0.90, 0.95, 0.99];
  const grid = await Promise.all(methods.flatMap(m =>
    confs.map(c => fetchJSON(`/var?method=${m}&confidence=${c}`))
  ));

  // cards: 95% across methods
  $('#var-cards').innerHTML = methods.map((m,i) => {
    const v = grid[i*3 + 1]; // 95%
    return `
      <div class="kpi bad">
        <div class="lbl">${m.replace('_',' ')} VaR · 95%</div>
        <div class="val">${fmt.pct(v.value_at_risk, 2)}</div>
        <div class="sub">μ = ${fmt.pct(v.mean_return, 2)} · σ = ${fmt.pct(v.std_return, 2)}</div>
      </div>`;
  }).join('');

  // heatmap matrix
  const z = methods.map((m,i) => confs.map((c,j) => grid[i*3+j].value_at_risk));
  Plotly.newPlot('chart-var', [{
    type:'heatmap',
    x: confs.map(c => `${(c*100).toFixed(0)}%`),
    y: methods.map(m => m.replace('_',' ')),
    z,
    colorscale: [[0,'#10b981'],[0.5,'#f59e0b'],[1,'#ef4444']],
    text: z.map(row => row.map(v => (v*100).toFixed(2) + '%')),
    texttemplate: '%{text}',
    textfont: { color:'#fff', size: 14 },
    hovertemplate: 'Method: %{y}<br>Confidence: %{x}<br>VaR: %{text}<extra></extra>',
    colorbar: { tickformat: '.1%', tickfont: { color:'#cbd5e1' } },
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Confidence level' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Method' },
  }, PL_CONFIG);

  // returns histogram with VaR line
  const var95 = returns.var_95_historical;
  Plotly.newPlot('chart-var-hist', [{
    type:'histogram', x: returns.returns, nbinsx: 30,
    marker: { color:'#7c8cff' },
    hovertemplate: 'Return: %{x:.2%}<br>Count: %{y}<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Daily portfolio return', tickformat:'.1%' },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Days' },
    shapes: [{ type:'line', x0: -var95, x1: -var95, y0:0, y1:1, yref:'paper',
               line: { color:'#ef4444', width: 2, dash:'dash' } }],
    annotations: [{ x: -var95, y: 1, yref:'paper', text:`VaR 95%: ${fmt.pct(var95, 2)}`,
                    showarrow:false, font:{color:'#ef4444'}, xanchor:'left' }],
  }, PL_CONFIG);
}

// ============ Model quality ============
const MODEL_COLORS = { xgboost:'#7c8cff', logistic:'#f59e0b' };
const modelLabel = m => m === 'xgboost' ? 'XGBoost' : m === 'logistic' ? 'Logistic Regression' : m;

function renderModelPanel(data) {
  const active = data.active_model;
  const m = data.models[active];
  if (!m) return;

  // KPI cards
  const kpis = [
    { lbl:'ROC-AUC', val: fmt.num(m.roc_auc, 3),
      sub:`${modelLabel(active)} · test n=${data.data.n_test}`, cls: m.roc_auc > 0.8 ? 'good' : m.roc_auc > 0.65 ? 'warn' : 'bad' },
    { lbl:'PR-AUC', val: fmt.num(m.pr_auc, 3),
      sub:`Base rate ${fmt.pct(data.data.positive_rate)}`, cls: m.pr_auc > data.data.positive_rate * 2 ? 'good' : 'warn' },
    { lbl:'Recall / Precision', val:`${fmt.pct(m.recall)} / ${fmt.pct(m.precision)}`,
      sub:`F1 ${fmt.num(m.f1, 3)} · threshold 0.5`, cls: m.recall > 0.7 ? 'good' : m.recall > 0.5 ? 'warn' : 'bad' },
    { lbl:'Brier score', val: fmt.num(m.brier_score, 3),
      sub:`Calibration · lower is better`, cls: m.brier_score < 0.15 ? 'good' : m.brier_score < 0.25 ? 'warn' : 'bad' },
  ];
  $('#model-kpis').innerHTML = kpis.map(k => `
    <div class="kpi ${k.cls||''}">
      <div class="lbl">${k.lbl}</div>
      <div class="val">${k.val}</div>
      <div class="sub">${k.sub}</div>
    </div>`).join('');

  const names = Object.keys(data.models);

  // ROC curves (all models) + chance diagonal
  const rocTraces = names.map(n => ({
    type:'scatter', mode:'lines', name:`${modelLabel(n)} (AUC ${data.models[n].roc_auc.toFixed(3)})`,
    x: data.models[n].roc_curve.fpr, y: data.models[n].roc_curve.tpr,
    line: { color: MODEL_COLORS[n] || '#9f7aea', width: n === active ? 3 : 2 },
  }));
  rocTraces.push({ type:'scatter', mode:'lines', name:'Chance',
    x:[0,1], y:[0,1], line:{ color:'#64748b', dash:'dash', width:1 }, hoverinfo:'skip' });
  Plotly.newPlot('chart-roc', rocTraces, {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'False positive rate', range:[0,1] },
    yaxis: { ...PL_LAYOUT.yaxis, title:'True positive rate', range:[0,1.02] },
    legend: { ...PL_LAYOUT.legend, orientation:'h', y:-0.2 },
  }, PL_CONFIG);

  // PR curves + base-rate line
  const prTraces = names.map(n => ({
    type:'scatter', mode:'lines', name:`${modelLabel(n)} (AP ${data.models[n].pr_auc.toFixed(3)})`,
    x: data.models[n].pr_curve.recall, y: data.models[n].pr_curve.precision,
    line: { color: MODEL_COLORS[n] || '#9f7aea', width: n === active ? 3 : 2 },
  }));
  prTraces.push({ type:'scatter', mode:'lines', name:'Base rate',
    x:[0,1], y:[data.data.positive_rate, data.data.positive_rate],
    line:{ color:'#64748b', dash:'dash', width:1 }, hoverinfo:'skip' });
  Plotly.newPlot('chart-pr', prTraces, {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Recall', range:[0,1] },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Precision', range:[0,1.02] },
    legend: { ...PL_LAYOUT.legend, orientation:'h', y:-0.2 },
  }, PL_CONFIG);

  // Calibration (active model)
  Plotly.newPlot('chart-calibration', [
    { type:'scatter', mode:'lines+markers', name: modelLabel(active),
      x: m.calibration.mean_predicted, y: m.calibration.fraction_positive,
      line:{ color: MODEL_COLORS[active] || '#7c8cff', width: 2 }, marker:{ size: 7 } },
    { type:'scatter', mode:'lines', name:'Perfect',
      x:[0,1], y:[0,1], line:{ color:'#64748b', dash:'dash', width:1 }, hoverinfo:'skip' },
  ], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Mean predicted probability', range:[0,1] },
    yaxis: { ...PL_LAYOUT.yaxis, title:'Observed default rate', range:[0,1] },
    showlegend: false,
  }, PL_CONFIG);

  // Confusion matrix (active model)
  const cm = m.confusion_matrix;
  Plotly.newPlot('chart-confusion', [{
    type:'heatmap',
    x:['Pred. safe','Pred. risky'], y:['Actually risky','Actually safe'],
    z: [[cm.fn, cm.tp],[cm.tn, cm.fp]],
    colorscale: [[0,'#141822'],[1,'#7c8cff']], showscale: false,
    text: [[`FN ${cm.fn}`,`TP ${cm.tp}`],[`TN ${cm.tn}`,`FP ${cm.fp}`]],
    texttemplate:'%{text}', textfont:{ color:'#fff', size:15 },
    hovertemplate:'%{y} · %{x}: %{text}<extra></extra>',
  }], { ...PL_LAYOUT, margin:{ t:8, r:18, b:48, l:110 } }, PL_CONFIG);

  // Feature importance (active model)
  const imp = (data.feature_importance[active] || []).slice().reverse();
  Plotly.newPlot('chart-importance', [{
    type:'bar', orientation:'h',
    x: imp.map(d => d.importance),
    y: imp.map(d => d.feature),
    marker:{ color:'#9f7aea' },
    hovertemplate:'<b>%{y}</b><br>Share: %{x:.1%}<extra></extra>',
  }], {
    ...PL_LAYOUT,
    xaxis: { ...PL_LAYOUT.xaxis, title:'Importance share', tickformat:'.0%' },
    yaxis: { ...PL_LAYOUT.yaxis, automargin: true },
    margin: { t:8, r:18, b:48, l:10 },
  }, PL_CONFIG);

  // Comparison table
  $('#model-tbody').innerHTML = names.map(n => {
    const r = data.models[n];
    return `
      <tr>
        <td>${modelLabel(n)}${n === active ? ' <span class="pill pill-A">active</span>' : ''}</td>
        <td class="num mono">${fmt.num(r.roc_auc, 3)}</td>
        <td class="num mono">${fmt.num(r.pr_auc, 3)}</td>
        <td class="num mono">${fmt.num(r.brier_score, 3)}</td>
        <td class="num">${fmt.pct(r.accuracy)}</td>
        <td class="num">${fmt.pct(r.precision)}</td>
        <td class="num">${fmt.pct(r.recall)}</td>
        <td class="num">${fmt.pct(r.f1)}</td>
      </tr>`;
  }).join('');
}

// ============ Detail modal ============
window.openCompany = async function(id) {
  try {
    showLoading('Loading company…');
    const c = await fetchJSON('/company/' + encodeURIComponent(id));
    $('#modal-title').innerHTML = `${c.company_name || c.company_id} ${ratingPill(c.risk_rating)}`;
    const section = (title, items) => `
      <div class="section">
        <h4>${title}</h4>
        <dl>${items.filter(([,v]) => v != null && v !== '').map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
        <dl></dl>
      </div>`;

    $('#modal-body').innerHTML = `
      <div class="kv">
        ${section('Identity', [
          ['Symbol', c.symbol],
          ['Industry', c.industry],
          ['Sector', c.sector],
          ['Region', c.region],
          ['As of', fmt.date(c.as_of_date)],
        ])}
        ${section('Financials', [
          ['Revenue', fmt.money(c.revenue)],
          ['EBITDA', fmt.money(c.ebitda)],
          ['Net income', fmt.money(c.net_income)],
          ['Total assets', fmt.money(c.total_assets)],
          ['Total equity', fmt.money(c.total_equity)],
          ['Debt', fmt.money(c.debt)],
          ['Cash', fmt.money(c.cash)],
        ])}
        ${section('Ratios', [
          ['Leverage', fmt.num(c.leverage_ratio)],
          ['Debt / Equity', fmt.num(c.debt_to_equity)],
          ['Current ratio', fmt.num(c.current_ratio)],
          ['Quick ratio', fmt.num(c.quick_ratio)],
          ['Interest coverage', fmt.num(c.interest_coverage)],
          ['EBITDA margin', fmt.pct(c.ebitda_margin)],
          ['Net margin', fmt.pct(c.net_margin)],
          ['ROA', fmt.pct(c.roa)],
          ['ROE', fmt.pct(c.roe)],
          ['Altman Z', fmt.num(c.altman_z_score)],
        ])}
        ${section('Risk scores', [
          ['Predicted default prob.', fmt.pct(c.predicted_proba)],
          ['Risk rating', c.risk_rating],
          ['Financial score', fmt.num(c.financial_score, 3)],
          ['Operational score', fmt.num(c.operational_score, 3)],
          ['Compliance score', fmt.num(c.compliance_score, 3)],
          ['Aggregated risk', fmt.num(c.aggregated_risk, 3)],
        ])}
      </div>`;
    $('#modal').classList.remove('hidden');
  } catch (e) {
    alert('Failed to load company: ' + e.message);
  } finally {
    hideLoading();
  }
};

window.closeModal = function () { $('#modal').classList.add('hidden'); };
