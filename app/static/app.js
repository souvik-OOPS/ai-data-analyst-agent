"use strict";

/* ------------------------------------------------------------------ state & helpers */

const state = {
  info: null,
  range: { preset: "all", start: null, end: null },
  history: [],
  plots: new Map(), // plot element -> chart spec (for theme re-render)
  pending: false,
};

const EXAMPLES = [
  "What were the top 5 products by revenue?",
  "Show monthly revenue.",
  "Which customers have not purchased recently?",
  "Which category generated the most revenue?",
  "Why did revenue decline last month?",
  "Show me monthly revenue as a chart.",
];
const UNSUPPORTED_EXAMPLE = "What is our profit margin?";

const PRESETS = [
  { id: "all", label: "All data", months: null },
  { id: "12m", label: "12 months", months: 12 },
  { id: "6m", label: "6 months", months: 6 },
  { id: "3m", label: "3 months", months: 3 },
  { id: "1m", label: "Last month", months: 1 },
];

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Could not reach the server.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error?.message || `Request failed (HTTP ${res.status})`);
  return body;
}

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const usd2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const usdCompact = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

function money(v, { compact = false } = {}) {
  if (v === null || v === undefined) return "–";
  if (compact && Math.abs(v) >= 100000) return usdCompact.format(v);
  return Math.abs(v) >= 1000 ? usd.format(v) : usd2.format(v);
}

function monthLabel(ym, long = false) {
  const [y, m] = ym.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-US", { month: long ? "long" : "short", year: "numeric", timeZone: "UTC" });
}

function shiftMonth(ym, delta) {
  const [y, m] = ym.split("-").map(Number);
  const idx = y * 12 + (m - 1) + delta;
  return `${Math.floor(idx / 12)}-${String((idx % 12) + 1).padStart(2, "0")}`;
}

function monthBounds(ym) {
  const [y, m] = ym.split("-").map(Number);
  const last = new Date(Date.UTC(y, m, 0)).getUTCDate();
  return [`${ym}-01`, `${ym}-${String(last).padStart(2, "0")}`];
}

function rangeQuery(extra = {}) {
  const params = new URLSearchParams(extra);
  if (state.range.start) params.set("start_date", state.range.start);
  if (state.range.end) params.set("end_date", state.range.end);
  const s = params.toString();
  return s ? `?${s}` : "";
}

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(text));
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, "<br>");
}

/* ------------------------------------------------------------------ theme */

function storedTheme() {
  try { return localStorage.getItem("theme"); } catch { return null; }
}

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function effectiveTheme() {
  return document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch { /* storage unavailable */ }
  redrawPlots();
}

/* ------------------------------------------------------------------ charts */

function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    surface: v("--surface-1"), text: v("--text-secondary"), primary: v("--text-primary"), muted: v("--text-muted"),
    grid: v("--grid"), baseline: v("--baseline"), border: v("--border"), font: v("--font-sans"),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v(`--series-${i}`)),
  };
}

function withAlpha(hex, alpha) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function valueFormat(spec) {
  if (spec.value_format !== "currency") return ",.0f";
  return spec.chart === "aov_trend" ? "$,.2f" : "$,.0f";
}

function figure(spec, t) {
  const fmt = valueFormat(spec);
  const currency = spec.value_format === "currency";
  const axisFont = { color: t.muted, size: 11 };
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: t.font, size: 12, color: t.text },
    margin: { l: 8, r: 16, t: 8, b: 8 },
    showlegend: spec.series.length > 1 || spec.type === "pie",
    legend: { orientation: "h", x: 0, y: 1.14, font: { color: t.text } },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.baseline, font: { color: t.primary, family: t.font } },
    bargap: 0.35, barcornerradius: 4,
  };
  let traces;
  let height = 260;

  if (spec.type === "pie") {
    const s = spec.series[0];
    traces = [{
      type: "pie", labels: s.x, values: s.y, hole: 0.55, sort: false, direction: "clockwise",
      marker: { colors: t.series, line: { color: t.surface, width: 2 } },
      textinfo: "percent", textfont: { color: "#ffffff" }, insidetextorientation: "horizontal",
      hovertemplate: `%{label}<br>%{value:${fmt}} (%{percent})<extra></extra>`,
    }];
    layout.legend = { orientation: "v", x: 1, y: 0.5, font: { color: t.text } };
  } else if (spec.orientation === "h") {
    const s = spec.series[0];
    traces = [{
      type: "bar", orientation: "h", x: s.y, y: s.x, marker: { color: t.series[0] },
      hovertemplate: `%{y}<br>%{x:${fmt}}<extra></extra>`,
    }];
    layout.yaxis = { autorange: "reversed", automargin: true, ticks: "", ticksuffix: "  ",
      tickfont: { color: t.text, size: 11 } };
    layout.xaxis = { tickprefix: currency ? "$" : "", tickformat: "~s", gridcolor: t.grid, zeroline: false,
      showline: false, tickfont: axisFont, rangemode: "tozero", automargin: true };
    layout.margin.b = 28;
    height = Math.max(220, s.x.length * 28 + 70);
  } else {
    traces = spec.series.map((s, i) => {
      const color = t.series[i % t.series.length];
      if (spec.type === "bar") {
        return { type: "bar", name: s.name, x: s.x, y: s.y, marker: { color },
          hovertemplate: `%{y:${fmt}}<extra>${s.name}</extra>` };
      }
      const last = s.x.length - 1;
      return {
        type: "scatter", mode: "lines+markers", name: s.name, x: s.x, y: s.y, cliponaxis: false,
        line: { color, width: 2 },
        marker: { color, size: s.x.map((_, j) => (j === last ? 9 : 0)), line: { color: t.surface, width: 2 } },
        fill: spec.series.length === 1 ? "tozeroy" : "none", fillcolor: withAlpha(color, 0.1),
        hovertemplate: `%{y:${fmt}}<extra>${s.name}</extra>`,
      };
    });
    layout.xaxis = { type: "date", tickformat: "%b %Y", hoverformat: "%B %Y", showgrid: false, linecolor: t.baseline,
      ticks: "outside", tickcolor: t.baseline, tickfont: axisFont, automargin: true };
    layout.yaxis = { tickprefix: currency ? "$" : "", tickformat: "~s", gridcolor: t.grid, zeroline: false,
      rangemode: "tozero", tickfont: axisFont, automargin: true };
    layout.hovermode = "x unified";
    if (layout.showlegend) layout.margin.t = 28;
  }
  layout.height = height;
  return { traces, layout };
}

function drawPlot(plot, spec) {
  if (!window.Plotly) {
    plot.replaceChildren(el("p", { class: "chart-error" }, "Charting library failed to load (check your internet connection)."));
    return;
  }
  const { traces, layout } = figure(spec, tokens());
  Plotly.react(plot, traces, layout, { displayModeBar: false, responsive: true });
  state.plots.set(plot, spec);
}

function redrawPlots() {
  for (const [plot, spec] of state.plots) {
    if (plot.isConnected) drawPlot(plot, spec);
    else state.plots.delete(plot);
  }
}

function dataTable(spec) {
  const time = spec.orientation === "v" && spec.type !== "pie";
  const fmtValue = (v) => (spec.value_format === "currency" ? money(v) : int.format(v));
  const head = [time ? "Month" : spec.y_label, ...spec.series.map((s) => s.name)];
  const rows = spec.series[0].x.map((x, i) => [time ? monthLabel(x) : x, ...spec.series.map((s) => fmtValue(s.y[i]))]);
  const table = el("table", {},
    el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: i ? "num" : null }, h)))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i ? "num" : null }, c))))));
  return el("details", { class: "data-view" }, el("summary", {}, "View data"), el("div", { class: "table-wrap" }, table));
}

function renderChartCard(card, spec) {
  const plot = el("div", { class: "plot", role: "img", "aria-label": `${spec.title} chart` });
  card.replaceChildren(
    el("div", { class: "chart-title" }, el("h2", {}, spec.title), el("p", { class: "muted small" }, spec.subtitle)),
    plot,
    dataTable(spec),
  );
  drawPlot(plot, spec);
}

/* ------------------------------------------------------------------ dashboard */

function buildPresets() {
  const wrap = $("#presets");
  wrap.replaceChildren(...PRESETS.map((p) =>
    el("button", { type: "button", "data-preset": p.id, "aria-pressed": String(p.id === state.range.preset),
      onclick: () => selectPreset(p.id) }, p.label)));
}

function selectPreset(id) {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset.months) {
    state.range = { preset: id, start: null, end: null };
  } else {
    const latest = state.info.latest_full_month;
    state.range = { preset: id, start: monthBounds(shiftMonth(latest, -(preset.months - 1)))[0], end: monthBounds(latest)[1] };
  }
  $("#start-date").value = state.range.start || "";
  $("#end-date").value = state.range.end || "";
  document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.preset === id)));
  loadDashboard();
}

function renderKpis(data) {
  const cur = data.current;
  const values = {
    revenue: money(cur.revenue, { compact: true }),
    orders: int.format(cur.orders),
    customers: int.format(cur.customers),
    average_order_value: money(cur.average_order_value),
  };
  for (const tile of document.querySelectorAll(".kpi")) {
    const key = tile.dataset.kpi;
    const value = $(".kpi-value", tile);
    value.textContent = values[key];
    value.title = key === "revenue" ? money(cur.revenue) : "";
    const delta = $(".kpi-delta", tile);
    const change = data.change_pct ? data.change_pct[key] : null;
    if (change === null || change === undefined) {
      delta.replaceChildren(document.createTextNode(data.previous ? "No change data" : "No comparable previous period"));
    } else {
      const up = change >= 0;
      delta.replaceChildren(
        el("span", { class: up ? "up" : "down" }, `${up ? "▲" : "▼"} ${Math.abs(change).toFixed(1)}%`),
        " vs previous period");
    }
  }
  const p = data.current.period;
  let label = p.start_date ? `Showing ${p.start_date} to ${p.end_date}` : `Showing all data (${p.data_from} to ${p.data_to})`;
  if (data.previous_period) label += ` · compared with ${data.previous_period.start_date} to ${data.previous_period.end_date}`;
  $("#range-label").textContent = `${label} · completed orders only`;
}

async function loadDashboard() {
  const cards = ["monthly_revenue", "revenue_by_category", "top_products", "customer_trend"];
  document.querySelectorAll(".chart-card, .kpi").forEach((c) => c.classList.add("is-loading"));
  const kpiPromise = api(`/api/kpis${rangeQuery()}`)
    .then(renderKpis)
    .catch((e) => { $("#range-label").textContent = `Could not load KPIs: ${e.message}`; });
  const chartPromises = cards.map((name) => {
    const card = $(`#card-${name}`);
    const extra = name === "top_products" ? { limit: 10 } : {};
    return api(`/api/charts/${name}${rangeQuery(extra)}`)
      .then((spec) => renderChartCard(card, spec))
      .catch((e) => card.replaceChildren(el("p", { class: "chart-error" }, e.message)));
  });
  await Promise.all([kpiPromise, ...chartPromises]);
  document.querySelectorAll(".is-loading").forEach((c) => c.classList.remove("is-loading"));
}

async function loadInsights() {
  const list = $("#insights");
  try {
    const data = await api("/api/insights?ai=true");
    $("#insights-engine").textContent = data.engine === "llm" ? "Written by AI · numbers verified" : "Computed from data";
    list.replaceChildren(...data.insights.map((i) => {
      const icon = { positive: "▲", negative: "▼", neutral: "●" }[i.sentiment] || "●";
      return el("li", { class: "insight" },
        el("span", { class: `icon ${i.sentiment}`, "aria-label": i.sentiment }, icon),
        el("div", {}, el("h3", {}, i.title), el("p", {}, i.text)));
    }));
    const warn = $("#insights-warning");
    warn.hidden = !data.warnings.length;
    warn.textContent = data.warnings.join(" ");
  } catch (e) {
    list.replaceChildren(el("li", { class: "muted" }, `Insights unavailable: ${e.message}`));
  }
}

function renderQuality(report) {
  const box = $("#quality");
  if (!report) { box.textContent = "No ingestion report available."; return; }
  const sum = (o) => Object.values(o || {}).reduce((a, b) => a + b, 0);
  const rows = Object.entries(report.tables).map(([name, t]) => [
    name, int.format(t.rows_read), int.format(t.duplicates_removed), int.format(sum(t.dropped)),
    int.format(sum(t.imputed)), int.format(sum(t.normalized)), int.format(t.rows_loaded)]);
  const head = ["Table", "Rows read", "Duplicates removed", "Invalid rows dropped", "Values filled", "Values normalised", "Rows loaded"];
  const table = el("table", {},
    el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: i ? "num" : null }, h)))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i ? "num" : null }, c))))));
  const details = [];
  for (const [name, t] of Object.entries(report.tables)) {
    for (const [kind, obj] of [["dropped", t.dropped], ["filled", t.imputed], ["normalised", t.normalized]]) {
      for (const [reason, n] of Object.entries(obj || {})) details.push(el("li", {}, `${name}: ${n} ${kind} - ${reason.replaceAll("_", " ")}`));
    }
  }
  box.replaceChildren(el("div", { class: "table-wrap" }, table), el("ul", {}, details));
}

/* ------------------------------------------------------------------ chat */

function scrollChat() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

function addMessage(node) {
  $("#messages").append(node);
  scrollChat();
  return node;
}

function typingIndicator() {
  const label = el("span", {}, "Analysing");
  const node = el("div", { class: "msg assistant" }, el("div", { class: "typing" }, label, el("i"), el("i"), el("i")));
  const started = Date.now();
  const timer = setInterval(() => { label.textContent = `Analysing · ${Math.round((Date.now() - started) / 1000)}s`; }, 1000);
  node.stop = () => { clearInterval(timer); node.remove(); };
  return addMessage(node);
}

function answerMeta(data) {
  const badges = [];
  badges.push(el("span", { class: "badge" }, data.engine === "llm" ? `AI · ${data.model}` : "Rule-based engine"));
  if (data.status === "unsupported") badges.push(el("span", { class: "badge" }, "Unsupported question"));
  if (data.status === "error") badges.push(el("span", { class: "badge bad" }, "Could not answer"));
  const g = data.grounding;
  if (g) {
    if (g.grounded) {
      badges.push(el("span", { class: "badge ok", title: "Every number in the answer matched a tool result" },
        g.checked_numbers ? `✓ ${g.verified_numbers} figure${g.verified_numbers === 1 ? "" : "s"} verified` : "✓ No figures to verify"));
    } else {
      badges.push(el("span", { class: "badge bad", title: g.unverified_numbers.join(", ") },
        `⚠ ${g.unverified_numbers.length} unverified figure${g.unverified_numbers.length === 1 ? "" : "s"}`));
    }
  }
  badges.push(el("span", {}, `${(data.latency_ms / 1000).toFixed(1)} s`));
  return el("div", { class: "meta" }, badges);
}

function toolSteps(calls) {
  if (!calls.length) return null;
  const items = calls.map((c) => el("li", {},
    el("code", {}, c.tool), ` ${c.ok ? "✓" : "✗"} · ${c.duration_ms} ms`,
    el("pre", {}, `input: ${JSON.stringify(c.input)}`),
    c.ok ? el("pre", {}, c.output_preview || "") : el("div", { class: "tool-err" }, c.error)));
  return el("details", { class: "steps" },
    el("summary", {}, `Analysis steps · ${calls.length} tool call${calls.length === 1 ? "" : "s"}`),
    el("ol", {}, items));
}

function addAnswer(data) {
  const body = el("div", { class: "body" });
  body.innerHTML = renderMarkdown(data.answer);
  const msg = el("div", { class: `msg assistant${data.status === "error" ? " error" : ""}` }, body);
  for (const w of data.warnings) msg.append(el("p", { class: "warn" }, w));
  addMessage(msg);
  for (const spec of data.charts) {
    const card = el("div", { class: "card chart-card" });
    msg.append(card);
    renderChartCard(card, spec);
  }
  msg.append(answerMeta(data));
  const steps = toolSteps(data.tool_calls);
  if (steps) msg.append(steps);
  scrollChat();
}

async function ask(question) {
  question = question.trim();
  if (!question || state.pending) return;
  state.pending = true;
  $("#ask-btn").disabled = true;
  addMessage(el("div", { class: "msg user" }, question));
  const typing = typingIndicator();
  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: state.history.slice(-4) }),
    });
    typing.stop();
    addAnswer(data);
    if (data.status === "answered") state.history.push({ question, answer: data.answer.slice(0, 8000) });
  } catch (e) {
    typing.stop();
    addMessage(el("div", { class: "msg assistant error" }, el("div", { class: "body" }, e.message)));
  } finally {
    state.pending = false;
    $("#ask-btn").disabled = false;
    $("#question").focus();
  }
}

function resetChat() {
  state.history = [];
  const box = $("#messages");
  [...box.children].slice(1).forEach((n) => n.remove());
}

/* ------------------------------------------------------------------ init */

async function init() {
  applyTheme(storedTheme());
  $("#theme-toggle").addEventListener("click", toggleTheme);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!storedTheme()) redrawPlots(); });

  $("#examples").replaceChildren(
    ...EXAMPLES.map((q) => el("button", { type: "button", class: "chip", onclick: () => ask(q) }, q)),
    el("button", { type: "button", class: "chip alt", title: "Shows how unsupported questions are handled",
      onclick: () => ask(UNSUPPORTED_EXAMPLE) }, UNSUPPORTED_EXAMPLE));
  $("#ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const box = $("#question");
    const q = box.value;
    box.value = "";
    ask(q);
  });
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
  });
  $("#clear-chat").addEventListener("click", resetChat);
  $("#custom-range").addEventListener("submit", (e) => {
    e.preventDefault();
    const start = $("#start-date").value || null;
    const end = $("#end-date").value || null;
    if (start && end && start > end) { $("#range-label").textContent = "Start date must be on or before end date."; return; }
    state.range = { preset: "custom", start, end };
    document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    loadDashboard();
  });

  try {
    const [health, dataset] = await Promise.all([api("/api/health"), api("/api/dataset")]);
    state.info = dataset.coverage;
    const c = dataset.coverage;
    $("#coverage").textContent = `Data: ${monthLabel(c.first_order_date.slice(0, 7))} – ${monthLabel(c.last_order_date.slice(0, 7))} · ${int.format(c.orders)} orders`;
    const mode = $("#mode");
    if (health.llm.mode === "llm") {
      mode.textContent = `AI: ${health.llm.provider} (${health.llm.model})`;
      mode.className = "pill llm";
    } else {
      mode.textContent = "Offline mode · rule-based";
      mode.className = "pill offline";
      mode.title = "Set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env to enable the AI model";
    }
    $("#start-date").min = $("#end-date").min = c.first_order_date;
    $("#start-date").max = $("#end-date").max = c.last_order_date;
    buildPresets();
    renderQuality(dataset.ingestion_report);
    loadDashboard();
    loadInsights();
    // Deep link: /?ask=Your+question asks it on load (handy for demos and sharing).
    const preset = new URLSearchParams(location.search).get("ask");
    if (preset) ask(preset);
  } catch (e) {
    $("#coverage").textContent = `Backend unavailable: ${e.message}`;
  }
}

document.addEventListener("DOMContentLoaded", init);
