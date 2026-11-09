# AI Data Analyst & Business Intelligence Agent

An internal AI business analyst for a fictional online retailer, **DDL Labs**. Business users ask questions in plain English, such as *"What were the top 5 products by revenue?"* or *"Why did revenue decline last month?"*. The agent picks the right analysis tools, computes the answer from the actual order database, checks every number in its reply against the data, and shows charts when asked. A dashboard shows the core KPIs, charts, and data-backed insights.

| | |
|---|---|
| **LLM** | **Google Gemini 3.5 Flash** (`gemini-3.5-flash`, with automatic failover to other Gemini models) via Google's official `google-genai` SDK. **Anthropic Claude Opus 5.5** (`claude-opus-5-5`) is also supported via the `anthropic` SDK. |
| **Database** | **SQLite** (explicit schema, constraints, indexes and analytical views) |
| **Backend** | Python 3.11+, **FastAPI**, Pydantic, pandas (ingestion) |
| **Frontend** | Plain HTML/CSS/JS served by FastAPI, **Plotly.js** charts |
| **Tests** | pytest: 176 offline tests + 5 optional live-API tests |

---

## Quick start

Requires Python 3.11+ (developed on 3.12). No API key is needed to try it: without one the app runs in **offline mode** with a rule-based engine.

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # optional: add GEMINI_API_KEY (or ANTHROPIC_API_KEY) to enable the AI model
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** for the dashboard, or **http://127.0.0.1:8000/docs** for the interactive API docs.

On first start, the SQLite database (`data/ecommerce.db`) is built automatically from the CSV files in `data/raw/`. To rebuild it manually and print the data-quality report, run `python -m app.ingestion`. To regenerate the sample CSVs (deterministic, seed 42), run `python scripts/generate_data.py`.

**Docker (optional):** `docker build -t ai-analyst .` then `docker run -p 8000:8000 ai-analyst` (offline mode), or add `--env-file .env` to pass your API key.

**Run the tests:**

```bash
pytest              # 176 tests, ~15 s, no network or API key needed
pytest -m live      # 5 end-to-end tests against the real LLM API configured in .env
```

---

## Features

**Data ingestion and quality.** The raw extracts deliberately contain the problems real exports have. Each one is handled and counted in a report, visible in the dashboard under *Data quality report* and at `GET /api/dataset`:

| Problem in raw data | Handling |
|---|---|
| Exact and key-level duplicates (15 customers, 40 orders, 60 order lines) | Removed; first occurrence kept |
| Inconsistent casing/whitespace (`"  alice "`, `"COMPLETED"`, `"canceled"`) | Normalised; status synonyms mapped |
| Prices exported as text (`"$1,299.00"`) | Parsed to numbers |
| Invalid dates (`2025-02-30`, `N/A`) | Rows dropped (an order without a valid date cannot be placed in time) |
| Missing unit price / discount | Filled from the product list price / 0 |
| Missing region, category, channel | Filled with `Unknown` / `Uncategorized` / `unknown` and reported |
| Missing or negative quantities, unknown products, orders for unknown customers | Rows dropped, along with orders left with no valid lines |

The database is written to a temporary file and swapped in atomically, then checked for foreign-key violations, so a failed rebuild never leaves a half-built database behind.

**Analytics** (`app/analytics/`), all computed with SQL:

- Sales: revenue, orders, AOV, units, monthly trend, revenue by category, region and channel
- Customers: total, active, new, returning, repeat (with repeat-revenue share), inactive, top customers
- Products: top sellers by units, highest revenue, low performers (including products that never sold)

**AI analyst.** Natural-language questions are answered by the LLM (Gemini or Claude) using 13 controlled tools; see [AI and agent approach](#ai-and-agent-approach).

**Multi-step analysis.** `analyze_revenue_change` investigates *why* revenue moved between two months in six steps:
1. Compare headline KPIs (revenue, orders, AOV, units, customers, cancellations).
2. Split the revenue change into an order-volume effect and an order-value effect.
3. Measure each category's contribution to the change.
4. Rank product movers and flag products whose sales nearly stopped.
5. Compare new and returning customers.
6. Measure each region's contribution.

It also checks the question's premise. If revenue did not actually decline, it says so instead of explaining a decline that didn't happen.

**Dynamic charts.** "Show me monthly revenue as a chart" makes the agent call `generate_chart`. The backend queries the data and returns a chart spec, which the browser renders with Plotly. Every chart has a *View data* table.

**Dashboard.** KPI tiles (Revenue, Orders, Customers, AOV) with change against the previous period; monthly revenue, revenue by category, top products and new-vs-returning customer charts; date presets plus a custom date range; light and dark themes; a layout that works on phones.

**AI-generated insights.** Seven insights are computed from the data, each carrying its evidence. With a key, the LLM rewrites them for executives (structured JSON output). A rewrite is kept only if every number in it matches that insight's evidence; otherwise the computed version is shown.

### The sample data and the story to find

`data/raw/` holds a synthetic but realistic dataset: 2,745 customers, 50 products in 7 categories, and 10,940 orders (18,166 order lines) from **Nov 2024 to Oct 2026**, with seasonality, growth, churn and a holiday peak. It also contains one planted business problem for the agent to diagnose:

> In **October 2026** revenue fell **27.8%** versus September. **Electronics** accounts for 94% of the drop: two top products almost stopped selling (a stock-out). New-customer acquisition fell 56%, and the cancellation rate doubled.

Because the data is a static snapshot, relative dates are anchored to the **last order date** (2026-10-31), not today's date. "Last month" means October 2026.

---

## AI and agent approach

```
Browser ──▶ FastAPI routes ──▶ AnalystAgent
                                 │
                                 ├─ LLM: Gemini or Claude    ◀── system prompt: dataset facts + rules
                                 │      │ tool calls
                                 │      ▼
                                 ├─ ToolRegistry ── Pydantic validation ──▶ analytics (SQL) / charts / guarded SQL
                                 │      │ JSON results go back to the LLM
                                 │      ▼
                                 ├─ Grounding check: every number must match a tool result
                                 │
                                 └─ Rule-based engine: offline mode, and the fallback when the LLM fails or can't be verified
```

**Where the LLM is used:**
1. **Understanding the question and choosing tools** (agent loop, `app/agent/agent.py`).
2. **Writing the final answer** from tool results.
3. **Rewriting dashboard insights** (`app/agent/insight_writer.py`).

All numbers are computed by SQL. The LLM never calculates them.

**Controlled tools** (`app/agent/tools.py`): `get_dataset_info`, `get_sales_metrics`, `get_monthly_revenue`, `get_revenue_breakdown`, `get_top_products`, `get_customer_metrics`, `get_top_customers`, `get_inactive_customers`, `get_customer_trend`, `analyze_revenue_change`, `generate_chart`, `run_sql`, and `flag_unsupported`. Each tool has:
- a Pydantic input model, so bad arguments come back to the model as `is_error` tool results it can correct instead of crashing;
- a handler that calls the analytics layer;
- timing and logging.

The same registry powers the offline engine, so both paths share one implementation.

**Agent loop.** This is a manual tool-use loop rather than the SDK's beta tool runner, because it needs to:
- cap iterations;
- record every call for the UI's "Analysis steps" trace;
- capture charts;
- run verification before anything is shown.

Conversation history is append-only, and assistant turns, including thinking blocks, are sent back unchanged.

**Grounding and reliability** (`app/agent/grounding.py`):
- **Numeric verification.** Every number in the answer (with `$`, `%`, `K/M` and rounding precision understood) must match a value in the tool results. Dates, years, small counts such as "top 5", numbers the user typed, and digits inside names ("27in 4K Monitor") are ignored.
- **Self-correction.** If a figure can't be verified, the model is told which figures failed and asked once to fix them. If verification still fails, the user gets the verified rule-based answer with a warning, or the LLM answer clearly flagged when no rule-based answer exists.
- **No answers from memory.** If the model answers a data question without calling any tool, it is sent back to use the tools.
- **Unsupported questions** (profit and margins, marketing spend, inventory, forecasts, off-topic) go through `flag_unsupported` and come back as a clear "can't answer from this dataset" with suggestions.
- **LLM/API failures.** Authentication errors, rate limits, timeouts, connection errors, 5xx responses and refusals are each mapped to a short reason, and the question is answered by the rule-based engine with a visible warning. The SDK retries transient errors (`LLM_MAX_RETRIES`). The **server-side refusal fallback** (beta) is enabled and switches itself off if your organisation doesn't have access.
- **Guarded SQL** (`app/sql_guard.py`) for ad-hoc questions:
  - a single SELECT/WITH statement with no comments;
  - a read-only connection;
  - an SQLite authorizer that permits only whitelisted tables and views;
  - a 3-second time budget and a 200-row cap.

**Providers and model choice.** The agent talks to a small provider interface (`create(system, messages, tools)`), with two implementations:

- **Google Gemini** (`app/agent/gemini.py`, default when `GEMINI_API_KEY` is set) uses `gemini-3.5-flash` for fast, reliable tool calling.
  - The adapter converts the conversation to Gemini's format on every call.
  - It replays Gemini's own turns unchanged, because Gemini 3 "thought signatures" must be returned with tool calls.
  - It keeps each conversation on one model.
  - **Model failover:** each Gemini model has its own quota. When the current model is out of quota (429), overloaded (503) or unavailable (404), it is put on a cool-down and the question is retried on the next model in `GEMINI_FALLBACK_MODELS`. The answer shows which model replied.
- **Anthropic Claude** (`app/agent/llm.py`, used when only `ANTHROPIC_API_KEY` is set) uses Claude Opus 5.5 at `medium` effort. The system prompt and tool list are byte-stable and cached (`cache_control`).

Set `LLM_PROVIDER` to force a provider.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, database counts, LLM mode/model |
| GET | `/api/dataset` | Coverage, schema, definitions, data-quality report |
| GET | `/api/kpis` | Revenue, orders, customers and AOV, plus change vs the previous period |
| GET | `/api/metrics/sales`, `/metrics/monthly`, `/metrics/breakdown`, `/metrics/customers` | Metrics |
| GET | `/api/products`, `/customers/top`, `/customers/inactive`, `/customers/trend` | Rankings and lists |
| GET | `/api/charts/{chart}` | Chart spec from live data (9 chart types) |
| GET | `/api/analysis/revenue-change` | Multi-step revenue diagnosis |
| GET | `/api/insights` | Data-backed insights (LLM-rewritten when available) |
| POST | `/api/ask` | Ask the analyst: `{"question": "...", "history": [...]}` |
| POST | `/api/query` | Guarded read-only SQL |

Date filters use `start_date` / `end_date` (`YYYY-MM-DD`). Errors share one envelope, `{"error": {"code", "message"}}`, with these status codes:
- 422: invalid input
- 404: no data for the period
- 400: rejected query
- 503: database unavailable

---

## Testing

`pytest` runs 176 tests offline in about 15 seconds:

| File | What it proves |
|---|---|
| `test_ingestion.py` | Duplicates, normalisation, type conversion, imputation, dropped-row reasons, referential integrity, failure modes (missing file or column, failed rebuild keeps the old DB) |
| `test_analytics.py` | Hand-computed answers on a small dirty dataset, plus SQL results on the full sample cross-checked against **independent pandas** calculations, and detection of the planted October story |
| `test_ai_questions.py` | **Independently verifiable AI questions**: each assignment question is asked in natural language, and the answer is checked against pandas-computed values (top products in order, every monthly figure, inactive count, top category, decline % and driver, chart data, unsupported questions) |
| `test_agent.py` | The agent loop with a scripted fake LLM: tool execution, invalid arguments, unknown tools, self-correction of invented figures, fallback when they persist, API failure fallback, step limit, no-tool answers, charts, history, insight verification |
| `test_gemini_client.py` | Gemini adapter with real `google-genai` types: message conversion, thought-signature replay, response parsing, safety blocks, error mapping, **model failover** for questions and insights, and correct model attribution |
| `test_llm_client.py` | The real Anthropic SDK over a mocked HTTP transport: request shape, mapping of 401/403/404/429/400/500/529/timeout/connection/refusal, beta-fallback auto-disable |
| `test_grounding.py` | Number parsing, rounding tolerance, invented-figure detection, masking of names and dates |
| `test_sql_guard.py` | Allowed queries; writes, multiple statements, PRAGMA, ATTACH, internal tables and comments rejected; data unchanged after attacks; row cap; runaway-query timeout |
| `test_api.py` / `test_router.py` | Endpoints, validation errors (422/404/400), intent and date-period parsing |

`pytest -m live` sends the key questions to the real LLM configured in `.env` and checks the answers against the analytics layer.

---

## Demo walkthrough

1. `uvicorn app.main:app`: the console logs the database build and the agent mode.
2. **Dashboard:** KPI tiles, the four charts, date presets (try *Last month*: every KPI shows red against September), insights, and the data-quality report.
3. **AI question:** *"What were the top 5 products by revenue?"*, then open **Analysis steps** to see the tool call, its arguments and its result. The badge shows how many figures were verified.
4. **Multi-step analysis and insight:** *"Why did revenue decline last month?"* gives the drivers in order of impact plus a revenue chart.
5. **Chart generation:** *"Show me monthly revenue as a chart"*, or *"Pie chart of revenue by region"*.
6. **Unsupported and invalid input:** *"What is our profit margin?"*, an empty question, or gibberish.

Tip: `http://127.0.0.1:8000/?ask=Why did revenue decline last month?` asks a question on page load.

---

## Project structure

```
app/
  main.py            FastAPI app factory, error handlers, static files
  config.py          Settings (env / .env)
  db.py              Schema DDL, views, connection helpers
  ingestion.py       Load -> clean -> validate -> SQLite (+ data-quality report)
  analytics/         sales, products, customers, diagnostics (multi-step), insights, periods
  charts.py          Chart specs from live queries
  sql_guard.py       Read-only SQL execution
  agent/             tools, gemini + llm (provider clients), agent loop, grounding, router (rule-based), prompts
  api/               routes and request schemas
  static/            dashboard (HTML/CSS/JS)
data/raw/            sample CSVs (committed); data/ecommerce.db is generated
scripts/generate_data.py
tests/
```

---

## Known limitations

- **Synthetic, static data.** Relative dates are anchored to the last order date. Revenue counts only completed orders and has no cost data, so profit questions are declined by design.
- **Grounding checks numbers, not meaning.** A correct figure attached to the wrong entity, or a wrong qualitative claim, would pass the numeric check. The system prompt and tool design reduce this risk, but cannot rule it out.
- **The offline engine is narrow.** It recognises common phrasings of supported questions and ignores conversation history. Anything else is reported as unsupported rather than guessed.
- **Latency.** AI answers that need several tool calls typically take several seconds to tens of seconds. Responses are not streamed.
- **Free-tier quotas.** The Gemini free tier allows only about 20 requests per model per day, and one question uses 2–4 requests. Model failover stretches this, but enable billing on the Google AI Studio project for regular use. When every model is exhausted, the rule-based engine answers, with a warning.
- **Single-user deployment.** There is no authentication, SQLite serves a read-heavy single instance, and insights are cached in memory per process.
- **Internet needed in the browser.** Plotly, marked and DOMPurify load from public CDNs.
- **LLM output varies between runs**, which is why the automated suite drives the agent loop with a scripted fake model and keeps real-API tests opt-in.
