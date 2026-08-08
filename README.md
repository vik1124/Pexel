# Graph Data Extraction Benchmark

A research-grade framework for benchmarking how accurately OpenAI vision LLMs extract structured data from chart images. Generates 22 chart types at three resolutions, runs 4 models against each, and evaluates precision/recall/F1 with full MLFlow experiment tracking.

---

## Project Structure

```
graphl/
├── config.py            # Models, chart types, resolutions, tolerance constants
├── schemas.py           # Pydantic models — one per chart type (LLM return types)
├── chart_generator.py   # Generates Plotly figures + ground-truth dicts, exports PNGs
├── llm_extractor.py     # Calls OpenAI vision models via pydantic-ai, returns typed extractions
├── evaluator.py         # Computes TP/FP/FN/precision/recall/F1 from slot matching
├── mlflow_tracker.py    # Logs runs and LLM call telemetry to MLFlow / SQLite
├── run_experiments.py   # Prefect flow — orchestrates the full benchmark
├── requirements.txt     # Python dependencies
├── .env                 # OPENAI_API_KEY (not committed)
├── mlflow.db            # SQLite MLFlow backend (auto-created)
└── output/
    ├── charts/          # Generated PNGs — low/ medium/ high/ subdirs
    └── results/
        └── results_summary.csv
```

---

## Setup

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Note:** `kaleido==0.2.1` is required for Plotly PNG export. Newer versions may break on Windows.

### 2. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-...
```

### 3. Verify the environment

```bash
python -c "from chart_generator import generate_all_charts; print('OK')"
python -c "from llm_extractor import _SCHEMAS; print(len(_SCHEMAS), 'schemas loaded')"
```

---

## Generating Chart Images

To regenerate all 66 PNG files (22 chart types × 3 resolutions) without running LLM experiments:

```bash
python -c "from chart_generator import generate_and_save_all; generate_and_save_all()"
```

Images are written to `output/charts/{low,medium,high}/{chart_type}.png`.

Resolutions:

| Name   | Width × Height |
|--------|---------------|
| low    | 400 × 300     |
| medium | 800 × 600     |
| high   | 1200 × 900    |

---

## Running the Full Benchmark

The runner is a [Prefect 3](https://docs.prefect.io) flow with three tasks:

| Task | Type | What it does |
|---|---|---|
| `generate-charts` | sync | Generates 66 PNGs; skipped if images already exist |
| `extract-and-evaluate` | async, retries=2 | One LLM call + evaluation + MLFlow log |
| `save-results` | sync | Writes `results_summary.csv`, prints pivot tables |

All `extract-and-evaluate` tasks are submitted concurrently via `ConcurrentTaskRunner` and rate-limited to `MAX_CONCURRENT=3` simultaneous API calls.

### Run directly

```bash
python run_experiments.py
```

This runs **264 experiments** (4 models × 22 chart types × 3 resolutions). Expect ~2–4 hours depending on API latency.

### Run via Prefect with UI observability

```bash
# Start the Prefect server (keep running in a separate terminal)
prefect server start

# Run the flow — progress and logs appear at http://localhost:4200
python run_experiments.py
```

The Prefect UI shows live task state, retry events, logs per task, and a full run history.

### Partial runs with environment variable overrides

Any axis of the benchmark can be overridden without editing code:

```bash
# Single model, two resolutions
MODELS="gpt-4o-mini" RESOLUTIONS="low,high" python run_experiments.py

# Subset of chart types
CHART_TYPES="bar_vertical,pie,heatmap" python run_experiments.py

# Single experiment for debugging
MODELS="gpt-4o-mini" CHART_TYPES="bar_vertical" RESOLUTIONS="medium" python run_experiments.py
```

### Run programmatically from Python

```python
import asyncio
from run_experiments import benchmark_flow

# Full run
asyncio.run(benchmark_flow())

# Targeted run
asyncio.run(benchmark_flow(
    models=["gpt-4o-mini"],
    chart_types=["bar_vertical", "pie", "heatmap"],
    resolutions=["medium"],
))
```

Results are saved to `output/results/results_summary.csv` and printed as pivot tables:

```
Mean metrics by model × resolution:
                         precision  recall    f1  accuracy
model        resolution
gpt-4.1      high          0.xxx   0.xxx  0.xxx     0.xxx
             low           ...
```

---

## Prefect UI

```bash
prefect server start
```

Opens at [http://localhost:4200](http://localhost:4200). Shows:
- Live task status and retry events for the running flow
- Per-task logs (each `extract-and-evaluate` task logs its P/R/F1 inline)
- Full run history and task duration breakdown
- Flow parameters used for each run

The Prefect server and MLFlow UI can run simultaneously on different ports.

---

## MLFlow Experiment Tracking

### Start the MLFlow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open [http://localhost:5000](http://localhost:5000).

Two experiments are tracked:

| Experiment | What it records |
|---|---|
| `graph_data_extraction` | Per-run: precision, recall, F1, accuracy, TP/FP/FN, tokens, latency, chart image artifact, ground-truth JSON, extracted JSON |
| `llm_calls` | Per-API-call telemetry: latency, prompt/completion/total tokens, success flag |

### What gets logged per run

**Parameters:** `model`, `chart_type`, `resolution`, `width`, `height`, `tolerance`, `total_gt_slots`

**Metrics:** `precision`, `recall`, `f1`, `accuracy`, `tp`, `fp`, `fn`, `latency_seconds`, `prompt_tokens`, `completion_tokens`, `total_tokens`

**Artifacts:** chart PNG, `gt_{chart_type}.json`, `extracted_{chart_type}_{model}.json`

---

## Chart Types and Ground Truth

All 22 chart types are listed below. Every chart has a corresponding **ground-truth Python dict** produced alongside the Plotly figure in `chart_generator.py`. The dict keys are specific to each chart type and map directly to the slot structure used by the evaluator.

### Single-axis charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `bar_vertical` | Apple FY-2023 revenue by segment ($B) | `{"data": {"label": value, ...}}` | (5 rows) label, value |
| `bar_horizontal` | GDP 2023 — top 8 economies ($T) | `{"data": {"label": value, ...}}` | (8 rows) label, value |
| `pie` | Desktop OS market share Jun 2024 | `{"slices": {"label": pct, ...}}` | (5 rows) label, percentage |
| `histogram` | US population by age group 2023 (M) | `{"bins": [{"bin_start", "bin_end", "count"}]}` | (8 rows) bin_start, bin_end, count |
| `funnel` | E-commerce conversion funnel | `{"stages": [{"stage", "value"}]}` | (6 rows) stage, value |

### Multi-series line/area charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `line` | S&P 500 + NASDAQ 100 monthly close 2023 | `{"series": {"name": {"x_str": y, ...}}}` | (24 rows) series, x, y |
| `area` | UK renewable energy (Wind + Solar, TWh) Jan–Jun 2023 | `{"series": {"name": {"x_str": y, ...}}}` | (12 rows) series, x, y |
| `log_scale` | Social platform MAU growth 2015–2022 (M) | `{"series": {"name": {"year": val, ...}}}` | (16 rows) series, year, mau |

### Distribution charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `box` | COVID-19 ICU length of stay by age group | `{"groups": {"name": {"median", "q1", "q3", "lower", "upper"}}}` | (4 groups × 5 stats = 20 rows) |
| `violin` | Systolic BP reduction (mmHg) — 4-arm RCT | `{"groups": {"name": {"median", "q1", "q3", "lower", "upper"}}}` | (4 groups × 5 stats = 20 rows) |

### Scatter / relational charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `scatter` | BMI vs systolic BP — 12 de-identified patients | `{"points": [{"x", "y", "label"}]}` | (12 rows) label, x (BMI), y (SBP) |
| `bubble` | GDP/capita vs life expectancy vs population — 8 countries 2022 | `{"points": [{"label", "x", "y", "size"}]}` | (8 rows × 3 numeric dims = 24 slots) |

### Matrix / 2-D charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `heatmap` | Mean temperature (°C) — 4 cities × 4 months | `{"rows": [...], "cols": [...], "values": [[...]]}` | (4×4 = 16 rows) city, month, temp |
| `radar` | WEF competitiveness scores USA vs Germany | `{"series": {"name": {"metric": value, ...}}}` | (2 series × 6 metrics = 12 rows) |

### Stacked / composition charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `stacked_bar` | Electricity generation mix (TWh) — 5 countries 2022 | `{"series": {"source": {"country": value, ...}}}` | (4 sources × 5 countries = 20 rows) |
| `stacked_bar_100` | Population age structure (%) — 5 countries 2023 | `{"series": {"age_group": {"country": pct, ...}}}` | (3 groups × 5 countries = 15 rows) |
| `treemap` | S&P 500 sector market cap ($B, mid-2024) | `{"nodes": [{"label", "parent", "value"}]}` | (21 leaf rows) label, parent, value |

### Dual-axis and multi-panel

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `dual_axis` | US job openings (M) vs unemployment rate (%) Jan–Jun 2023 | `{"bar_values": {...}, "line_values": {...}}` | (6 rows) month, bar_value, line_value |
| `subplots_2x2` | Quarterly regional sales ($M) — 4 quarters × 4 regions | `{"panels": [{"title", "data": {"region": value}}]}` | (4 panels × 4 regions = 16 rows) |

### Financial charts

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `candlestick` | AAPL OHLC — 6 sessions Jan 2024 | `{"candles": [{"date", "open", "high", "low", "close"}]}` | (6 rows × 4 OHLC = 24 slots) |
| `waterfall` | TechCorp FY-2023 P&L ($M) | `{"bars": [{"label", "value", "bar_type"}]}` | (9 rows) label, value, type |

### 3-D

| Chart type | Dataset | GT dict structure | DataFrame shape |
|---|---|---|---|
| `surface_3d` | Mean January temperature (°C) — 4×4 lat/lon grid | `{"lat_labels": [...], "lon_labels": [...], "z": [[...]]}` | (16 rows) lat, lon, temp_celsius |

---

## Evaluation Methodology

Each chart type is reduced to a flat list of `(label, value)` slots. Labels are matched with fuzzy string similarity; values must be within a numeric tolerance.

| Constant | Default | Meaning |
|---|---|---|
| `LABEL_SIMILARITY_THRESHOLD` | 0.70 | Minimum fuzzy ratio for two labels to be considered the same |
| `NUMERIC_TOLERANCE` | 0.10 | Maximum relative error for a value to count as correct (10%) |

**Matching rules:**
- A slot is a **True Positive** only if label similarity ≥ threshold **and** value within tolerance.
- A label match with a wrong value is a **near-miss**: it counts as both FP (bad prediction) and FN (missed ground truth) — it does NOT inflate TP.
- Matching is greedy: highest-similarity candidate pairs are claimed first.
- `surface_3d` uses unlabeled proximity matching (no lat/lon strings in the schema).

---

## Adding a New Chart Type

1. **`chart_generator.py`** — add a `_new_type() -> tuple[go.Figure, dict]` function and register it in `_GENERATORS`.
2. **`schemas.py`** — define `NewTypeExtraction(BaseModel)` with a Pydantic model matching what the LLM should return.
3. **`llm_extractor.py`** — add entries to `_SCHEMAS` and `_PROMPTS`.
4. **`evaluator.py`** — add cases to `_gt_slots` and `_pred_slots`.
5. **`config.py`** — append the key to `CHART_TYPES`.

---

## Models Tested

| Model | Notes |
|---|---|
| `gpt-4o` | Highest capability baseline |
| `gpt-4o-mini` | Smaller / cheaper baseline |
| `gpt-4.1` | Latest GPT-4.1 flagship |
| `gpt-4.1-mini` | Latest GPT-4.1 efficient |

Add or remove models in `config.py` → `OPENAI_MODELS`, or override at runtime with the `MODELS` env var.

---

## Requirements

- Python 3.10+
- `pydantic-ai[openai] >= 0.4.0`
- `openai >= 1.0.0`
- `plotly >= 5.18.0`
- `kaleido == 0.2.1`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `mlflow >= 2.15.0`
- `prefect >= 3.0.0`
- `python-dotenv >= 1.0.0`
