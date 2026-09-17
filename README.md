# Graph Data Extraction Benchmark

A research-grade framework for benchmarking how accurately vision LLMs extract structured data from chart images. Generates 30 chart types at three resolutions, runs 7 models across providers (OpenAI, Anthropic, OpenRouter), and evaluates precision/recall/F1 with Prefect orchestration and MLFlow experiment tracking.

---

## Project Structure

```
graphl/
├── config.py            # Resolutions, tolerance constants, paths
├── schemas.py           # Pydantic models — one per chart type (LLM return types)
├── chart.py             # Chart ABC — generate_plot(), save_images()
├── chart_generator.py   # 30 Chart subclasses, CHART_MAPPINGS registry, chart_factory()
├── llm_extractor.py     # Calls vision models via pydantic-ai, returns typed extractions
├── evaluator.py         # Heuristic slot-matching: TP/FP/FN/precision/recall/F1
├── llm_evaluator.py     # Semantic LLM fallback evaluator (used when heuristic P < 1.0)
├── mlflow_tracker.py    # Logs runs and LLM call telemetry to MLFlow / SQLite
├── graph_comparison.py  # Prefect flow — orchestrates the full benchmark (main entry point)
├── pyproject.toml       # Python dependencies (managed by uv)
├── .env                 # API keys (not committed)
├── mlflow.db            # SQLite MLFlow backend (auto-created)
└── output/
    └── charts/          # Generated PNGs — low/ medium/ high/ subdirs + results.csv
```

---

## Setup

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Note:** `kaleido==0.2.1` is required for Plotly PNG export. Newer versions may break on Windows.

### 2. Set your API keys

Create a `.env` file in the project root with whichever providers you want to use:

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

### 3. Verify the environment

```bash
python -c "from chart_generator import CHART_MAPPINGS; print(len(CHART_MAPPINGS), 'chart types loaded')"
python -c "from llm_extractor import _PROMPTS; print(len(_PROMPTS), 'prompts loaded')"
```

---

## Generating Chart Images

To regenerate all PNGs (30 chart types × 3 resolutions = 90 files) without running LLM experiments:

```bash
python -c "from chart_generator import generate_all_charts, CHART_MAPPINGS; [c.generate_plot().save_images() for c in [__import__('chart_generator').chart_factory(t) for t in CHART_MAPPINGS]]"
```

Or use the Prefect-cached task (idempotent):

```python
from graph_comparison import generate_and_save_chart
from chart_generator import CHART_MAPPINGS
for chart_type in CHART_MAPPINGS:
    generate_and_save_chart(chart_type)
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

The runner is a [Prefect 3](https://docs.prefect.io) flow in `graph_comparison.py` with three task stages:

| Task | Type | What it does |
|---|---|---|
| `generate_and_save_chart` | sync, cached | Generates PNGs for one chart type; re-skipped if chart_generator.py is unchanged |
| `extract` | async, retries=3 | One LLM call; cached on (model, chart_type, resolution, image hash) |
| `evaluate` | async, retries=1 | Heuristic evaluation; falls back to LLM evaluator if heuristic precision < 1.0 |

All `extract` and `evaluate` tasks run concurrently (Prefect submits them in parallel across all model × chart_type combinations).

### Run directly

```bash
python graph_comparison.py
```

This runs all combinations across the models configured in `evaluate_extraction()`. Results land in `output/charts/results.csv` and as Prefect table artifacts visible at [http://localhost:4200](http://localhost:4200).

### Run via Prefect with UI observability

```bash
# Start the Prefect server (keep running in a separate terminal)
prefect server start

# Run the flow — progress and per-slot comparison tables appear at http://localhost:4200
python graph_comparison.py
```

### Editing the model list

Open `graph_comparison.py` and edit the `models` list in `evaluate_extraction()`:

```python
models = [
    "openai:gpt-5.5",
    "openai:gpt-5.6-luna",
    "anthropic:claude-sonnet-5",
    "openrouter:qwen/qwen3.8-27b",
]
```

Provider prefixes (`openai:`, `anthropic:`, `openrouter:`) are handled by pydantic-ai — set the corresponding API key in `.env`.

---

## Prefect UI

```bash
prefect server start
```

Opens at [http://localhost:4200](http://localhost:4200). Shows:
- Live task status and retry events for the running flow
- Per-task logs (each `extract` and `evaluate` task logs its P/R/F1 inline)
- Per-run slot comparison tables (as Prefect artifacts — one per chart_type × resolution × model)
- Full run history and task duration breakdown

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

30 chart types are registered in `CHART_MAPPINGS`. Every chart has a **ground-truth Pydantic model** stored as `Chart.ground_truth` — the same schema class that the LLM returns, which makes slot comparison straightforward. See [docs/chart_types.md](docs/chart_types.md) for per-type details.

### Chart type groups

### Labelled originals (22)

| Group | Chart types |
|---|---|
| Single-axis | `bar_vertical`, `bar_horizontal`, `pie`, `histogram`, `funnel` |
| Line / area | `line`, `area`, `log_scale` |
| Distribution | `box`, `violin` |
| Relational | `scatter`, `bubble` |
| Matrix | `heatmap`, `radar` |
| Stacked | `stacked_bar`, `stacked_bar_100`, `treemap` |
| Multi-panel | `dual_axis`, `subplots_2x2` |
| Financial | `candlestick`, `waterfall` |
| 3-D | `surface_3d` |

### No-label variants (4)

Identical charts with axis tick labels removed to test extraction without textual anchors:
`bar_vertical_nolabel`, `bar_horizontal_nolabel`, `histogram_nolabel`, `subplots_2x2_nolabel`

### New chart types (4)

| Chart type | Description |
|---|---|
| `log_log_scatter` | Scatter plot with both axes on log scale |
| `line_multi_series` | Line chart with many overlapping series |
| `dense_scatter` | High-density scatter (many points, small markers) |
| `dense_line` | High-density multi-series line chart |

---

## Evaluation Methodology

Each chart type is reduced to a flat list of `(label, value)` slots. Labels are matched with fuzzy string similarity; values must be within a numeric tolerance.

| Constant | Default | Meaning |
|---|---|---|
| `LABEL_SIMILARITY_THRESHOLD` | 0.70 | Minimum fuzzy ratio for two labels to be considered the same |
| `NUMERIC_TOLERANCE` | 0.10 | Maximum relative error for a value to count as correct (10%) |

**Two-stage evaluation:**
1. **Heuristic** (`evaluator.py`) — greedy slot matching. Fast and deterministic.
2. **LLM fallback** (`llm_evaluator.py`) — if heuristic precision < 1.0, a pydantic-ai agent re-evaluates the same slots with extended thinking. It is lenient on label formatting (`"Jan"` matches `"January"`) and provides a comment explaining each discrepancy.

**Heuristic matching rules:**
- A slot is a **True Positive** only if label similarity ≥ threshold **and** value within tolerance.
- A label match with a wrong value is a **near-miss**: counts as both FP and FN — does NOT inflate TP.
- Matching is greedy: highest-similarity pairs are claimed first.
- `surface_3d` uses unlabeled proximity matching (no lat/lon strings in the schema).

---

## Adding a New Chart Type

1. **`schemas.py`** — define `NewTypeExtraction(BaseModel)` matching what the LLM should return.
2. **`chart_generator.py`** — add `class NewTypeChart(Chart)` with `self.extraction_class = NewTypeExtraction` and `generate_plot()` setting `self.fig` and `self.ground_truth` (as a `NewTypeExtraction` instance). Register it in `CHART_MAPPINGS`.
3. **`llm_extractor.py`** — add an entry to `_PROMPTS`.
4. **`evaluator.py`** — add a `_pred_slots` branch for the new chart type.

---

## Models Tested

Models are specified via pydantic-ai's `provider:model-name` string format. Edit the `models` list in `graph_comparison.py:evaluate_extraction()` to change which models run.

| Provider prefix | Example | Key needed |
|---|---|---|
| `openai:` | `openai:gpt-5.6-luna` | `OPENAI_API_KEY` |
| `anthropic:` | `anthropic:claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| `openrouter:` | `openrouter:qwen/qwen3.8-27b` | `OPENROUTER_API_KEY` |

Default model list (as of the most recent run):
- `openai:gpt-5.5`, `openai:gpt-5.6-luna`, `openai:gpt-5.6-terra`, `openai:gpt-5.6-sol`
- `anthropic:claude-sonnet-5`, `anthropic:claude-opus-5`
- `openrouter:qwen/qwen3.8-27b`

---

## Requirements

Managed via `uv` and `pyproject.toml`. Key dependencies:

- Python 3.10+
- `pydantic-ai` (multi-provider: openai, anthropic, openrouter)
- `plotly` + `kaleido==0.2.1` (PNG export — newer kaleido versions break on Windows)
- `pandas`, `numpy`
- `mlflow`
- `prefect >= 3.0.0`
- `python-dotenv`
