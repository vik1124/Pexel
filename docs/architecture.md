# Architecture

## What this project does

Benchmarks OpenAI vision models on how accurately they can extract structured data from chart images.

**Dimensions:** 4 models × 22 chart types × 3 resolutions = **264 experiments per run**

## Data flow

```
chart_generator.py
  chart_factory(chart_type)
       │
       ▼
  PlotlyChart.generate_plot()   →  self.fig  +  self.ground_truth
       │
       ▼
  chart.save_images()           →  output/charts/{low,medium,high}/{chart_type}.png
       │
       ▼
llm_extractor.extract_chart_data(image_path, chart_type, model)
       │  pydantic-ai Agent + OpenAI vision
       ▼
  Pydantic extraction model (e.g. BarChartExtraction)
       │
       ▼
evaluator.evaluate(chart_type, ground_truth, extracted)
       │  slot-based fuzzy matching
       ▼
  {precision, recall, f1, accuracy, tp, fp, fn}
       │
       ▼
mlflow_tracker.log_experiment_run(...)   →  sqlite:///mlflow.db
```

Orchestration is done by **Prefect** (`run_experiments.py`). All 264 tasks run concurrently, capped at `MAX_CONCURRENT` simultaneous API calls.

## Module map

| File | Responsibility |
|------|---------------|
| `config.py` | All constants — models, chart types, resolutions, paths, tolerances |
| `chart.py` | `Chart` ABC — `generate_plot()`, `save_images()` |
| `chart_generator.py` | `PlotlyChart` subclass, 22 generator functions, `chart_factory`, public API |
| `schemas.py` | Pydantic models the LLM fills in (one per chart type family) |
| `llm_extractor.py` | Calls pydantic-ai Agent, returns extraction + timing/token metadata |
| `evaluator.py` | Converts ground truth + extraction to flat slots, greedy-matches them |
| `mlflow_tracker.py` | Logs params, metrics, and artifacts to MLFlow |
| `run_experiments.py` | Prefect flow + tasks — orchestrates the full benchmark |
| `main.py` | Placeholder entrypoint |

## Key design decisions

**`Chart` base class** — `chart.py` defines the ABC. All chart generation goes through `PlotlyChart.generate_plot()` which populates `self.fig` and `self.ground_truth`. Saving always goes through `Chart.save_images()`, which writes all resolutions.

**One subclass per chart type** — Each of the 22 chart types has its own `*Chart(Chart)` class whose `generate_plot()` contains the Plotly figure logic and sets `self.fig` + `self.ground_truth`. `chart_factory(chart_type)` dispatches to the right class via `_CHART_CLASSES`. Adding a new type means adding one class and one registry entry.

**Slot-based evaluation** — Every chart type is reduced to `list[(label, value)]` pairs ("slots"). Ground truth and LLM extraction are both converted to slots, then greedy-matched by label similarity + numeric proximity. This avoids writing per-chart-type comparison logic.

**Schema reuse** — `log_scale` reuses `LineChartExtraction`; `stacked_bar_100` reuses `StackedBarExtraction`. The evaluator handles both correctly because it dispatches on the chart_type string, not the schema class.
