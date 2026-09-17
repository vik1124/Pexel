# Architecture

## What this project does

Benchmarks vision LLMs on how accurately they can extract structured data from chart images.

**Dimensions:** N models × 30 chart types × 3 resolutions (configurable)

## Data flow

```
chart_generator.py
  chart_factory(chart_type)
       │
       ▼
  ChartSubclass.generate_plot()  →  self.fig  +  self.ground_truth (Pydantic model)
       │
       ▼
  generate_and_save_chart(chart_type)  →  output/charts/{low,medium,high}/{chart_type}.png
       │
       ▼
llm_extractor.extract_chart_data(image_path, chart_type, extraction_class, model)
       │  pydantic-ai Agent + vision model
       ▼
  Pydantic extraction model (e.g. BarChartExtraction)
       │
       ▼
evaluator.evaluate_detail(chart_type, ground_truth, extracted)
       │  heuristic slot-based fuzzy matching
       ▼
  {precision, recall, f1, accuracy, tp, fp, fn} + per-slot comparison rows
       │
       │  if heuristic precision < 1.0:
       ▼
llm_evaluator.evaluate_with_llm(chart_type, ground_truth, extracted, model)
       │  semantic re-evaluation with extended thinking
       ▼
  refined metrics + annotated comparison rows
       │
       ▼
Prefect create_table_artifact(...)   →  visible in Prefect UI at localhost:4200
CSV written to output/charts/{resolution}/{artifact_key}.csv
```

Orchestration is done by **Prefect** (`graph_comparison.py`). Chart generation, extraction, and evaluation each run as separate cacheable tasks. Extraction tasks are submitted concurrently across all model × chart_type combinations.

## Module map

| File | Responsibility |
|------|---------------|
| `config.py` | Constants — resolutions, paths, tolerances |
| `chart.py` | `Chart` ABC — `generate_plot()`, `save_images()` |
| `chart_generator.py` | 30 `Chart` subclasses, `CHART_MAPPINGS` registry, `chart_factory()`, `generate_and_save_chart` Prefect task |
| `schemas.py` | Pydantic models the LLM fills in (one per chart type family) |
| `llm_extractor.py` | Creates a `pydantic_ai.Agent` per call, returns extraction + timing/token metadata |
| `evaluator.py` | Converts ground truth + extraction to flat slots, greedy-matches them; `evaluate_detail()` returns per-slot rows |
| `llm_evaluator.py` | Semantic fallback evaluator: uses a pydantic-ai agent with extended thinking when heuristic precision < 1.0 |
| `mlflow_tracker.py` | Logs params, metrics, and artifacts to MLFlow (currently commented out in main flow) |
| `graph_comparison.py` | Prefect flow + `extract` / `evaluate` tasks — orchestrates the full benchmark |

## Key design decisions

**`Chart` base class** — `chart.py` defines the ABC. Every subclass sets `self.extraction_class` (the Pydantic schema the LLM will return) and stores `self.ground_truth` as an instance of that same schema class. This means heuristic and LLM evaluators both operate on the same typed object without any dict-to-model conversion.

**One subclass per chart type** — Each of the 30 chart types has its own `*Chart(Chart)` class. `chart_factory(chart_type)` dispatches via `CHART_MAPPINGS`. Adding a new type means one new class + one registry entry.

**Slot-based evaluation** — Every chart type is reduced to `list[(label, value)]` pairs ("slots") by `_pred_slots()`. The same function handles both ground truth and extracted models (they have the same type). Greedy label-similarity + numeric-proximity matching computes TP/FP/FN without per-chart-type comparison logic.

**LLM fallback evaluator** — When heuristic precision < 1.0, `llm_evaluator.evaluate_with_llm()` re-runs the same GT vs. extracted comparison using a pydantic-ai agent with extended thinking. It is more lenient on label formatting and emits a comment explaining each mismatch. This helps distinguish real extraction errors from evaluator false negatives.

**Task caching** — Both `generate_and_save_chart` and `extract` use Prefect `cache_key_fn`. Chart generation is cached on a hash of `chart_generator.py`. Extraction is cached on (model, chart_type, resolution, image bytes hash) so re-runs skip unchanged combinations.

**Schema reuse** — `log_scale` reuses `LineChartExtraction`; `stacked_bar_100` reuses `StackedBarExtraction`; `log_log_scatter` and `dense_scatter` reuse `ScatterChartExtraction`. The evaluator dispatches on `chart_type` string, not schema class.
