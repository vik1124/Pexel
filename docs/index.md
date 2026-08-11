# graphl — Wiki

Benchmark for evaluating OpenAI vision models on structured chart-data extraction.

## Pages

- [architecture.md](architecture.md) — Data flow, module map, key design decisions
- [modules.md](modules.md) — Public API of every module (classes, functions, signatures)
- [chart_types.md](chart_types.md) — All 22 chart types: ground truth shape, extraction schema, evaluator slots

## Quick orientation

```
chart_factory("bar_vertical")        # → PlotlyChart with .fig and .ground_truth set
chart.save_images()                  # → {"low": Path, "medium": Path, "high": Path}
extract_chart_data(path, type, model) # → (PydanticModel, metadata_dict)
evaluate(type, ground_truth, model)  # → {precision, recall, f1, accuracy, ...}
```

Full benchmark: `python run_experiments.py`

MLFlow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
