# graphl — Wiki

Benchmark for evaluating vision LLMs on structured chart-data extraction. Supports OpenAI, Anthropic, and OpenRouter models.

## Pages

- [architecture.md](architecture.md) — Data flow, module map, key design decisions
- [modules.md](modules.md) — Public API of every module (classes, functions, signatures)
- [chart_types.md](chart_types.md) — All 30 chart types: ground truth shape, extraction schema, evaluator slots

## Quick orientation

```python
from chart_generator import chart_factory

chart = chart_factory("bar_vertical")   # → Chart with .fig and .ground_truth set
chart.generate_plot().save_images()     # → {"low": Path, "medium": Path, "high": Path}

from llm_extractor import extract_chart_data
extracted, meta = await extract_chart_data(
    image_path, "bar_vertical", chart.extraction_class, "openai:gpt-5.6-luna"
)

from evaluator import evaluate_detail
metrics, rows = evaluate_detail("bar_vertical", chart.ground_truth, extracted)
# metrics: {precision, recall, f1, accuracy, tp, fp, fn}
# rows: per-slot comparison table (status, label, gt_value, extracted_value, diff)
```

Full benchmark: `python graph_comparison.py`

Prefect UI: `prefect server start` → [http://localhost:4200](http://localhost:4200)

MLFlow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
