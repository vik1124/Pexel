# Module Reference

## config.py

All global constants. Import from here; never hardcode values elsewhere.

```python
OPENAI_MODELS: list[str]          # ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]
RESOLUTIONS: dict[str, (int,int)] # {"low": (400,300), "medium": (800,600), "high": (1200,900)}
CHART_TYPES: list[str]            # all 22 chart type strings (see chart_types.md)

NUMERIC_TOLERANCE = 0.10          # ±10% relative tolerance for value matching
LABEL_SIMILARITY_THRESHOLD = 0.70 # minimum fuzzy string ratio for label match

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "graph_data_extraction"
LLM_CALLS_EXPERIMENT = "llm_calls"

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"   # PNGs land here: charts/{low,medium,high}/{type}.png

MAX_CONCURRENT = 3   # max simultaneous OpenAI API calls
```

---

## chart.py

```python
@dataclass
class Chart(ABC):
    chart_type:   str
    ground_truth: Dict = field(default_factory=dict)  # populated by generate_plot()
    fig:          go.Figure = None                     # populated by generate_plot()

    @abstractmethod
    def generate_plot(self) -> "Chart": ...

    def save_images(self) -> Dict[str, Path]:
        # Writes {CHARTS_DIR}/{res}/{chart_type}.png for every resolution in config
        # Returns {"low": Path, "medium": Path, "high": Path}
```

---

## chart_generator.py

One concrete `Chart` subclass per chart type. Each class:
- Calls `super().__init__(chart_type="<type>")` in `__init__`
- Implements `generate_plot()` which sets `self.fig` and `self.ground_truth` and returns `self`
- No extra fields — all data is hardcoded in `generate_plot()`

```
BarVerticalChart    BarHorizontalChart  LineChart       ScatterChart
PieChart            HistogramChart      BoxChart        HeatmapChart
AreaChart           DualAxisChart       LogScaleChart   Subplots2x2Chart
BubbleChart         WaterfallChart      CandlestickChart ViolinChart
RadarChart          StackedBarChart     StackedBar100Chart Surface3DChart
FunnelChart         TreemapChart
```

Module-level layout helpers shared by all classes:
```python
_L    = dict(font=..., plot_bgcolor="white", paper_bgcolor="white", margin=...)
_GRID = dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=1)
```

### Public API

```python
chart_factory(chart_type: str) -> Chart
# Looks up the right subclass in _CHART_CLASSES, instantiates it (no args),
# calls generate_plot(), and returns the populated Chart object.
# This is the primary way to get a Chart object.

generate_all_charts() -> dict[str, Chart]
# Returns {chart_type: Chart} for all 22 types. Figures and ground_truth populated.
# Does NOT write images to disk.

generate_and_save_all() -> dict[str, Chart]
# Like generate_all_charts() but also calls chart.save_images() for each chart.
# Prints "OK {chart_type} [resolutions]" as it goes.
```

### Adding a new chart type

1. Write `class MyChart(Chart)` with `__init__` setting `chart_type` and `generate_plot()` setting `self.fig` and `self.ground_truth`
2. Add `"my_chart": MyChart` to `_CHART_CLASSES`
3. Add an extraction schema to `schemas.py`
4. Add `"my_chart": MySchema` to `_SCHEMAS` in `llm_extractor.py`
5. Add a prompt to `_PROMPTS` in `llm_extractor.py`
6. Add slot converter branches in `_gt_slots()` and `_pred_slots()` in `evaluator.py`
7. Add `"my_chart"` to `CHART_TYPES` in `config.py`

---

## schemas.py

Pydantic models that the LLM fills in. One `*Extraction` class per chart type family.

| Schema class | Used by chart types |
|---|---|
| `BarChartExtraction` | `bar_vertical`, `bar_horizontal` |
| `LineChartExtraction` | `line`, `log_scale` |
| `ScatterChartExtraction` | `scatter` |
| `PieChartExtraction` | `pie` |
| `HistogramExtraction` | `histogram` |
| `BoxPlotExtraction` | `box` |
| `HeatmapExtraction` | `heatmap` |
| `AreaChartExtraction` | `area` |
| `DualAxisExtraction` | `dual_axis` |
| `SubplotsExtraction` | `subplots_2x2` |
| `BubbleChartExtraction` | `bubble` |
| `WaterfallExtraction` | `waterfall` |
| `CandlestickExtraction` | `candlestick` |
| `ViolinExtraction` | `violin` |
| `RadarExtraction` | `radar` |
| `StackedBarExtraction` | `stacked_bar`, `stacked_bar_100` |
| `SurfaceExtraction` | `surface_3d` |
| `FunnelExtraction` | `funnel` |
| `TreemapExtraction` | `treemap` |

All `*Extraction` classes have an optional `title` field. The list/collection field is always named after what it contains (`data_points`, `slices`, `bins`, `groups`, `cells`, `panels`, `points`, `bars`, `candles`, `stages`, `nodes`, `series`).

---

## llm_extractor.py

```python
async def extract_chart_data(
    image_path: Path,
    chart_type: str,
    model_name: str,
    api_key:    str,
) -> tuple[Any | None, dict]:
    # Returns (pydantic_model_instance | None, metadata_dict)
    # metadata keys: success, latency_seconds, prompt_tokens, completion_tokens,
    #                total_tokens, model, chart_type, image_path, [error]
```

Internally: creates a `pydantic_ai.Agent` per call with the schema from `_SCHEMAS[chart_type]`
and the prompt from `_PROMPTS[chart_type]`. Passes the image as `BinaryContent`.

**`_SCHEMAS`** — maps chart_type → Pydantic class (the LLM's structured output type)

**`_PROMPTS`** — maps chart_type → extraction instruction string (tells the LLM what to read)

**`_SYSTEM_PROMPT`** — shared system prompt emphasising accuracy and not hallucinating

---

## evaluator.py

```python
def evaluate(chart_type: str, ground_truth: dict, extracted: Any) -> dict:
    # Returns: {precision, recall, f1, accuracy, tp, fp, fn, total_gt}
    # extracted is a Pydantic model instance (or None → all zeros)
```

### How matching works

1. `_gt_slots(chart_type, ground_truth)` → `list[(label, value)]`
2. `_pred_slots(chart_type, extracted)` → `list[(label, value)]`
3. `_match(gt_slots, pred_slots)` → `(tp, fp, fn)` via greedy label-aware matching:
   - For each (pred, gt) pair: compute label similarity (fuzzy string ratio)
   - Keep pairs where `sim >= 0.70` AND `|gt - pred| / |gt| <= 0.10`
   - Sort by similarity, greedily assign (each slot matched at most once)
   - `tp` = matched pairs, `fp` = unmatched predictions, `fn` = unmatched gt slots

`_within(gt, pred)` — relative tolerance check: `|gt - pred| / |gt| <= NUMERIC_TOLERANCE`

`_sim(a, b)` — `difflib.SequenceMatcher` ratio on lowercased, stripped strings

---

## mlflow_tracker.py

```python
def setup_mlflow() -> None
# Sets tracking URI, creates experiments if they don't exist.

def log_experiment_run(*, model, chart_type, resolution, width, height,
                       metrics, llm_meta, ground_truth, extracted, image_path) -> str
# Logs to EXPERIMENT_NAME ("graph_data_extraction"). Returns run_id.
# Params: model, chart_type, resolution, width, height, tolerance, total_gt_slots
# Metrics: precision, recall, f1, accuracy, tp, fp, fn, latency_seconds, token counts
# Artifacts: chart PNG, ground_truth JSON, extracted JSON

def log_llm_call(*, model, chart_type, resolution, llm_meta) -> None
# Logs raw call telemetry to LLM_CALLS_EXPERIMENT ("llm_calls").
# Lighter-weight — no artifacts, just latency + token counts.
```

MLFlow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

---

## run_experiments.py

Entry point for the full benchmark. Uses Prefect for concurrent execution.

```python
async def benchmark_flow(
    models:      list[str] | None = None,  # defaults to config.OPENAI_MODELS
    chart_types: list[str] | None = None,  # defaults to config.CHART_TYPES
    resolutions: list[str] | None = None,  # defaults to config.RESOLUTIONS
) -> Path  # path to results_summary.csv
```

**Tasks:**

`generate_charts_task(chart_types, resolutions) -> dict[str, Chart]`
— Generates images if any are missing (idempotent). Returns Chart objects with
  `chart.ground_truth` populated.

`run_one_task(*, model, chart, resolution) -> dict`
— Calls `extract_chart_data`, then `evaluate`, then logs to MLFlow.
  Retried up to 2× on failure. Rate-limited by `asyncio.Semaphore(MAX_CONCURRENT)`.

`save_results_task(rows) -> Path`
— Writes `output/results/results_summary.csv`. Prints pivot tables by
  model×resolution and by chart_type.

**Environment variables:**
```
OPENAI_API_KEY   required
MODELS           comma-separated override, e.g. "gpt-4o-mini"
CHART_TYPES      comma-separated override, e.g. "bar_vertical,pie"
RESOLUTIONS      comma-separated override, e.g. "low,high"
```

**Quick run (subset):**
```python
from run_experiments import benchmark_flow
import asyncio
asyncio.run(benchmark_flow(models=["gpt-4o-mini"], chart_types=["pie", "bar_vertical"]))
```
