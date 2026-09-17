# Module Reference

## config.py

All global constants. Import from here; never hardcode values elsewhere.

```python
RESOLUTIONS: dict[str, (int,int)] # {"low": (400,300), "medium": (800,600), "high": (1200,900)}

NUMERIC_TOLERANCE = 0.10          # ±10% relative tolerance for value matching
LABEL_SIMILARITY_THRESHOLD = 0.70 # minimum fuzzy string ratio for label match

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "graph_data_extraction"
LLM_CALLS_EXPERIMENT = "llm_calls"

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"   # PNGs land here: charts/{low,medium,high}/{type}.png
```

The model list and chart type list live in `graph_comparison.py` and `chart_generator.CHART_MAPPINGS` respectively, not in `config.py`.

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
- Sets `self.extraction_class` to the Pydantic schema the LLM will return
- Sets `self.ground_truth` to an instance of that same schema (not a raw dict)
- Implements `generate_plot()` which sets `self.fig` and returns `self`

```
BarVerticalChart    BarHorizontalChart  LineChart           ScatterChart
PieChart            HistogramChart      BoxChart            HeatmapChart
AreaChart           DualAxisChart       LogScaleChart       Subplots2x2Chart
BubbleChart         WaterfallChart      CandlestickChart    ViolinChart
RadarChart          StackedBarChart     StackedBar100Chart  Surface3DChart
FunnelChart         TreemapChart
# No-label variants:
BarVerticalNoLabelChart  BarHorizontalNoLabelChart  HistogramNoLabelChart  Subplots2x2NoLabelChart
# New types:
LogLogScatterChart  LineMultiSeriesChart  DenseScatterChart  DenseLineChart
```

### Public API

```python
CHART_MAPPINGS: dict[str, type[Chart]]
# Registry of all 30 chart types → their Chart subclass.

chart_factory(chart_type: str) -> Chart
# Instantiates the subclass, calls generate_plot(), returns the populated Chart.

generate_all_charts() -> dict[str, Chart]
# Returns {chart_type: Chart} for all types in CHART_MAPPINGS. Does NOT write PNGs.

@task(cache_key_fn=...)
def generate_and_save_chart(chart_type: str) -> str
# Prefect-cacheable task: generates and writes PNGs for one chart type.
# Cached on hash of chart_generator.py — skipped if file hasn't changed.
```

### Adding a new chart type

1. Add `NewTypeExtraction(BaseModel)` to `schemas.py`
2. Write `class NewTypeChart(Chart)` setting `self.extraction_class` and `self.ground_truth`; add to `CHART_MAPPINGS`
3. Add `"new_type"` to `_PROMPTS` in `llm_extractor.py`
4. Add a `_pred_slots` branch in `evaluator.py`

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
    image_path:       Path,
    chart_type:       str,
    extraction_class: type,   # the Pydantic schema class (from Chart.extraction_class)
    model_name:       str,    # pydantic-ai format: "openai:gpt-5.6-luna", "anthropic:claude-sonnet-5", etc.
) -> tuple[Any | None, dict]:
    # Returns (pydantic_model_instance | None, metadata_dict)
    # metadata keys: success, latency_seconds, prompt_tokens, completion_tokens,
    #                total_tokens, model, chart_type, image_path, [error]
```

Internally: creates a `pydantic_ai.Agent` per call with `output_type=extraction_class`
and the prompt from `_PROMPTS[chart_type]`. Passes the image as `BinaryContent`.

**`_PROMPTS`** — maps chart_type → extraction instruction string

**`_SYSTEM_PROMPT`** — shared system prompt emphasising accuracy and not hallucinating

---

## evaluator.py

```python
def evaluate(chart_type: str, ground_truth: Any, extracted: Any) -> dict:
    # Returns: {precision, recall, f1, accuracy, tp, fp, fn, total_gt}
    # Both ground_truth and extracted are Pydantic model instances.

def evaluate_detail(chart_type: str, ground_truth: Any, extracted: Any) -> tuple[dict, list[dict]]:
    # Like evaluate() but also returns per-slot comparison rows for human review.
    # Each row: {status, label, gt_value, extracted_value, diff, comment}
    # status ∈ {"match", "missed (GT only)", "extra (pred only)"}
```

### How matching works

1. `_pred_slots(chart_type, model_instance)` → `list[(label, value)]` — same function for both GT and extracted
2. `_match_detail(gt_slots, pred_slots)` → `(tp, fp, fn, rows)` via greedy label-aware matching:
   - For each (pred, gt) pair: compute label similarity (fuzzy string ratio)
   - Keep pairs where `sim >= 0.70` AND `|gt - pred| / |gt| <= 0.10`
   - Sort by similarity, greedily assign (each slot matched at most once)
   - `tp` = matched pairs, `fp` = unmatched predictions, `fn` = unmatched gt slots

`_within(gt, pred)` — relative tolerance check: `|gt - pred| / |gt| <= NUMERIC_TOLERANCE`

`_sim(a, b)` — `difflib.SequenceMatcher` ratio on lowercased, stripped strings

---

## llm_evaluator.py

Semantic fallback evaluator. Used when heuristic precision < 1.0 to re-evaluate with a vision LLM.

```python
async def evaluate_with_llm(
    chart_type:   str,
    ground_truth: BaseModel,
    extracted:    BaseModel,
    model_name:   str = "openai:gpt-5.6-luna",
) -> tuple[dict, list[dict]]:
    # Returns (metrics_dict, comparison_rows) in same format as evaluate_detail().
    # Uses pydantic_ai.Agent with Thinking(effort="high").
```

**`LLMMatchResult`** — structured output schema: `comparisons: list[SlotComparison]`, `tp`, `fp`, `fn`, `comment`

**`SlotComparison`** — per-slot row: `status ∈ {"match", "missed", "extra"}`, `label`, `gt_value`, `extracted_value`, `diff`, `comment`

The LLM is prompted to be lenient on label formatting (`"Jan"` ↔ `"January"`, `"2020"` ↔ `"2020.0"`) while still requiring values to be within 10%.

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

## graph_comparison.py

Main entry point for the full benchmark. Uses Prefect for concurrent execution.

```python
@flow
def evaluate_extraction() -> None:
    # Generates all charts, then runs extract + evaluate for every (chart_type, model) pair.
    # Results written to output/charts/results.csv and as Prefect table artifacts.
```

**Tasks:**

```python
@task(cache_key_fn=_extract_cache_key, retries=3)
async def extract(chart_type: str, model: str, resolution: str = "high") -> dict:
    # Calls extract_chart_data(). Returns {extracted, llm_meta, chart_type, model, resolution}.
    # Cached on hash of (llm_extractor.py, schemas.py, image bytes, model, chart_type, resolution).

@task(cache_key_fn=_evaluate_cache_key, retries=1)
async def evaluate(extract_data: dict) -> dict:
    # Runs heuristic evaluate_detail(); falls back to evaluate_with_llm() if precision < 1.0.
    # Creates a Prefect table artifact with per-slot comparison rows.
    # Returns metrics dict with model, chart_type, resolution, evaluator, P/R/F1, tokens.
```

**Edit the model list** by changing the `models` list inside `evaluate_extraction()` in the file.

**Quick run (Python):**
```python
from graph_comparison import evaluate_extraction
from dotenv import load_dotenv
load_dotenv()
evaluate_extraction()
```
