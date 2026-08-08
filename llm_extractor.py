"""Use Pydantic AI to extract chart data from PNG images via OpenAI vision models."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from schemas import (
    AreaChartExtraction,
    BarChartExtraction,
    BoxPlotExtraction,
    BubbleChartExtraction,
    CandlestickExtraction,
    DualAxisExtraction,
    FunnelExtraction,
    HeatmapExtraction,
    HistogramExtraction,
    LineChartExtraction,
    PieChartExtraction,
    RadarExtraction,
    ScatterChartExtraction,
    StackedBarExtraction,
    SubplotsExtraction,
    SurfaceExtraction,
    TreemapExtraction,
    ViolinExtraction,
    WaterfallExtraction,
)

# ── Schema registry ───────────────────────────────────────────────────────────

_SCHEMAS: dict[str, type] = {
    # Original 9
    "bar_vertical":   BarChartExtraction,
    "bar_horizontal": BarChartExtraction,
    "line":           LineChartExtraction,
    "scatter":        ScatterChartExtraction,
    "pie":            PieChartExtraction,
    "histogram":      HistogramExtraction,
    "box":            BoxPlotExtraction,
    "heatmap":        HeatmapExtraction,
    "area":           AreaChartExtraction,
    # Extended 13
    "dual_axis":      DualAxisExtraction,
    "log_scale":      LineChartExtraction,   # reuses line schema
    "subplots_2x2":   SubplotsExtraction,
    "bubble":         BubbleChartExtraction,
    "waterfall":      WaterfallExtraction,
    "candlestick":    CandlestickExtraction,
    "violin":         ViolinExtraction,
    "radar":          RadarExtraction,
    "stacked_bar":    StackedBarExtraction,
    "stacked_bar_100": StackedBarExtraction,  # reuses stacked schema
    "surface_3d":     SurfaceExtraction,
    "funnel":         FunnelExtraction,
    "treemap":        TreemapExtraction,
}

# ── Per-chart extraction prompts ─────────────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "bar_vertical": (
        "Extract every bar from this vertical bar chart. "
        "For each bar provide the exact category label (from x-axis) and its numeric value "
        "(read from the y-axis gridlines). Do not skip any bars."
    ),
    "bar_horizontal": (
        "Extract every bar from this horizontal bar chart. "
        "For each bar provide the exact category label (from y-axis) and its numeric value "
        "(read from the x-axis gridlines). Do not skip any bars."
    ),
    "line": (
        "Extract every data point from every line in this line chart. "
        "For each point provide the series/line name, the x-axis value, and the y-axis value. "
        "Use the gridlines to read values as precisely as possible."
    ),
    "scatter": (
        "Extract every point from this scatter plot. "
        "For each point provide the x-coordinate, y-coordinate, and the label text if visible. "
        "Read axis values from the gridlines."
    ),
    "pie": (
        "Extract every slice from this pie chart. "
        "For each slice provide the label text and the percentage value shown."
    ),
    "histogram": (
        "Extract every bar from this histogram. "
        "For each bar provide the bin start value, bin end value, and the count/frequency "
        "shown on the y-axis."
    ),
    "box": (
        "Extract every box from this box plot. "
        "For each box provide: group name, median value (center line), Q1 (bottom of box), "
        "Q3 (top of box), lower whisker end, and upper whisker end."
    ),
    "heatmap": (
        "Extract every cell from this heatmap. "
        "For each cell provide the row label, column label, and numeric value displayed in the cell."
    ),
    "area": (
        "Extract every data point from every area/series in this area chart. "
        "For each point provide the series name, x-axis value, and y-axis value. "
        "Use the gridlines to read values as precisely as possible."
    ),
    "dual_axis": (
        "This is a dual-axis chart combining bars (left y-axis) and a line (right y-axis). "
        "For each x-axis category extract both the bar value (left axis) and the line value "
        "(right axis). Read each axis scale independently — they may have different ranges."
    ),
    "log_scale": (
        "Extract every data point from every line in this line chart. "
        "The y-axis uses a logarithmic scale — read the actual (non-log) numeric values "
        "as shown on the axis labels. For each point provide the series name, "
        "x-axis value, and y-axis value."
    ),
    "subplots_2x2": (
        "This chart contains four subplots arranged in a 2×2 grid. "
        "For each subplot, extract the panel title exactly as shown and all bar values. "
        "For each bar provide the category label and its numeric value."
    ),
    "bubble": (
        "Extract every bubble from this bubble chart. "
        "For each bubble provide: the text label if visible, the x-axis value, "
        "the y-axis value, and the bubble size value (the third dimension). "
        "Read x and y from the axis gridlines; estimate size from the legend or scale bar."
    ),
    "waterfall": (
        "Extract every bar from this waterfall (bridge) chart. "
        "For each bar provide: the label, the value (positive for increases, "
        "negative for decreases), and the type — 'increase', 'decrease', or 'total'."
    ),
    "candlestick": (
        "Extract every candlestick from this OHLC/candlestick chart. "
        "For each candle provide: the date/period label, open price, high price, "
        "low price, and close price. Read prices from the y-axis gridlines."
    ),
    "violin": (
        "Extract the summary statistics from each violin in this chart. "
        "For each group provide: group name, median (center line), Q1 (25th percentile), "
        "Q3 (75th percentile), lower whisker end, and upper whisker end."
    ),
    "radar": (
        "Extract every data series from this radar/spider chart. "
        "For each series provide the series name and, for each spoke/axis, "
        "the metric name and its numeric value."
    ),
    "stacked_bar": (
        "Extract every segment from this stacked bar chart. "
        "For each colored segment provide: the x-axis category label, "
        "the series/stack name (from the legend), and the segment's numeric value."
    ),
    "stacked_bar_100": (
        "Extract every segment from this 100% stacked bar chart. "
        "For each colored segment provide: the x-axis category label, "
        "the series/stack name (from the legend), and the percentage value."
    ),
    "surface_3d": (
        "Extract data points from this 3-D surface plot. "
        "Sample points that cover the full grid. "
        "For each point provide x (longitude or column index as a number — e.g., 0°E → 0, "
        "30°E → 30), y (latitude or row index — e.g., 60°N → 60, 30°S → -30), "
        "and z (the height / temperature / color-mapped value)."
    ),
    "funnel": (
        "Extract every stage from this funnel chart. "
        "For each stage provide the exact stage name and its numeric value."
    ),
    "treemap": (
        "Extract every labelled rectangle from this treemap. "
        "For each rectangle provide: its label, the parent group label "
        "(the larger rectangle it belongs to), and its numeric value."
    ),
}

_SYSTEM_PROMPT = (
    "You are a precise data extraction expert that specialises in reading charts and graphs. "
    "Extract ALL visible data points as accurately as possible. "
    "Use axis gridlines and tick labels to estimate numeric values precisely. "
    "Never skip or omit data points. Never hallucinate values that are not visible in the image."
)


# ── Extraction function ───────────────────────────────────────────────────────

async def extract_chart_data(
    image_path: Path,
    chart_type: str,
    model_name: str,
    api_key: str,
) -> tuple[Any | None, dict]:
    """
    Run the LLM extraction for one (image, chart_type, model) combination.

    Returns
    -------
    extracted : Pydantic model instance or None on failure
    metadata  : dict with timing, token counts, success flag, etc.
    """
    schema = _SCHEMAS[chart_type]
    prompt = _PROMPTS[chart_type]

    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)

    provider = OpenAIProvider(api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)

    agent: Agent = Agent(
        model,
        result_type=schema,
        system_prompt=_SYSTEM_PROMPT,
    )

    image_bytes = image_path.read_bytes()

    start = time.perf_counter()
    try:
        result = await agent.run(
            [
                BinaryContent(data=image_bytes, media_type="image/png"),
                prompt,
            ]
        )
        elapsed = time.perf_counter() - start

        usage = result.usage()
        req  = getattr(usage, "request_tokens",  None) or getattr(usage, "input_tokens",  0) or 0
        resp = getattr(usage, "response_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        metadata: dict = {
            "success":           True,
            "latency_seconds":   round(elapsed, 3),
            "prompt_tokens":     req,
            "completion_tokens": resp,
            "total_tokens":      req + resp,
            "model":             model_name,
            "chart_type":        chart_type,
            "image_path":        str(image_path),
        }
        return result.data, metadata

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return None, {
            "success":           False,
            "error":             str(exc),
            "latency_seconds":   round(elapsed, 3),
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
            "model":             model_name,
            "chart_type":        chart_type,
            "image_path":        str(image_path),
        }
