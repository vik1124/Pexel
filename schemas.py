"""Pydantic models used as structured return types for LLM chart extraction."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Bar (vertical and horizontal share the same schema) ──────────────────────

class BarDataPoint(BaseModel):
    label: str  = Field(description="Exact category label as shown on the axis")
    value: float = Field(description="Numeric value read from the value axis")

class BarChartExtraction(BaseModel):
    title:        Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    data_points:  list[BarDataPoint] = Field(default_factory=list)


# ── Line (also reused by log_scale) ──────────────────────────────────────────

class LineDataPoint(BaseModel):
    series: str   = Field(description="Name of the line/series")
    x:      float = Field(description="X-axis value")
    y:      float = Field(description="Y-axis value")

class LineChartExtraction(BaseModel):
    title:       Optional[str] = None
    data_points: list[LineDataPoint] = Field(default_factory=list)


# ── Scatter ───────────────────────────────────────────────────────────────────

class ScatterPoint(BaseModel):
    x:     float
    y:     float
    label: Optional[str] = None

class ScatterChartExtraction(BaseModel):
    title:  Optional[str] = None
    points: list[ScatterPoint] = Field(default_factory=list)


# ── Pie ───────────────────────────────────────────────────────────────────────

class PieSlice(BaseModel):
    label:      str
    percentage: float = Field(description="Percentage value 0-100")

class PieChartExtraction(BaseModel):
    title:  Optional[str] = None
    slices: list[PieSlice] = Field(default_factory=list)


# ── Histogram ─────────────────────────────────────────────────────────────────

class HistogramBin(BaseModel):
    bin_start: float
    bin_end:   float
    count:     float

class HistogramExtraction(BaseModel):
    title: Optional[str] = None
    bins:  list[HistogramBin] = Field(default_factory=list)


# ── Box plot ──────────────────────────────────────────────────────────────────

class BoxGroup(BaseModel):
    group:         str
    median:        float
    q1:            float
    q3:            float
    lower_whisker: Optional[float] = None
    upper_whisker: Optional[float] = None

class BoxPlotExtraction(BaseModel):
    title:  Optional[str] = None
    groups: list[BoxGroup] = Field(default_factory=list)


# ── Heatmap ───────────────────────────────────────────────────────────────────

class HeatmapCell(BaseModel):
    row:   str
    col:   str
    value: float

class HeatmapExtraction(BaseModel):
    title: Optional[str] = None
    cells: list[HeatmapCell] = Field(default_factory=list)


# ── Area ──────────────────────────────────────────────────────────────────────

class AreaDataPoint(BaseModel):
    series: str
    x:      float
    y:      float

class AreaChartExtraction(BaseModel):
    title:       Optional[str] = None
    data_points: list[AreaDataPoint] = Field(default_factory=list)


# ── Dual-axis (bar + line, two y-axes) ───────────────────────────────────────

class DualAxisPoint(BaseModel):
    x_label:    str
    bar_value:  Optional[float] = Field(None, description="Value on the left y-axis (bars)")
    line_value: Optional[float] = Field(None, description="Value on the right y-axis (line)")

class DualAxisExtraction(BaseModel):
    title:            Optional[str] = None
    bar_series_name:  Optional[str] = None
    line_series_name: Optional[str] = None
    left_y_label:     Optional[str] = None
    right_y_label:    Optional[str] = None
    data_points:      list[DualAxisPoint] = Field(default_factory=list)


# ── Subplots 2×2 ──────────────────────────────────────────────────────────────

class SubplotPanel(BaseModel):
    panel_title: str
    data_points: list[BarDataPoint] = Field(default_factory=list)

class SubplotsExtraction(BaseModel):
    title:  Optional[str] = None
    panels: list[SubplotPanel] = Field(default_factory=list)


# ── Bubble chart ──────────────────────────────────────────────────────────────

class BubblePoint(BaseModel):
    label: Optional[str] = None
    x:     float = Field(description="X-axis value")
    y:     float = Field(description="Y-axis value")
    size:  float = Field(description="Bubble size value (third dimension)")

class BubbleChartExtraction(BaseModel):
    title:          Optional[str] = None
    x_axis_label:   Optional[str] = None
    y_axis_label:   Optional[str] = None
    size_label:     Optional[str] = None
    points:         list[BubblePoint] = Field(default_factory=list)


# ── Waterfall ─────────────────────────────────────────────────────────────────

class WaterfallBar(BaseModel):
    label:    str
    value:    float   = Field(description="Positive for increases, negative for decreases")
    bar_type: Optional[str] = Field(None, description="'increase', 'decrease', or 'total'")

class WaterfallExtraction(BaseModel):
    title: Optional[str] = None
    bars:  list[WaterfallBar] = Field(default_factory=list)


# ── Candlestick (OHLC) ───────────────────────────────────────────────────────

class CandlestickCandle(BaseModel):
    date:  str
    open:  float
    high:  float
    low:   float
    close: float

class CandlestickExtraction(BaseModel):
    title:   Optional[str] = None
    candles: list[CandlestickCandle] = Field(default_factory=list)


# ── Violin ────────────────────────────────────────────────────────────────────

class ViolinGroup(BaseModel):
    group:         str
    median:        float
    q1:            float
    q3:            float
    lower_whisker: Optional[float] = None
    upper_whisker: Optional[float] = None

class ViolinExtraction(BaseModel):
    title:  Optional[str] = None
    groups: list[ViolinGroup] = Field(default_factory=list)


# ── Radar / Spider ────────────────────────────────────────────────────────────

class RadarPoint(BaseModel):
    metric: str
    value:  float

class RadarSeries(BaseModel):
    series_name: str
    data_points: list[RadarPoint] = Field(default_factory=list)

class RadarExtraction(BaseModel):
    title:  Optional[str] = None
    series: list[RadarSeries] = Field(default_factory=list)


# ── Stacked bar (absolute and 100% share the same schema) ────────────────────

class StackedBarPoint(BaseModel):
    category: str
    series:   str
    value:    float

class StackedBarExtraction(BaseModel):
    title:       Optional[str] = None
    data_points: list[StackedBarPoint] = Field(default_factory=list)


# ── 3-D surface ───────────────────────────────────────────────────────────────

class SurfacePoint(BaseModel):
    x: float
    y: float
    z: float

class SurfaceExtraction(BaseModel):
    title:         Optional[str] = None
    z_min:         Optional[float] = None
    z_max:         Optional[float] = None
    sample_points: list[SurfacePoint] = Field(default_factory=list)


# ── Funnel ────────────────────────────────────────────────────────────────────

class FunnelStage(BaseModel):
    stage: str
    value: float

class FunnelExtraction(BaseModel):
    title:  Optional[str] = None
    stages: list[FunnelStage] = Field(default_factory=list)


# ── Treemap ───────────────────────────────────────────────────────────────────

class TreemapNode(BaseModel):
    label:  str
    parent: Optional[str] = None
    value:  float

class TreemapExtraction(BaseModel):
    title: Optional[str] = None
    nodes: list[TreemapNode] = Field(default_factory=list)
