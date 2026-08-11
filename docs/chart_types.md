# Chart Types Reference

22 chart types across two groups. Each entry shows:
- **Ground truth shape** — the dict structure stored in `Chart.ground_truth`
- **Extraction schema** — the Pydantic class the LLM returns
- **Evaluator slots** — how `_gt_slots()` flattens the ground truth to `(label, value)` pairs

---

## Original 9

### bar_vertical
Apple FY-2023 revenue by product segment.

- **Ground truth:** `{"chart_type": "bar_vertical", "data": {"iPhone": 200.6, ...}}`
- **Schema:** `BarChartExtraction` → `data_points: list[BarDataPoint(label, value)]`
- **Slots:** `[(segment_name, value), ...]`

### bar_horizontal
Top-8 economies by GDP 2023.

- **Ground truth:** `{"chart_type": "bar_horizontal", "data": {"USA": 27.4, ...}}`
- **Schema:** `BarChartExtraction` (same as bar_vertical)
- **Slots:** `[(country_name, gdp), ...]`

### line
S&P 500 vs NASDAQ 100 monthly closing values 2023.

- **Ground truth:** `{"chart_type": "line", "series": {"S&P 500": {"1": 4077, ...}, "NASDAQ 100": {...}}}`
- **Schema:** `LineChartExtraction` → `data_points: list[LineDataPoint(series, x, y)]`
- **Slots:** `[("S&P 500|1", 4077), ("NASDAQ 100|1", 11622), ...]`  label = `"{series}|{x}"`

### scatter
BMI vs systolic BP — 12 clinical records.

- **Ground truth:** `{"chart_type": "scatter", "points": [{"x": 22.1, "y": 118, "label": "P01"}, ...]}`
- **Schema:** `ScatterChartExtraction` → `points: list[ScatterPoint(x, y, label)]`
- **Slots:** `[("P01", 118.0), ...]`  label = point label, value = y

### pie
Desktop OS market share Jun 2024.

- **Ground truth:** `{"chart_type": "pie", "slices": {"Windows": 72.9, ...}}`
- **Schema:** `PieChartExtraction` → `slices: list[PieSlice(label, percentage)]`
- **Slots:** `[("Windows", 72.9), ...]`

### histogram
US population by age bracket 2023 (millions).

- **Ground truth:** `{"chart_type": "histogram", "bins": [{"bin_start": 0, "bin_end": 10, "count": 40}, ...]}`
- **Schema:** `HistogramExtraction` → `bins: list[HistogramBin(bin_start, bin_end, count)]`
- **Slots:** `[("0-10", 40.0), ("10-20", 42.0), ...]`  label = `"{start}-{end}"`

### box
COVID-19 ICU length of stay by age group.

- **Ground truth:** `{"chart_type": "box", "groups": {"18-40": {"median": 4, "q1": 2, "q3": 7, "lower": 1, "upper": 14}, ...}}`
- **Schema:** `BoxPlotExtraction` → `groups: list[BoxGroup(group, median, q1, q3, lower_whisker, upper_whisker)]`
- **Slots:** `[("18-40|median", 4), ("18-40|q1", 2), ("18-40|q3", 7), ("18-40|lower", 1), ("18-40|upper", 14), ...]`

### heatmap
Monthly average temperature — 4 cities × 4 months.

- **Ground truth:** `{"chart_type": "heatmap", "rows": ["New York", ...], "cols": ["Jan", ...], "values": [[...], ...]}`
- **Schema:** `HeatmapExtraction` → `cells: list[HeatmapCell(row, col, value)]`
- **Slots:** `[("New York|Jan", -2.0), ("New York|Apr", 13.0), ...]`

### area
UK renewable energy generation Jan–Jun 2023.

- **Ground truth:** `{"chart_type": "area", "series": {"Wind": {"1": 18.4, ...}, "Solar": {...}}}`
- **Schema:** `AreaChartExtraction` → `data_points: list[AreaDataPoint(series, x, y)]`
- **Slots:** `[("Wind|1", 18.4), ("Solar|1", 1.2), ...]`  same pattern as line

---

## Extended 13

### dual_axis
US job openings (bars, left axis) vs unemployment rate (line, right axis).

- **Ground truth:** `{"chart_type": "dual_axis", "bar_values": {"Jan": 10.6, ...}, "line_values": {"Jan": 3.4, ...}}`
- **Schema:** `DualAxisExtraction` → `data_points: list[DualAxisPoint(x_label, bar_value, line_value)]`
- **Slots:** `[("Jan|bar", 10.6), ("Jan|line", 3.4), ...]`

### log_scale
Social platform MAU growth 2015–2022, log y-axis.

- **Ground truth:** `{"chart_type": "log_scale", "series": {"Platform A": {"2015": 325, ...}, ...}}`
- **Schema:** `LineChartExtraction` (reuses line schema)
- **Slots:** same pattern as `line` — `("{series}|{year}", value)`

### subplots_2x2
Quarterly regional sales $M — 4 quarters in a 2×2 grid.

- **Ground truth:** `{"chart_type": "subplots_2x2", "panels": [{"title": "Q1 2023", "data": {"North": 125, ...}}, ...]}`
- **Schema:** `SubplotsExtraction` → `panels: list[SubplotPanel(panel_title, data_points)]`
- **Slots:** `[("Q1 2023|North", 125.0), ("Q1 2023|South", 98.0), ...]`

### bubble
GDP/capita vs life expectancy vs population — 8 countries.

- **Ground truth:** `{"chart_type": "bubble", "points": [{"label": "USA", "x": 76399, "y": 76.1, "size": 335}, ...]}`
- **Schema:** `BubbleChartExtraction` → `points: list[BubblePoint(label, x, y, size)]`
- **Slots:** `[("USA|x", 76399), ("USA|y", 76.1), ("USA|size", 335), ...]`  3 slots per point

### waterfall
TechCorp FY-2023 P&L bridge.

- **Ground truth:** `{"chart_type": "waterfall", "bars": [{"label": "Net Revenue", "value": 1250, "bar_type": "absolute"}, ...]}`
- **Schema:** `WaterfallExtraction` → `bars: list[WaterfallBar(label, value, bar_type)]`
- **Slots:** `[("Net Revenue", 1250.0), ("Cost of Revenue", -520.0), ...]`

### candlestick
AAPL OHLC — 6 sessions Jan 2024.

- **Ground truth:** `{"chart_type": "candlestick", "candles": [{"date": "Jan 02", "open": 187.15, "high": 188.44, "low": 183.89, "close": 185.20}, ...]}`
- **Schema:** `CandlestickExtraction` → `candles: list[CandlestickCandle(date, open, high, low, close)]`
- **Slots:** `[("Jan 02|open", 187.15), ("Jan 02|high", 188.44), ("Jan 02|low", 183.89), ("Jan 02|close", 185.20), ...]`  4 slots per candle

### violin
Systolic BP reduction by treatment arm (4-arm RCT).

- **Ground truth:** `{"chart_type": "violin", "groups": {"Placebo": {"median": 2, "q1": -1, "q3": 5, "lower": -5, "upper": 10}, ...}}`
- **Schema:** `ViolinExtraction` → `groups: list[ViolinGroup(group, median, q1, q3, lower_whisker, upper_whisker)]`
- **Slots:** same pattern as `box` — `("{group}|{stat}", value)` for median, q1, q3, lower, upper
- **Note:** Violin data is synthetically generated from the stored summary stats (the actual violin shape varies per run due to `np.random.normal`). Only summary stats are in ground truth.

### radar
WEF competitiveness — USA vs Germany on 6 metrics.

- **Ground truth:** `{"chart_type": "radar", "series": {"USA": {"Infrastructure": 90, ...}, "Germany": {...}}}`
- **Schema:** `RadarExtraction` → `series: list[RadarSeries(series_name, data_points: list[RadarPoint(metric, value)])]`
- **Slots:** `[("USA|Infrastructure", 90), ("USA|Innovation", 88), ..., ("Germany|Infrastructure", 87), ...]`

### stacked_bar
Electricity generation mix by source — 5 economies, 2022.

- **Ground truth:** `{"chart_type": "stacked_bar", "series": {"Coal": {"USA": 778, ...}, ...}}`
- **Schema:** `StackedBarExtraction` → `data_points: list[StackedBarPoint(category, series, value)]`
- **Slots:** `[("Coal|USA", 778.0), ("Coal|China", 5300.0), ...]`  label = `"{series}|{category}"`

### stacked_bar_100
Population age structure (%) — 5 countries.

- **Ground truth:** `{"chart_type": "stacked_bar_100", "series": {"0-14 yrs": {"USA": 18.3, ...}, ...}}`
- **Schema:** `StackedBarExtraction` (reuses stacked_bar schema)
- **Slots:** same pattern as `stacked_bar`

### surface_3d
Mean January temperature over a 4×4 lat/lon grid.

- **Ground truth:** `{"chart_type": "surface_3d", "lat_labels": [...], "lon_labels": [...], "z": [[...], ...]}`
- **Schema:** `SurfaceExtraction` → `sample_points: list[SurfacePoint(x, y, z)]`
- **Slots:** `[(None, z_value), ...]`  — label-less; matched by numeric proximity only. 16 slots total.

### funnel
E-commerce conversion funnel — 6 stages.

- **Ground truth:** `{"chart_type": "funnel", "stages": [{"stage": "Website Visits", "value": 125420}, ...]}`
- **Schema:** `FunnelExtraction` → `stages: list[FunnelStage(stage, value)]`
- **Slots:** `[("Website Visits", 125420.0), ...]`

### treemap
S&P 500 sector market cap $B — mid-2024. Leaf nodes only in ground truth.

- **Ground truth:** `{"chart_type": "treemap", "nodes": [{"label": "Apple", "parent": "Technology", "value": 3050}, ...]}`
- **Schema:** `TreemapExtraction` → `nodes: list[TreemapNode(label, parent, value)]`
- **Slots:** `[("Technology|Apple", 3050.0), ("Technology|Microsoft", 3100.0), ...]`  label = `"{parent}|{label}"`

---

## Slot matching summary

| Slot key format | Chart types |
|---|---|
| `category` | `bar_vertical`, `bar_horizontal`, `pie`, `histogram`, `waterfall`, `funnel` |
| `"{series}\|{x}"` | `line`, `area`, `log_scale` |
| `point_label` (value = y) | `scatter` |
| `"{group}\|{stat}"` | `box`, `violin` (stats: median, q1, q3, lower, upper) |
| `"{row}\|{col}"` | `heatmap` |
| `"{x_label}\|bar"`, `"{x_label}\|line"` | `dual_axis` |
| `"{panel}\|{region}"` | `subplots_2x2` |
| `"{label}\|x"`, `"{label}\|y"`, `"{label}\|size"` | `bubble` |
| `"{date}\|{ohlc}"` | `candlestick` |
| `"{series}\|{metric}"` | `radar` |
| `"{series}\|{category}"` | `stacked_bar`, `stacked_bar_100` |
| `"{parent}\|{label}"` | `treemap` |
| `None` (value only) | `surface_3d` |
