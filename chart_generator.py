"""Generate Plotly charts with real-world ground-truth data and export as PNG images."""

from __future__ import annotations

import hashlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from prefect import task
from chart import Chart
from schemas import (
    AreaChartExtraction, AreaDataPoint,
    BarChartExtraction, BarDataPoint,
    BoxPlotExtraction, BoxGroup,
    BubbleChartExtraction, BubblePoint,
    CandlestickExtraction, CandlestickCandle,
    DualAxisExtraction, DualAxisPoint,
    FunnelExtraction, FunnelStage,
    HeatmapExtraction, HeatmapCell,
    HistogramExtraction, HistogramBin,
    LineChartExtraction, LineDataPoint,
    PieChartExtraction, PieSlice,
    RadarExtraction, RadarSeries, RadarPoint,
    ScatterChartExtraction, ScatterPoint,
    StackedBarExtraction, StackedBarPoint,
    SubplotsExtraction, SubplotPanel,
    SurfaceExtraction, SurfacePoint,
    TreemapExtraction, TreemapNode,
    ViolinExtraction, ViolinGroup,
    WaterfallExtraction, WaterfallBar,
)


np.random.seed(42)

# ── Shared layout helpers ─────────────────────────────────────────────────────

_L    = dict(font=dict(size=14), plot_bgcolor="white", paper_bgcolor="white",
             margin=dict(l=70, r=50, t=70, b=70))
_GRID = dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=1)


# ══════════════════════════════════════════════════════════════════════════════
#  Chart subclasses — labelled variants
# ══════════════════════════════════════════════════════════════════════════════

class BarVerticalChart(Chart):
    """Apple FY-2023 revenue by product segment ($B)."""
    def __init__(self):
        super().__init__(chart_type="bar_vertical")
        self.extraction_class = BarChartExtraction
        self.segments = ["iPhone", "Mac", "iPad", "Wearables", "Services"]
        self.values   = [200.6, 29.4, 28.3, 39.8, 85.2]
        self.ground_truth = BarChartExtraction(data_points=[BarDataPoint(label=s, value=v) for s, v in zip(self.segments, self.values)])

    def generate_plot(self) -> "BarVerticalChart":
        self.fig = go.Figure(go.Bar(
            x=self.segments, y=self.values, text=[f"${v}B" for v in self.values],
            textposition="outside", marker_color="steelblue",
        ))
        self.fig.update_layout(
            title="Apple FY-2023 Revenue by Segment ($B)",
            xaxis_title="Segment", yaxis_title="Revenue ($B)",
            xaxis=dict(showgrid=False), yaxis=dict(**_GRID, range=[0, 230]), **_L,
        )
        return self


class BarHorizontalChart(Chart):
    """Top-8 economies by GDP 2023 (World Bank, $T)."""
    def __init__(self):
        super().__init__(chart_type="bar_horizontal")
        self.extraction_class = BarChartExtraction
        self.countries = ["USA", "China", "Germany", "Japan", "India", "UK", "France", "Brazil"]
        self.gdp       = [27.4, 17.8, 4.5, 4.2, 3.7, 3.1, 2.9, 2.1]
        self.ground_truth = BarChartExtraction(data_points=[BarDataPoint(label=c, value=v) for c, v in zip(self.countries, self.gdp)])

    def generate_plot(self) -> "BarHorizontalChart":
        self.fig = go.Figure(go.Bar(
            y=self.countries, x=self.gdp, orientation="h",
            text=[f"${v}T" for v in self.gdp], textposition="outside",
            marker_color="coral",
        ))
        self.fig.update_layout(
            title="GDP 2023 – Top 8 Economies ($T)",
            xaxis_title="GDP ($T)", yaxis_title="Country",
            xaxis=dict(**_GRID, range=[0, 31]),
            yaxis=dict(showgrid=False, autorange="reversed"), **_L,
        )
        return self


class LineChart(Chart):
    """S&P 500 and NASDAQ 100 monthly closing values Jan–Dec 2023."""
    def __init__(self):
        super().__init__(chart_type="line")
        self.extraction_class = LineChartExtraction
        self.months = list(range(1, 13))
        self.sp500  = [4077, 3952, 4109, 4169, 4205, 4450, 4589, 4508, 4288, 4194, 4568, 4769]
        self.ndq    = [11622, 11319, 12221, 12981, 13181, 14344, 14762, 14347, 13507, 13138, 14225, 14926]
        self.ground_truth = LineChartExtraction(data_points=[
            LineDataPoint(series="S&P 500",    x=float(m), y=v) for m, v in zip(self.months, self.sp500)
        ] + [
            LineDataPoint(series="NASDAQ 100", x=float(m), y=v) for m, v in zip(self.months, self.ndq)
        ])

    def generate_plot(self) -> "LineChart":
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=self.months, y=self.sp500, mode="lines+markers", name="S&P 500",
                                      marker=dict(size=7), line=dict(width=2, color="steelblue")))
        self.fig.add_trace(go.Scatter(x=self.months, y=self.ndq, mode="lines+markers", name="NASDAQ 100",
                                      marker=dict(size=7), line=dict(width=2, color="tomato", dash="dash")))
        self.fig.update_layout(
            title="S&P 500 vs NASDAQ 100  —  Monthly Close 2023",
            xaxis_title="Month", yaxis_title="Index Level",
            xaxis=dict(**_GRID, dtick=1), yaxis=dict(**_GRID), **_L,
        )
        return self


class ScatterChart(Chart):
    """BMI vs systolic blood pressure — 12 de-identified clinical records."""
    def __init__(self):
        super().__init__(chart_type="scatter")
        self.extraction_class = ScatterChartExtraction
        self.labels = [f"P{i+1:02d}" for i in range(12)]
        self.bmi    = [22.1, 25.4, 28.7, 31.2, 19.8, 27.5, 34.1, 23.6, 29.8, 36.5, 24.9, 32.7]
        self.sbp    = [118,  128,  135,  145,  112,  132,  152,  122,  138,  160,  124,  148]
        self.ground_truth = ScatterChartExtraction(points=[
            ScatterPoint(x=b, y=s, label=l) for b, s, l in zip(self.bmi, self.sbp, self.labels)
        ])

    def generate_plot(self) -> "ScatterChart":
        self.fig = go.Figure(go.Scatter(
            x=self.bmi, y=self.sbp, mode="markers+text",
            text=self.labels, textposition="top center",
            marker=dict(size=10, color="darkorchid"),
        ))
        self.fig.update_layout(
            title="BMI vs Systolic Blood Pressure (Clinical)",
            xaxis_title="BMI (kg/m²)", yaxis_title="Systolic BP (mmHg)",
            xaxis=dict(**_GRID), yaxis=dict(**_GRID), **_L,
        )
        return self


class PieChart(Chart):
    """Desktop OS market share — StatCounter Jun 2024."""
    def __init__(self):
        super().__init__(chart_type="pie")
        self.extraction_class = PieChartExtraction
        self.labels = ["Windows", "macOS", "Linux", "ChromeOS", "Other"]
        self.values = [72.9, 15.4, 4.0, 3.1, 4.6]
        self.ground_truth = PieChartExtraction(slices=[PieSlice(label=l, percentage=v) for l, v in zip(self.labels, self.values)])

    def generate_plot(self) -> "PieChart":
        self.fig = go.Figure(go.Pie(
            labels=self.labels, values=self.values,
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent:.1%}",
        ))
        self.fig.update_layout(title="Desktop OS Market Share  —  Jun 2024", **_L)
        return self


class HistogramChart(Chart):
    """US population by age bracket, 2023 Census estimate (millions)."""
    def __init__(self):
        super().__init__(chart_type="histogram")
        self.extraction_class = HistogramExtraction
        self.bins   = [(0,10,40),(10,20,42),(20,30,46),(30,40,45),(40,50,43),(50,60,44),(60,70,38),(70,999,35)]
        self.labels = [f"{b[0]}-{b[1]}" if b[1] != 999 else "70+" for b in self.bins]
        self.counts = [b[2] for b in self.bins]
        self.ground_truth = HistogramExtraction(bins=[HistogramBin(bin_start=b[0], bin_end=b[1], count=b[2]) for b in self.bins])

    def generate_plot(self) -> "HistogramChart":
        self.fig = go.Figure(go.Bar(x=self.labels, y=self.counts, marker_color="seagreen",
                                    text=self.counts, textposition="outside"))
        self.fig.update_layout(
            title="US Population by Age Group 2023 (Millions)",
            xaxis_title="Age Group", yaxis_title="Population (M)",
            xaxis=dict(showgrid=False), yaxis=dict(**_GRID, range=[0, 52]),
            bargap=0.05, **_L,
        )
        return self


class BoxChart(Chart):
    """COVID-19 ICU length of stay (days) by age group."""
    def __init__(self):
        super().__init__(chart_type="box")
        self.extraction_class = BoxPlotExtraction
        self.groups = {
            "18-40": dict(median=4,  q1=2, q3=7,  lower=1, upper=14),
            "41-60": dict(median=7,  q1=4, q3=12, lower=2, upper=21),
            "61-75": dict(median=11, q1=6, q3=18, lower=2, upper=32),
            "76+":   dict(median=8,  q1=4, q3=14, lower=2, upper=25),
        }
        self.ground_truth = BoxPlotExtraction(groups=[
            BoxGroup(group=name, median=s["median"], q1=s["q1"], q3=s["q3"],
                     lower_whisker=s["lower"], upper_whisker=s["upper"])
            for name, s in self.groups.items()
        ])

    def generate_plot(self) -> "BoxChart":
        self.fig = go.Figure()
        for name, s in self.groups.items():
            self.fig.add_trace(go.Box(name=name,
                                      median=[s["median"]], q1=[s["q1"]], q3=[s["q3"]],
                                      lowerfence=[s["lower"]], upperfence=[s["upper"]]))
        self.fig.update_layout(
            title="COVID-19 ICU Length of Stay (Days) by Age Group",
            yaxis_title="Days", yaxis=dict(**_GRID), **_L,
        )
        return self


class HeatmapChart(Chart):
    """Monthly average temperature (°C) — 4 global cities, seasonal quarters."""
    def __init__(self):
        super().__init__(chart_type="heatmap")
        self.extraction_class = HeatmapExtraction
        self.cities = ["New York", "London", "Tokyo", "Dubai"]
        self.months = ["Jan", "Apr", "Jul", "Oct"]
        self.values = [[-2,13,29,14],[4,11,19,13],[5,15,27,18],[19,28,38,28]]
        self.ground_truth = HeatmapExtraction(cells=[
            HeatmapCell(row=city, col=month, value=val)
            for city, row_vals in zip(self.cities, self.values)
            for month, val in zip(self.months, row_vals)
        ])

    def generate_plot(self) -> "HeatmapChart":
        self.fig = go.Figure(go.Heatmap(
            z=self.values, x=self.months, y=self.cities,
            text=self.values, texttemplate="%{text}°C",
            colorscale="RdBu_r", showscale=True,
        ))
        self.fig.update_layout(
            title="Average Monthly Temperature (°C) by City",
            xaxis_title="Month", yaxis_title="City", **_L,
        )
        return self


class AreaChart(Chart):
    """UK renewable energy generation (TWh) by source  Jan–Jun 2023."""
    def __init__(self):
        super().__init__(chart_type="area")
        self.extraction_class = AreaChartExtraction
        self.months = [1, 2, 3, 4, 5, 6]
        self.wind   = [18.4, 17.1, 14.8, 12.3, 10.9, 11.6]
        self.solar  = [1.2,  1.8,  3.6,  5.8,  7.9,  9.2]
        self.ground_truth = AreaChartExtraction(data_points=[
            AreaDataPoint(series="Wind",  x=float(m), y=v) for m, v in zip(self.months, self.wind)
        ] + [
            AreaDataPoint(series="Solar", x=float(m), y=v) for m, v in zip(self.months, self.solar)
        ])

    def generate_plot(self) -> "AreaChart":
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=self.months, y=self.wind, fill="tozeroy", name="Wind",
                                      mode="lines", line=dict(color="steelblue")))
        self.fig.add_trace(go.Scatter(x=self.months, y=self.solar, fill="tonexty", name="Solar",
                                      mode="lines", line=dict(color="goldenrod")))
        self.fig.update_layout(
            title="UK Renewable Energy Generation (TWh)  Jan–Jun 2023",
            xaxis_title="Month", yaxis_title="Generation (TWh)",
            xaxis=dict(**_GRID, dtick=1), yaxis=dict(**_GRID), **_L,
        )
        return self


class DualAxisChart(Chart):
    """Monthly US unemployment rate (%) vs total job openings (M)  Jan–Jun 2023."""
    def __init__(self):
        super().__init__(chart_type="dual_axis")
        self.extraction_class = DualAxisExtraction
        self.months   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        self.unemp    = [3.4, 3.6, 3.5, 3.4, 3.7, 3.6]
        self.openings = [10.6, 10.4, 9.9, 9.6, 10.3, 9.8]
        self.ground_truth = DualAxisExtraction(
            bar_series_name="Job Openings (M)",
            line_series_name="Unemployment Rate (%)",
            data_points=[DualAxisPoint(x_label=m, bar_value=o, line_value=u)
                         for m, o, u in zip(self.months, self.openings, self.unemp)],
        )

    def generate_plot(self) -> "DualAxisChart":
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fig.add_trace(go.Bar(x=self.months, y=self.openings, name="Job Openings (M)",
                                  marker_color="steelblue", opacity=0.7), secondary_y=False)
        self.fig.add_trace(go.Scatter(x=self.months, y=self.unemp, name="Unemployment Rate (%)",
                                      mode="lines+markers", line=dict(color="tomato", width=2),
                                      marker=dict(size=8)), secondary_y=True)
        self.fig.update_yaxes(title_text="Job Openings (Millions)", secondary_y=False,
                              **_GRID, range=[0, 13])
        self.fig.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=True,
                              **_GRID, range=[0, 6])
        self.fig.update_layout(
            title="US Job Openings vs Unemployment Rate  Jan–Jun 2023",
            xaxis_title="Month", **_L,
        )
        return self


class LogScaleChart(Chart):
    """Exponential user growth for two social platforms 2015–2022 (millions)."""
    def __init__(self):
        super().__init__(chart_type="log_scale")
        self.extraction_class = LineChartExtraction
        self.years  = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        self.plat_a = [325, 550, 920, 1500, 2070, 2630, 3050, 3370]
        self.plat_b = [110, 195, 328, 525,  785, 1082, 1300, 1450]
        self.ground_truth = LineChartExtraction(data_points=[
            LineDataPoint(series="Platform A", x=float(y), y=v) for y, v in zip(self.years, self.plat_a)
        ] + [
            LineDataPoint(series="Platform B", x=float(y), y=v) for y, v in zip(self.years, self.plat_b)
        ])

    def generate_plot(self) -> "LogScaleChart":
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(x=self.years, y=self.plat_a, mode="lines+markers",
                                      name="Platform A", marker=dict(size=8),
                                      line=dict(width=2, color="royalblue")))
        self.fig.add_trace(go.Scatter(x=self.years, y=self.plat_b, mode="lines+markers",
                                      name="Platform B", marker=dict(size=8),
                                      line=dict(width=2, color="darkorange", dash="dash")))
        self.fig.update_layout(
            title="Social Platform MAU Growth 2015–2022 (Log Scale, Millions)",
            xaxis_title="Year", yaxis_title="Monthly Active Users (M, log scale)",
            xaxis=dict(**_GRID, dtick=1), yaxis=dict(**_GRID, type="log"), **_L,
        )
        return self


class Subplots2x2Chart(Chart):
    """Quarterly regional sales ($M) — 4 quarters × 4 regions (2×2 grid)."""
    def __init__(self):
        super().__init__(chart_type="subplots_2x2")
        self.extraction_class = SubplotsExtraction
        self.regions  = ["North", "South", "East", "West"]
        self.quarters = {
            "Q1 2023": [125, 98,  145, 167],
            "Q2 2023": [138, 112, 156, 182],
            "Q3 2023": [152, 128, 168, 195],
            "Q4 2023": [175, 145, 189, 220],
        }
        self.ground_truth = SubplotsExtraction(panels=[
            SubplotPanel(panel_title=q, data_points=[BarDataPoint(label=r, value=v) for r, v in zip(self.regions, vals)])
            for q, vals in self.quarters.items()
        ])

    def generate_plot(self) -> "Subplots2x2Chart":
        self.fig = make_subplots(rows=2, cols=2, subplot_titles=list(self.quarters.keys()))
        for (r, c), (q, vals) in zip([(1,1),(1,2),(2,1),(2,2)], self.quarters.items()):
            self.fig.add_trace(go.Bar(x=self.regions, y=vals, name=q,
                                      text=vals, textposition="outside",
                                      showlegend=False), row=r, col=c)
        self.fig.update_layout(title="Quarterly Regional Sales ($M) 2023", **_L)
        self.fig.update_yaxes(**_GRID)
        return self


class BubbleChart(Chart):
    """GDP per capita vs life expectancy vs population — 8 countries, World Bank 2022."""
    def __init__(self):
        super().__init__(chart_type="bubble")
        self.extraction_class = BubbleChartExtraction
        self.countries = ["USA",   "China", "India", "Germany", "Japan",  "Brazil", "Nigeria", "Mexico"]
        self.gdp_pc    = [76399,   12720,   2389,    48718,     39313,    8917,     2184,      10046]
        self.life_exp  = [76.1,    78.2,    67.7,    80.6,      84.3,     74.0,     53.6,      75.1]
        self.pop_m     = [335,     1412,    1417,    84,        125,      215,      219,       130]
        self.ground_truth = BubbleChartExtraction(points=[
            BubblePoint(label=c, x=g, y=l, size=p)
            for c, g, l, p in zip(self.countries, self.gdp_pc, self.life_exp, self.pop_m)
        ])

    def generate_plot(self) -> "BubbleChart":
        self.fig = go.Figure(go.Scatter(
            x=self.gdp_pc, y=self.life_exp, mode="markers+text",
            text=self.countries, textposition="top center",
            marker=dict(size=[p / 20 for p in self.pop_m], color=self.gdp_pc, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="GDP/capita")),
        ))
        self.fig.update_layout(
            title="GDP per Capita vs Life Expectancy (bubble = population, M)  2022",
            xaxis_title="GDP per Capita ($)", yaxis_title="Life Expectancy (years)",
            xaxis=dict(**_GRID), yaxis=dict(**_GRID), **_L,
        )
        return self


class WaterfallChart(Chart):
    """TechCorp FY-2023 P&L waterfall ($M)."""
    def __init__(self):
        super().__init__(chart_type="waterfall")
        self.extraction_class = WaterfallExtraction
        self.bars = [
            ("Net Revenue",        1250, "absolute"),
            ("Cost of Revenue",    -520, "relative"),
            ("Gross Profit",        730, "total"),
            ("R&D",                -185, "relative"),
            ("Sales & Marketing",  -145, "relative"),
            ("G&A",                 -65, "relative"),
            ("Operating Income",    335, "total"),
            ("Tax (20%)",           -67, "relative"),
            ("Net Income",          268, "total"),
        ]
        self.ground_truth = WaterfallExtraction(bars=[WaterfallBar(label=b[0], value=b[1], bar_type=b[2]) for b in self.bars])

    def generate_plot(self) -> "WaterfallChart":
        self.fig = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=[b[2] for b in self.bars],
            x=[b[0] for b in self.bars],
            y=[b[1] for b in self.bars],
            text=[f"${abs(b[1])}M" for b in self.bars], textposition="outside",
            connector=dict(line=dict(color="gray")),
        ))
        self.fig.update_layout(
            title="TechCorp FY-2023 P&L Waterfall ($M)",
            yaxis_title="$M", yaxis=dict(**_GRID), **_L,
        )
        return self


class CandlestickChart(Chart):
    """AAPL-like stock OHLC data — 6 trading sessions Jan 2024."""
    def __init__(self):
        super().__init__(chart_type="candlestick")
        self.extraction_class = CandlestickExtraction
        self.candles = [
            ("Jan 02", 187.15, 188.44, 183.89, 185.20),
            ("Jan 03", 184.22, 185.88, 183.43, 184.25),
            ("Jan 04", 182.15, 183.09, 180.63, 181.91),
            ("Jan 05", 181.99, 182.32, 179.25, 181.18),
            ("Jan 08", 182.09, 188.74, 181.99, 188.32),
            ("Jan 09", 188.68, 190.00, 186.37, 185.64),
        ]
        self.ground_truth = CandlestickExtraction(candles=[
            CandlestickCandle(date=c[0], open=c[1], high=c[2], low=c[3], close=c[4]) for c in self.candles
        ])

    def generate_plot(self) -> "CandlestickChart":
        self.fig = go.Figure(go.Candlestick(
            x=[c[0] for c in self.candles],
            open =[c[1] for c in self.candles],
            high =[c[2] for c in self.candles],
            low  =[c[3] for c in self.candles],
            close=[c[4] for c in self.candles],
        ))
        self.fig.update_layout(
            title="AAPL Stock Price  —  Jan 2024 (OHLC)",
            yaxis_title="Price ($)", xaxis_rangeslider_visible=False,
            yaxis=dict(**_GRID), **_L,
        )
        return self


class ViolinChart(Chart):
    """Systolic BP reduction (mmHg) in a 4-arm RCT (placebo-controlled)."""
    def __init__(self):
        super().__init__(chart_type="violin")
        self.extraction_class = ViolinExtraction
        self.groups = {
            "Placebo":    dict(median=2,  q1=-1, q3=5,  lower=-5, upper=10),
            "Drug 10 mg": dict(median=8,  q1=5,  q3=12, lower=0,  upper=18),
            "Drug 20 mg": dict(median=14, q1=10, q3=18, lower=4,  upper=25),
            "Drug 40 mg": dict(median=19, q1=15, q3=23, lower=8,  upper=30),
        }
        self.ground_truth = ViolinExtraction(groups=[
            ViolinGroup(group=name, median=s["median"], q1=s["q1"], q3=s["q3"],
                        lower_whisker=s["lower"], upper_whisker=s["upper"])
            for name, s in self.groups.items()
        ])

    def generate_plot(self) -> "ViolinChart":
        self.fig = go.Figure()
        for (name, s), col in zip(self.groups.items(), ["#636EFA","#EF553B","#00CC96","#AB63FA"]):
            data = np.clip(np.random.normal(s["median"], (s["q3"] - s["q1"]) / 1.35, 80),
                           s["lower"], s["upper"])
            self.fig.add_trace(go.Violin(y=data, name=name, box_visible=True,
                                         meanline_visible=True,
                                         line_color=col, fillcolor=col, opacity=0.6))
        self.fig.update_layout(
            title="Systolic BP Reduction (mmHg) by Treatment Group (RCT)",
            yaxis_title="BP Reduction (mmHg)",
            yaxis=dict(**_GRID), violinmode="group", **_L,
        )
        return self


class RadarChart(Chart):
    """WEF Global Competitiveness metrics — USA vs Germany (score 0–100)."""
    def __init__(self):
        super().__init__(chart_type="radar")
        self.extraction_class = RadarExtraction
        self.metrics  = ["Infrastructure", "Innovation", "Institutions", "Macro-stability", "Health", "Skills"]
        self.usa_vals = [90, 88, 74, 72, 83, 84]
        self.ger_vals = [87, 85, 82, 79, 88, 86]
        self.ground_truth = RadarExtraction(series=[
            RadarSeries(series_name="USA",     data_points=[RadarPoint(metric=m, value=v) for m, v in zip(self.metrics, self.usa_vals)]),
            RadarSeries(series_name="Germany", data_points=[RadarPoint(metric=m, value=v) for m, v in zip(self.metrics, self.ger_vals)]),
        ])

    def generate_plot(self) -> "RadarChart":
        theta = self.metrics + [self.metrics[0]]
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatterpolar(r=self.usa_vals + [self.usa_vals[0]], theta=theta,
                                           fill="toself", name="USA",
                                           line=dict(color="steelblue")))
        self.fig.add_trace(go.Scatterpolar(r=self.ger_vals + [self.ger_vals[0]], theta=theta,
                                           fill="toself", name="Germany",
                                           line=dict(color="tomato")))
        self.fig.update_layout(
            title="WEF Competitiveness Scores  —  USA vs Germany",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], dtick=20)), **_L,
        )
        return self


class StackedBarChart(Chart):
    """Electricity generation mix by source (TWh) — 5 major economies, 2022."""
    def __init__(self):
        super().__init__(chart_type="stacked_bar")
        self.extraction_class = StackedBarExtraction
        self.countries = ["USA", "China", "Germany", "France", "UK"]
        self.sources   = {
            "Coal":       [778,  5300, 108,  7,   7],
            "Gas":        [1686,  253,  68,  24, 100],
            "Nuclear":    [772,   411,  32, 295,  43],
            "Renewables": [908,  2200, 254, 130, 134],
        }
        self.ground_truth = StackedBarExtraction(data_points=[
            StackedBarPoint(category=c, series=src, value=v)
            for src, vals in self.sources.items()
            for c, v in zip(self.countries, vals)
        ])

    def generate_plot(self) -> "StackedBarChart":
        colors = {"Coal": "#5a5a5a", "Gas": "#f4a261", "Nuclear": "#e9c46a", "Renewables": "#2a9d8f"}
        self.fig = go.Figure()
        for src, vals in self.sources.items():
            self.fig.add_trace(go.Bar(name=src, x=self.countries, y=vals, marker_color=colors[src]))
        self.fig.update_layout(
            title="Electricity Generation Mix by Source (TWh, 2022)",
            xaxis_title="Country", yaxis_title="Generation (TWh)",
            barmode="stack", yaxis=dict(**_GRID), **_L,
        )
        return self


class StackedBar100Chart(Chart):
    """Population age structure (%) — 5 countries with contrasting demographics."""
    def __init__(self):
        super().__init__(chart_type="stacked_bar_100")
        self.extraction_class = StackedBarExtraction
        self.countries  = ["USA", "Japan", "India", "Nigeria", "Germany"]
        self.age_groups = {
            "0-14 yrs":  [18.3, 11.9, 26.3, 43.6, 13.5],
            "15-64 yrs": [64.8, 59.4, 67.3, 53.7, 64.8],
            "65+ yrs":   [16.9, 28.7,  6.4,  2.7, 21.7],
        }
        self.ground_truth = StackedBarExtraction(data_points=[
            StackedBarPoint(category=c, series=grp, value=v)
            for grp, vals in self.age_groups.items()
            for c, v in zip(self.countries, vals)
        ])

    def generate_plot(self) -> "StackedBar100Chart":
        colors = {"0-14 yrs": "#2196F3", "15-64 yrs": "#4CAF50", "65+ yrs": "#FF5722"}
        self.fig = go.Figure()
        for grp, vals in self.age_groups.items():
            self.fig.add_trace(go.Bar(name=grp, x=self.countries, y=vals,
                                      marker_color=colors[grp],
                                      text=[f"{v}%" for v in vals], textposition="inside"))
        self.fig.update_layout(
            title="Population Age Structure (%)  —  Selected Countries 2023",
            xaxis_title="Country", yaxis_title="Share (%)",
            barmode="stack", yaxis=dict(**_GRID, range=[0, 102]), **_L,
        )
        return self


class Surface3DChart(Chart):
    """Climatological surface: mean temperature (°C) over a 4×4 lat/lon grid, Jan."""
    def __init__(self):
        super().__init__(chart_type="surface_3d")
        self.extraction_class = SurfaceExtraction
        self.lat_labels  = ["60°N", "30°N", "0°",  "30°S"]
        self.lon_labels  = ["0°E",  "30°E", "60°E", "90°E"]
        self.lat_numeric = [60, 30, 0, -30]
        self.lon_numeric = [0, 30, 60, 90]
        self.z = [[-8,-5,3,12],[8,12,18,24],[26,28,27,25],[22,25,28,20]]
        self.ground_truth = SurfaceExtraction(sample_points=[
            SurfacePoint(x=lon, y=lat, z=val)
            for lat, row in zip(self.lat_numeric, self.z)
            for lon, val in zip(self.lon_numeric, row)
        ])

    def generate_plot(self) -> "Surface3DChart":
        self.fig = go.Figure(go.Surface(
            x=self.lon_labels, y=self.lat_labels, z=self.z,
            colorscale="RdBu_r", showscale=True, colorbar=dict(title="°C"),
            text=[[f"{v}°C" for v in row] for row in self.z],
            hovertemplate="Lat: %{y}<br>Lon: %{x}<br>Temp: %{z}°C",
        ))
        self.fig.update_layout(
            title="Mean January Temperature (°C) — Lat/Lon Grid",
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Temp (°C)"),
            **_L,
        )
        return self


class FunnelChart(Chart):
    """E-commerce conversion funnel — realistic SaaS metrics."""
    def __init__(self):
        super().__init__(chart_type="funnel")
        self.extraction_class = FunnelExtraction
        self.stages = ["Website Visits", "Product Views", "Add to Cart",
                       "Checkout Started", "Payment Entered", "Orders Completed"]
        self.values = [125420, 84350, 31240, 18560, 12380, 9845]
        self.ground_truth = FunnelExtraction(stages=[FunnelStage(stage=s, value=v) for s, v in zip(self.stages, self.values)])

    def generate_plot(self) -> "FunnelChart":
        self.fig = go.Figure(go.Funnel(
            y=self.stages, x=self.values,
            text=[f"{v:,}" for v in self.values],
            textinfo="text+percent initial",
            marker=dict(color=["#1f77b4","#2196F3","#42A5F5","#64B5F6","#90CAF9","#BBDEFB"]),
        ))
        self.fig.update_layout(title="E-Commerce Conversion Funnel (raw visitor counts)", **_L)
        return self


class TreemapChart(Chart):
    """S&P 500 approximate sector market cap ($B) with top constituents, mid-2024."""
    def __init__(self):
        super().__init__(chart_type="treemap")
        self.extraction_class = TreemapExtraction
        self.nodes = [
            ("S&P 500",       "",              0),
            ("Technology",    "S&P 500",       0),
            ("Financials",    "S&P 500",       0),
            ("Healthcare",    "S&P 500",       0),
            ("Consumer Disc", "S&P 500",       0),
            ("Industrials",   "S&P 500",       0),
            ("Energy",        "S&P 500",       0),
            ("Apple",         "Technology", 3050),
            ("Microsoft",     "Technology", 3100),
            ("NVIDIA",        "Technology", 2800),
            ("Meta",          "Technology", 1250),
            ("Tech Others",   "Technology", 2600),
            ("JPMorgan",      "Financials",  590),
            ("Berkshire",     "Financials",  870),
            ("Visa",          "Financials",  540),
            ("Fin Others",    "Financials", 2200),
            ("UnitedHealth",  "Healthcare",  480),
            ("Eli Lilly",     "Healthcare",  760),
            ("HC Others",     "Healthcare", 3360),
            ("Amazon",        "Consumer Disc", 1950),
            ("Tesla",         "Consumer Disc",  680),
            ("CD Others",     "Consumer Disc", 1570),
            ("Caterpillar",   "Industrials",   180),
            ("Honeywell",     "Industrials",   130),
            ("Ind Others",    "Industrials",  2890),
            ("ExxonMobil",    "Energy",  490),
            ("Chevron",       "Energy",  280),
            ("Energy Others", "Energy", 2030),
        ]
        sectors    = {"Technology","Financials","Healthcare","Consumer Disc","Industrials","Energy"}
        leaf_nodes = [n for n in self.nodes if n[1] in sectors and n[2] > 0]
        self.ground_truth = TreemapExtraction(nodes=[TreemapNode(label=n[0], parent=n[1], value=n[2]) for n in leaf_nodes])

    def generate_plot(self) -> "TreemapChart":
        self.fig = go.Figure(go.Treemap(
            labels=[n[0] for n in self.nodes],
            parents=[n[1] for n in self.nodes],
            values=[n[2] for n in self.nodes],
            textinfo="label+value",
            texttemplate="%{label}<br>$%{value}B",
            root_color="lightgrey",
        ))
        self.fig.update_layout(title="S&P 500 Sector Market Cap ($B, mid-2024)", **_L)
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  No-label variants (same data, no bar text annotations)
# ══════════════════════════════════════════════════════════════════════════════

class BarVerticalNoLabelChart(BarVerticalChart):
    """Same data as BarVerticalChart but no bar value annotations."""
    def __init__(self):
        super().__init__()
        self.chart_type = "bar_vertical_nolabel"

    def generate_plot(self) -> "BarVerticalNoLabelChart":
        self.fig = go.Figure(go.Bar(x=self.segments, y=self.values, marker_color="steelblue"))
        self.fig.update_layout(
            title="Apple FY-2023 Revenue by Segment ($B)",
            xaxis_title="Segment", yaxis_title="Revenue ($B)",
            xaxis=dict(showgrid=False), yaxis=dict(**_GRID, range=[0, 230]), **_L,
        )
        return self


class BarHorizontalNoLabelChart(BarHorizontalChart):
    """Same data as BarHorizontalChart but no bar value annotations."""
    def __init__(self):
        super().__init__()
        self.chart_type = "bar_horizontal_nolabel"

    def generate_plot(self) -> "BarHorizontalNoLabelChart":
        self.fig = go.Figure(go.Bar(
            y=self.countries, x=self.gdp, orientation="h", marker_color="coral",
        ))
        self.fig.update_layout(
            title="GDP 2023 – Top 8 Economies ($T)",
            xaxis_title="GDP ($T)", yaxis_title="Country",
            xaxis=dict(**_GRID, range=[0, 31]),
            yaxis=dict(showgrid=False, autorange="reversed"), **_L,
        )
        return self


class HistogramNoLabelChart(HistogramChart):
    """Same data as HistogramChart but no bar value annotations."""
    def __init__(self):
        super().__init__()
        self.chart_type = "histogram_nolabel"

    def generate_plot(self) -> "HistogramNoLabelChart":
        self.fig = go.Figure(go.Bar(x=self.labels, y=self.counts, marker_color="seagreen"))
        self.fig.update_layout(
            title="US Population by Age Group 2023 (Millions)",
            xaxis_title="Age Group", yaxis_title="Population (M)",
            xaxis=dict(showgrid=False), yaxis=dict(**_GRID, range=[0, 52]),
            bargap=0.05, **_L,
        )
        return self


class Subplots2x2NoLabelChart(Subplots2x2Chart):
    """Same data as Subplots2x2Chart but no bar value annotations."""
    def __init__(self):
        super().__init__()
        self.chart_type = "subplots_2x2_nolabel"

    def generate_plot(self) -> "Subplots2x2NoLabelChart":
        self.fig = make_subplots(rows=2, cols=2, subplot_titles=list(self.quarters.keys()))
        for (r, c), (q, vals) in zip([(1,1),(1,2),(2,1),(2,2)], self.quarters.items()):
            self.fig.add_trace(go.Bar(x=self.regions, y=vals, name=q, showlegend=False), row=r, col=c)
        self.fig.update_layout(title="Quarterly Regional Sales ($M) 2023", **_L)
        self.fig.update_yaxes(**_GRID)
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  New chart types
# ══════════════════════════════════════════════════════════════════════════════

class LogLogScatterChart(Chart):
    """Revenue vs market cap for 15 major tech companies — both axes log scale."""
    def __init__(self):
        super().__init__(chart_type="log_log_scatter")
        self.extraction_class = ScatterChartExtraction
        self.companies = ["Apple", "Microsoft", "Alphabet", "Amazon", "Meta",
                          "NVIDIA", "Tesla", "Netflix", "Salesforce", "Oracle",
                          "Adobe", "Qualcomm", "Intel", "AMD", "Broadcom"]
        self.revenue = [394, 212, 283, 513, 117, 61, 97, 33, 31, 50, 19, 36, 54, 23, 35]   # $B
        self.mktcap  = [2990, 2850, 1740, 1540, 1220, 1220, 790, 230, 200, 330, 260, 165, 180, 205, 510]  # $B
        self.ground_truth = ScatterChartExtraction(points=[
            ScatterPoint(x=float(r), y=float(m), label=c)
            for c, r, m in zip(self.companies, self.revenue, self.mktcap)
        ])

    def generate_plot(self) -> "LogLogScatterChart":
        self.fig = go.Figure(go.Scatter(
            x=self.revenue, y=self.mktcap, mode="markers+text",
            text=self.companies, textposition="top center",
            marker=dict(size=10, color="royalblue"),
        ))
        self.fig.update_layout(
            title="Tech Companies: Revenue vs Market Cap 2023 ($B, log-log scale)",
            xaxis_title="Annual Revenue ($B, log scale)",
            yaxis_title="Market Cap ($B, log scale)",
            xaxis=dict(**_GRID, type="log"),
            yaxis=dict(**_GRID, type="log"),
            **_L,
        )
        return self


class LineMultiSeriesChart(Chart):
    """Monthly cumulative performance (base=100) for 8 asset classes, 2023."""
    def __init__(self):
        super().__init__(chart_type="line_multi_series")
        self.extraction_class = LineChartExtraction
        self.months = list(range(1, 13))
        self.series = {
            "US Large Cap":   [100, 97, 105, 108, 109, 116, 120, 116, 110, 107, 116, 124],
            "Intl Developed": [100, 99, 105, 108, 109, 111, 112, 108, 104, 101, 107, 113],
            "Emerging Mkts":  [100, 98, 102, 104, 100, 100, 100,  96,  92,  90,  94, 101],
            "US Bonds":       [100, 97,  96,  95,  93,  95,  95,  93,  91,  90,  93,  97],
            "Real Estate":    [100, 96, 101, 102, 100, 102, 103,  99,  95,  94,  98, 105],
            "Gold":           [100,102, 107, 108,  99,  96,  94,  93,  96,  99,  97, 100],
            "Commodities":    [100, 94,  97,  97,  92,  92,  91,  92,  95,  89,  83,  84],
            "Cash (T-Bills)": [100,100, 101, 101, 102, 102, 103, 103, 104, 104, 105, 105],
        }
        self.ground_truth = LineChartExtraction(data_points=[
            LineDataPoint(series=name, x=float(m), y=float(v))
            for name, vals in self.series.items()
            for m, v in zip(self.months, vals)
        ])

    def generate_plot(self) -> "LineMultiSeriesChart":
        colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
        self.fig = go.Figure()
        for (name, vals), color in zip(self.series.items(), colors):
            self.fig.add_trace(go.Scatter(
                x=self.months, y=vals, mode="lines+markers",
                name=name, marker=dict(size=5), line=dict(width=1.5, color=color),
            ))
        self.fig.update_layout(
            title="Asset Class Performance 2023 (Monthly, Base = 100)",
            xaxis_title="Month", yaxis_title="Cumulative Return (Base = 100)",
            xaxis=dict(**_GRID, dtick=1), yaxis=dict(**_GRID), **_L,
        )
        return self


class DenseScatterChart(Chart):
    """House size vs sale price — 50 labeled properties."""
    def __init__(self):
        super().__init__(chart_type="dense_scatter")
        self.extraction_class = ScatterChartExtraction
        self.sqft = [
             850,  920, 1050, 1100, 1200, 1250, 1300, 1350, 1400, 1420,
            1500, 1550, 1600, 1650, 1700, 1720, 1750, 1800, 1820, 1850,
            1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350,
            2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850,
            2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3400,
        ]
        self.price = [
            145, 162, 180, 195, 210, 225, 235, 248, 262, 270,
            290, 302, 318, 330, 345, 358, 365, 380, 390, 400,
            415, 430, 445, 458, 472, 485, 498, 512, 525, 540,
            555, 568, 582, 595, 610, 625, 638, 652, 665, 680,
            695, 708, 722, 735, 750, 765, 778, 792, 805, 835,
        ]
        self.labels = [f"H{i+1:02d}" for i in range(50)]
        self.ground_truth = ScatterChartExtraction(points=[
            ScatterPoint(x=float(s), y=float(p), label=l)
            for s, p, l in zip(self.sqft, self.price, self.labels)
        ])

    def generate_plot(self) -> "DenseScatterChart":
        self.fig = go.Figure(go.Scatter(
            x=self.sqft, y=self.price, mode="markers+text",
            text=self.labels, textposition="top center",
            marker=dict(size=7, color="darkorchid"),
            textfont=dict(size=8),
        ))
        self.fig.update_layout(
            title="House Size vs Sale Price — 50 Properties ($K)",
            xaxis_title="Size (sq ft)", yaxis_title="Sale Price ($K)",
            xaxis=dict(**_GRID), yaxis=dict(**_GRID), **_L,
        )
        return self


class DenseLineChart(Chart):
    """Monthly mean temperature (°C) — Chicago vs Miami, Jan 2021 – Dec 2023 (36 pts/series)."""
    def __init__(self):
        super().__init__(chart_type="dense_line")
        self.extraction_class = LineChartExtraction
        self.months  = list(range(1, 37))
        self.chicago = [
            -5, -2,  3, 10, 17, 22, 25, 24, 19, 12,  5, -2,   # 2021
            -5, -4,  4, 11, 17, 22, 26, 25, 20, 13,  5, -2,   # 2022
            -6, -4,  5, 12, 18, 23, 26, 25, 20, 14,  6, -1,   # 2023
        ]
        self.miami = [
            21, 22, 24, 26, 27, 29, 30, 30, 29, 28, 25, 22,   # 2021
            22, 23, 24, 26, 28, 29, 31, 31, 30, 28, 25, 23,   # 2022
            22, 23, 25, 27, 28, 30, 31, 31, 30, 28, 26, 23,   # 2023
        ]
        self.ground_truth = LineChartExtraction(data_points=[
            LineDataPoint(series="Chicago", x=float(m), y=float(t)) for m, t in zip(self.months, self.chicago)
        ] + [
            LineDataPoint(series="Miami", x=float(m), y=float(t)) for m, t in zip(self.months, self.miami)
        ])

    def generate_plot(self) -> "DenseLineChart":
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(
            x=self.months, y=self.chicago, mode="lines+markers",
            name="Chicago", marker=dict(size=4), line=dict(width=1.5, color="steelblue"),
        ))
        self.fig.add_trace(go.Scatter(
            x=self.months, y=self.miami, mode="lines+markers",
            name="Miami", marker=dict(size=4), line=dict(width=1.5, color="tomato"),
        ))
        self.fig.update_layout(
            title="Monthly Mean Temperature (°C) — Chicago vs Miami, 2021–2023",
            xaxis_title="Month index (1 = Jan 2021, 36 = Dec 2023)",
            yaxis_title="Mean Temperature (°C)",
            xaxis=dict(**_GRID, dtick=6), yaxis=dict(**_GRID), **_L,
        )
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  Registry & public API
# ══════════════════════════════════════════════════════════════════════════════

CHART_MAPPINGS: dict[str, type[Chart]] = {
    # Labelled originals
    "bar_vertical":         BarVerticalChart,
    "bar_horizontal":       BarHorizontalChart,
    "line":                 LineChart,
    "scatter":              ScatterChart,
    "pie":                  PieChart,
    "histogram":            HistogramChart,
    "box":                  BoxChart,
    "heatmap":              HeatmapChart,
    "area":                 AreaChart,
    "dual_axis":            DualAxisChart,
    "log_scale":            LogScaleChart,
    "subplots_2x2":         Subplots2x2Chart,
    "bubble":               BubbleChart,
    "waterfall":            WaterfallChart,
    "candlestick":          CandlestickChart,
    "violin":               ViolinChart,
    "radar":                RadarChart,
    "stacked_bar":          StackedBarChart,
    "stacked_bar_100":      StackedBar100Chart,
    "surface_3d":           Surface3DChart,
    "funnel":               FunnelChart,
    "treemap":              TreemapChart,
    # No-label variants
    "bar_vertical_nolabel":    BarVerticalNoLabelChart,
    "bar_horizontal_nolabel":  BarHorizontalNoLabelChart,
    "histogram_nolabel":       HistogramNoLabelChart,
    "subplots_2x2_nolabel":    Subplots2x2NoLabelChart,
    # New chart types
    "log_log_scatter":     LogLogScatterChart,
    "line_multi_series":   LineMultiSeriesChart,
    "dense_scatter":       DenseScatterChart,
    "dense_line":          DenseLineChart,
}


def chart_factory(chart_type: str) -> Chart:
    """Instantiate and return the Chart subclass for the given chart type."""
    return CHART_MAPPINGS[chart_type]()


def _chart_cache_key(_context, parameters: dict) -> str:
    chart_type = parameters["chart_type"]
    file_hash = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    return f"{chart_type}-{file_hash}"


@task(log_prints=True, cache_key_fn=_chart_cache_key)
def generate_and_save_chart(chart_type: str) -> str:
    chart_class = chart_factory(chart_type=chart_type)
    chart_class.generate_plot()
    chart_class.save_images()
    return chart_type


def generate_all_charts() -> dict[str, Chart]:
    """Return all charts with figures and ground truth populated."""
    return {ct: chart_factory(ct) for ct in CHART_MAPPINGS}


def generate_and_save_all() -> dict[str, Chart]:
    """Generate every chart, save all resolutions to disk, return the Chart objects."""
    charts = generate_all_charts()
    for chart in charts.values():
        image_paths = chart.save_images()
        print(f"  OK {chart.chart_type:20s}  {list(image_paths.keys())}")
    return charts
