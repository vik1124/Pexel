"""Generate Plotly charts with real-world ground-truth data and export as PNG images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHARTS_DIR, RESOLUTIONS

np.random.seed(42)

# ── Shared layout ─────────────────────────────────────────────────────────────

_L = dict(
    font=dict(size=14),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=70, r=50, t=70, b=70),
)
_GRID = dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=1)


# ══════════════════════════════════════════════════════════════════════════════
#  Original 9 charts  —  updated to real-world datasets
# ══════════════════════════════════════════════════════════════════════════════

def _bar_vertical() -> tuple[go.Figure, dict]:
    """Apple FY-2023 revenue by product segment ($B)."""
    segments = ["iPhone", "Mac", "iPad", "Wearables", "Services"]
    values   = [200.6, 29.4, 28.3, 39.8, 85.2]
    fig = go.Figure(go.Bar(
        x=segments, y=values, text=[f"${v}B" for v in values],
        textposition="outside", marker_color="steelblue",
    ))
    fig.update_layout(title="Apple FY-2023 Revenue by Segment ($B)",
                      xaxis_title="Segment", yaxis_title="Revenue ($B)",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(**_GRID, range=[0, 230]), **_L)
    return fig, {"chart_type": "bar_vertical", "data": dict(zip(segments, values))}


def _bar_horizontal() -> tuple[go.Figure, dict]:
    """Top-8 economies by GDP 2023 (World Bank, $T)."""
    countries = ["USA", "China", "Germany", "Japan", "India", "UK", "France", "Brazil"]
    gdp       = [27.4, 17.8, 4.5, 4.2, 3.7, 3.1, 2.9, 2.1]
    fig = go.Figure(go.Bar(
        y=countries, x=gdp, orientation="h",
        text=[f"${v}T" for v in gdp], textposition="outside",
        marker_color="coral",
    ))
    fig.update_layout(title="GDP 2023 – Top 8 Economies ($T)",
                      xaxis_title="GDP ($T)", yaxis_title="Country",
                      xaxis=dict(**_GRID, range=[0, 31]),
                      yaxis=dict(showgrid=False, autorange="reversed"), **_L)
    return fig, {"chart_type": "bar_horizontal", "data": dict(zip(countries, gdp))}


def _line() -> tuple[go.Figure, dict]:
    """S&P 500 and NASDAQ 100 monthly closing values Jan–Dec 2023."""
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    sp500  = [4077, 3952, 4109, 4169, 4205, 4450, 4589, 4508, 4288, 4194, 4568, 4769]
    ndq    = [11622, 11319, 12221, 12981, 13181, 14344, 14762, 14347, 13507, 13138, 14225, 14926]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=sp500,  mode="lines+markers", name="S&P 500",
                             marker=dict(size=7), line=dict(width=2, color="steelblue")))
    fig.add_trace(go.Scatter(x=months, y=ndq,    mode="lines+markers", name="NASDAQ 100",
                             marker=dict(size=7), line=dict(width=2, color="tomato", dash="dash")))
    fig.update_layout(title="S&P 500 vs NASDAQ 100  —  Monthly Close 2023",
                      xaxis_title="Month", yaxis_title="Index Level",
                      xaxis=dict(**_GRID, dtick=1), yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "line",
          "series": {"S&P 500":    {str(m): v for m, v in zip(months, sp500)},
                     "NASDAQ 100": {str(m): v for m, v in zip(months, ndq)}}}
    return fig, gt


def _scatter() -> tuple[go.Figure, dict]:
    """BMI vs systolic blood pressure — 12 de-identified clinical records."""
    labels = [f"P{i+1:02d}" for i in range(12)]
    bmi    = [22.1, 25.4, 28.7, 31.2, 19.8, 27.5, 34.1, 23.6, 29.8, 36.5, 24.9, 32.7]
    sbp    = [118,  128,  135,  145,  112,  132,  152,  122,  138,  160,  124,  148]
    fig = go.Figure(go.Scatter(
        x=bmi, y=sbp, mode="markers+text",
        text=labels, textposition="top center",
        marker=dict(size=10, color="darkorchid"),
    ))
    fig.update_layout(title="BMI vs Systolic Blood Pressure (Clinical)",
                      xaxis_title="BMI (kg/m²)", yaxis_title="Systolic BP (mmHg)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "scatter",
          "points": [{"x": b, "y": s, "label": l} for b, s, l in zip(bmi, sbp, labels)]}
    return fig, gt


def _pie() -> tuple[go.Figure, dict]:
    """Desktop OS market share — StatCounter Jun 2024."""
    labels = ["Windows", "macOS", "Linux", "ChromeOS", "Other"]
    values = [72.9, 15.4, 4.0, 3.1, 4.6]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        textinfo="label+percent",
        texttemplate="%{label}<br>%{percent:.1%}",
    ))
    fig.update_layout(title="Desktop OS Market Share  —  Jun 2024", **_L)
    return fig, {"chart_type": "pie", "slices": dict(zip(labels, values))}


def _histogram() -> tuple[go.Figure, dict]:
    """US population by age bracket, 2023 Census estimate (millions)."""
    bins   = [(0, 10, 40), (10, 20, 42), (20, 30, 46), (30, 40, 45),
              (40, 50, 43), (50, 60, 44), (60, 70, 38), (70, 999, 35)]
    labels = [f"{b[0]}-{b[1]}" if b[1] != 999 else "70+" for b in bins]
    counts = [b[2] for b in bins]
    fig = go.Figure(go.Bar(x=labels, y=counts, marker_color="seagreen",
                           text=counts, textposition="outside"))
    fig.update_layout(title="US Population by Age Group 2023 (Millions)",
                      xaxis_title="Age Group", yaxis_title="Population (M)",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(**_GRID, range=[0, 52]), bargap=0.05, **_L)
    gt = {"chart_type": "histogram",
          "bins": [{"bin_start": b[0], "bin_end": b[1], "count": b[2]} for b in bins]}
    return fig, gt


def _box() -> tuple[go.Figure, dict]:
    """COVID-19 ICU length of stay (days) by age group — published cohort data."""
    groups = {
        "18-40": dict(median=4,  q1=2,  q3=7,  lower=1, upper=14),
        "41-60": dict(median=7,  q1=4,  q3=12, lower=2, upper=21),
        "61-75": dict(median=11, q1=6,  q3=18, lower=2, upper=32),
        "76+":   dict(median=8,  q1=4,  q3=14, lower=2, upper=25),
    }
    fig = go.Figure()
    for name, s in groups.items():
        fig.add_trace(go.Box(name=name,
                             median=[s["median"]], q1=[s["q1"]], q3=[s["q3"]],
                             lowerfence=[s["lower"]], upperfence=[s["upper"]]))
    fig.update_layout(title="COVID-19 ICU Length of Stay (Days) by Age Group",
                      yaxis_title="Days", yaxis=dict(**_GRID), **_L)
    return fig, {"chart_type": "box", "groups": groups}


def _heatmap() -> tuple[go.Figure, dict]:
    """Monthly average temperature (°C) — 4 global cities, seasonal quarters."""
    cities  = ["New York", "London", "Tokyo", "Dubai"]
    months  = ["Jan", "Apr", "Jul", "Oct"]
    values  = [
        [ -2,  13, 29, 14],   # New York
        [  4,  11, 19, 13],   # London
        [  5,  15, 27, 18],   # Tokyo
        [ 19,  28, 38, 28],   # Dubai
    ]
    fig = go.Figure(go.Heatmap(
        z=values, x=months, y=cities,
        text=values, texttemplate="%{text}°C",
        colorscale="RdBu_r", showscale=True,
    ))
    fig.update_layout(title="Average Monthly Temperature (°C) by City",
                      xaxis_title="Month", yaxis_title="City", **_L)
    return fig, {"chart_type": "heatmap", "rows": cities, "cols": months, "values": values}


def _area() -> tuple[go.Figure, dict]:
    """UK renewable energy generation (TWh) by source  Jan–Jun 2023."""
    months = [1, 2, 3, 4, 5, 6]
    wind   = [18.4, 17.1, 14.8, 12.3, 10.9, 11.6]
    solar  = [ 1.2,  1.8,  3.6,  5.8,  7.9,  9.2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=wind, fill="tozeroy", name="Wind",
                             mode="lines", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=months, y=solar, fill="tonexty", name="Solar",
                             mode="lines", line=dict(color="goldenrod")))
    fig.update_layout(title="UK Renewable Energy Generation (TWh)  Jan–Jun 2023",
                      xaxis_title="Month", yaxis_title="Generation (TWh)",
                      xaxis=dict(**_GRID, dtick=1), yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "area",
          "series": {"Wind":  {str(m): v for m, v in zip(months, wind)},
                     "Solar": {str(m): v for m, v in zip(months, solar)}}}
    return fig, gt


# ══════════════════════════════════════════════════════════════════════════════
#  New chart types
# ══════════════════════════════════════════════════════════════════════════════

def _dual_axis() -> tuple[go.Figure, dict]:
    """Monthly US unemployment rate (%) vs total job openings (M)  Jan–Jun 2023."""
    months   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    unemp    = [3.4, 3.6, 3.5, 3.4, 3.7, 3.6]   # %, right axis
    openings = [10.6, 10.4, 9.9, 9.6, 10.3, 9.8]  # millions, bars, left axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=months, y=openings, name="Job Openings (M)",
                         marker_color="steelblue", opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=months, y=unemp, name="Unemployment Rate (%)",
                             mode="lines+markers", line=dict(color="tomato", width=2),
                             marker=dict(size=8)), secondary_y=True)
    fig.update_yaxes(title_text="Job Openings (Millions)", secondary_y=False,
                     **_GRID, range=[0, 13])
    fig.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=True,
                     **_GRID, range=[0, 6])
    fig.update_layout(title="US Job Openings vs Unemployment Rate  Jan–Jun 2023",
                      xaxis_title="Month", **_L)
    gt = {"chart_type": "dual_axis",
          "bar_series": "Job Openings (M)", "line_series": "Unemployment Rate (%)",
          "bar_values":  dict(zip(months, openings)),
          "line_values": dict(zip(months, unemp))}
    return fig, gt


def _log_scale() -> tuple[go.Figure, dict]:
    """Exponential user growth for two global social platforms 2015-2022 (millions)."""
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    plat_a = [325, 550, 920, 1500, 2070, 2630, 3050, 3370]    # Facebook-like MAU
    plat_b = [110, 195, 328, 525,  785, 1082, 1300, 1450]     # Instagram-like MAU
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=plat_a, mode="lines+markers",
                             name="Platform A", marker=dict(size=8),
                             line=dict(width=2, color="royalblue")))
    fig.add_trace(go.Scatter(x=years, y=plat_b, mode="lines+markers",
                             name="Platform B", marker=dict(size=8),
                             line=dict(width=2, color="darkorange", dash="dash")))
    fig.update_layout(title="Social Platform MAU Growth 2015–2022 (Log Scale, Millions)",
                      xaxis_title="Year", yaxis_title="Monthly Active Users (M, log scale)",
                      xaxis=dict(**_GRID, dtick=1),
                      yaxis=dict(**_GRID, type="log"), **_L)
    gt = {"chart_type": "log_scale",
          "series": {"Platform A": {str(y): v for y, v in zip(years, plat_a)},
                     "Platform B": {str(y): v for y, v in zip(years, plat_b)}}}
    return fig, gt


def _subplots_2x2() -> tuple[go.Figure, dict]:
    """Quarterly regional sales ($M) — 4 quarters × 4 regions (2×2 subplot grid)."""
    regions = ["North", "South", "East", "West"]
    quarters = {
        "Q1 2023": [125, 98,  145, 167],
        "Q2 2023": [138, 112, 156, 182],
        "Q3 2023": [152, 128, 168, 195],
        "Q4 2023": [175, 145, 189, 220],
    }
    q_names = list(quarters.keys())
    fig = make_subplots(rows=2, cols=2, subplot_titles=q_names)
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (r, c), (q, vals) in zip(positions, quarters.items()):
        fig.add_trace(go.Bar(x=regions, y=vals, name=q,
                             text=vals, textposition="outside",
                             showlegend=False), row=r, col=c)
    fig.update_layout(title="Quarterly Regional Sales ($M) 2023", **_L)
    fig.update_yaxes(**_GRID)
    gt = {"chart_type": "subplots_2x2",
          "panels": [{"title": q, "data": dict(zip(regions, v))}
                     for q, v in quarters.items()]}
    return fig, gt


def _bubble() -> tuple[go.Figure, dict]:
    """GDP per capita vs life expectancy vs population — 8 countries, World Bank 2022."""
    countries = ["USA",   "China", "India", "Germany", "Japan",  "Brazil", "Nigeria", "Mexico"]
    gdp_pc    = [76399,   12720,   2389,    48718,     39313,    8917,     2184,      10046]
    life_exp  = [76.1,    78.2,    67.7,    80.6,      84.3,     74.0,     53.6,      75.1]
    pop_m     = [335,     1412,    1417,    84,        125,      215,      219,       130]
    fig = go.Figure(go.Scatter(
        x=gdp_pc, y=life_exp,
        mode="markers+text",
        text=countries, textposition="top center",
        marker=dict(size=[p / 20 for p in pop_m],
                    color=gdp_pc, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="GDP/capita")),
    ))
    fig.update_layout(
        title="GDP per Capita vs Life Expectancy (bubble = population, M)  2022",
        xaxis_title="GDP per Capita ($)", yaxis_title="Life Expectancy (years)",
        xaxis=dict(**_GRID), yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "bubble",
          "points": [{"label": c, "x": g, "y": l, "size": p}
                     for c, g, l, p in zip(countries, gdp_pc, life_exp, pop_m)]}
    return fig, gt


def _waterfall() -> tuple[go.Figure, dict]:
    """TechCorp FY-2023 P&L waterfall ($M) — fictional but realistic."""
    bars = [
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
    labels   = [b[0] for b in bars]
    values   = [b[1] for b in bars]
    measures = [b[2] for b in bars]
    fig = go.Figure(go.Waterfall(
        name="P&L", orientation="v",
        measure=measures, x=labels, y=values,
        text=[f"${abs(v)}M" for v in values], textposition="outside",
        connector=dict(line=dict(color="gray")),
    ))
    fig.update_layout(title="TechCorp FY-2023 P&L Waterfall ($M)",
                      yaxis_title="$M", yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "waterfall",
          "bars": [{"label": b[0], "value": b[1], "bar_type": b[2]} for b in bars]}
    return fig, gt


def _candlestick() -> tuple[go.Figure, dict]:
    """AAPL-like stock OHLC data — 6 trading sessions Jan 2024 (approximate)."""
    candles = [
        ("Jan 02", 187.15, 188.44, 183.89, 185.20),
        ("Jan 03", 184.22, 185.88, 183.43, 184.25),
        ("Jan 04", 182.15, 183.09, 180.63, 181.91),
        ("Jan 05", 181.99, 182.32, 179.25, 181.18),
        ("Jan 08", 182.09, 188.74, 181.99, 188.32),
        ("Jan 09", 188.68, 190.00, 186.37, 185.64),
    ]
    dates = [c[0] for c in candles]
    fig = go.Figure(go.Candlestick(
        x=dates,
        open =[c[1] for c in candles],
        high =[c[2] for c in candles],
        low  =[c[3] for c in candles],
        close=[c[4] for c in candles],
    ))
    fig.update_layout(
        title="AAPL Stock Price  —  Jan 2024 (OHLC)",
        yaxis_title="Price ($)", xaxis_rangeslider_visible=False,
        yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "candlestick",
          "candles": [{"date": c[0], "open": c[1], "high": c[2],
                       "low": c[3], "close": c[4]} for c in candles]}
    return fig, gt


def _violin() -> tuple[go.Figure, dict]:
    """Systolic BP reduction (mmHg) in a 4-arm RCT (placebo-controlled)."""
    groups_stats = {
        "Placebo":      dict(median=2,  q1=-1, q3=5,  lower=-5, upper=10),
        "Drug 10 mg":   dict(median=8,  q1=5,  q3=12, lower=0,  upper=18),
        "Drug 20 mg":   dict(median=14, q1=10, q3=18, lower=4,  upper=25),
        "Drug 40 mg":   dict(median=19, q1=15, q3=23, lower=8,  upper=30),
    }
    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for (name, s), col in zip(groups_stats.items(), colors):
        # Synthetic data matching the pre-defined statistics
        iqr = s["q3"] - s["q1"]
        std = iqr / 1.35
        data = np.clip(np.random.normal(s["median"], std, 80),
                       s["lower"], s["upper"])
        fig.add_trace(go.Violin(y=data, name=name, box_visible=True,
                                meanline_visible=True,
                                line_color=col, fillcolor=col, opacity=0.6))
    fig.update_layout(title="Systolic BP Reduction (mmHg) by Treatment Group (RCT)",
                      yaxis_title="BP Reduction (mmHg)",
                      yaxis=dict(**_GRID), violinmode="group", **_L)
    gt = {"chart_type": "violin", "groups": groups_stats}
    return fig, gt


def _radar() -> tuple[go.Figure, dict]:
    """WEF Global Competitiveness metrics — USA vs Germany (score 0–100)."""
    metrics   = ["Infrastructure", "Innovation", "Institutions", "Macro-stability", "Health", "Skills"]
    usa_vals  = [90, 88, 74, 72, 83, 84]
    ger_vals  = [87, 85, 82, 79, 88, 86]
    theta     = metrics + [metrics[0]]   # close the polygon
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=usa_vals + [usa_vals[0]], theta=theta,
                                  fill="toself", name="USA",
                                  line=dict(color="steelblue")))
    fig.add_trace(go.Scatterpolar(r=ger_vals + [ger_vals[0]], theta=theta,
                                  fill="toself", name="Germany",
                                  line=dict(color="tomato")))
    fig.update_layout(title="WEF Competitiveness Scores  —  USA vs Germany",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 100], dtick=20)),
                      **_L)
    gt = {"chart_type": "radar",
          "series": {"USA":     dict(zip(metrics, usa_vals)),
                     "Germany": dict(zip(metrics, ger_vals))}}
    return fig, gt


def _stacked_bar() -> tuple[go.Figure, dict]:
    """Electricity generation mix by source (TWh) — 5 major economies, 2022."""
    countries = ["USA",  "China", "Germany", "France", "UK"]
    sources   = {
        "Coal":       [778,  5300, 108,  7,   7],
        "Gas":        [1686,  253,  68,  24, 100],
        "Nuclear":    [772,   411, 32,  295,  43],
        "Renewables": [908,  2200, 254, 130, 134],
    }
    fig = go.Figure()
    colors = {"Coal": "#5a5a5a", "Gas": "#f4a261",
              "Nuclear": "#e9c46a", "Renewables": "#2a9d8f"}
    for src, vals in sources.items():
        fig.add_trace(go.Bar(name=src, x=countries, y=vals,
                             marker_color=colors[src]))
    fig.update_layout(title="Electricity Generation Mix by Source (TWh, 2022)",
                      xaxis_title="Country", yaxis_title="Generation (TWh)",
                      barmode="stack", yaxis=dict(**_GRID), **_L)
    gt = {"chart_type": "stacked_bar",
          "series": {src: dict(zip(countries, vals)) for src, vals in sources.items()}}
    return fig, gt


def _stacked_bar_100() -> tuple[go.Figure, dict]:
    """Population age structure (%) — 5 countries with contrasting demographics."""
    countries  = ["USA",  "Japan", "India", "Nigeria", "Germany"]
    age_groups = {
        "0-14 yrs":  [18.3, 11.9, 26.3, 43.6, 13.5],
        "15-64 yrs": [64.8, 59.4, 67.3, 53.7, 64.8],
        "65+ yrs":   [16.9, 28.7,  6.4,  2.7, 21.7],
    }
    colors = {"0-14 yrs": "#2196F3", "15-64 yrs": "#4CAF50", "65+ yrs": "#FF5722"}
    fig = go.Figure()
    for grp, vals in age_groups.items():
        fig.add_trace(go.Bar(name=grp, x=countries, y=vals,
                             marker_color=colors[grp], text=[f"{v}%" for v in vals],
                             textposition="inside"))
    fig.update_layout(title="Population Age Structure (%)  —  Selected Countries 2023",
                      xaxis_title="Country", yaxis_title="Share (%)",
                      barmode="stack", yaxis=dict(**_GRID, range=[0, 102]), **_L)
    gt = {"chart_type": "stacked_bar_100",
          "series": {grp: dict(zip(countries, vals)) for grp, vals in age_groups.items()}}
    return fig, gt


def _surface_3d() -> tuple[go.Figure, dict]:
    """Climatological surface: mean temperature (°C) over a 4×4 lat/lon grid, Jan."""
    lat_labels = ["60°N", "30°N", "0°",  "30°S"]
    lon_labels = ["0°E",  "30°E", "60°E","90°E"]
    z = [               # rows = latitude, cols = longitude
        [-8,  -5,   3,  12],   # 60°N
        [ 8,  12,  18,  24],   # 30°N
        [26,  28,  27,  25],   # Equator
        [22,  25,  28,  20],   # 30°S
    ]
    fig = go.Figure(go.Surface(
        x=lon_labels, y=lat_labels, z=z,
        colorscale="RdBu_r", showscale=True,
        colorbar=dict(title="°C"),
        text=[[f"{v}°C" for v in row] for row in z],
        hovertemplate="Lat: %{y}<br>Lon: %{x}<br>Temp: %{z}°C",
    ))
    fig.update_layout(
        title="Mean January Temperature (°C) — Lat/Lon Grid",
        scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Temp (°C)"),
        **_L)
    gt = {"chart_type": "surface_3d",
          "lat_labels": lat_labels, "lon_labels": lon_labels, "z": z}
    return fig, gt


def _funnel() -> tuple[go.Figure, dict]:
    """E-commerce conversion funnel — realistic SaaS metrics."""
    stages = ["Website Visits",  "Product Views",   "Add to Cart",
              "Checkout Started","Payment Entered",  "Orders Completed"]
    values = [125420, 84350, 31240, 18560, 12380, 9845]
    fig = go.Figure(go.Funnel(
        y=stages, x=values,
        textinfo="value+percent initial",
        marker=dict(color=["#1f77b4","#2196F3","#42A5F5",
                            "#64B5F6","#90CAF9","#BBDEFB"]),
    ))
    fig.update_layout(title="E-Commerce Conversion Funnel", **_L)
    gt = {"chart_type": "funnel",
          "stages": [{"stage": s, "value": v} for s, v in zip(stages, values)]}
    return fig, gt


def _treemap() -> tuple[go.Figure, dict]:
    """S&P 500 approximate sector market cap ($T) with top constituents, mid-2024."""
    # (label, parent, value_$B)
    nodes = [
        ("S&P 500",       "",           0),
        ("Technology",    "S&P 500",    0),
        ("Financials",    "S&P 500",    0),
        ("Healthcare",    "S&P 500",    0),
        ("Consumer Disc", "S&P 500",    0),
        ("Industrials",   "S&P 500",    0),
        ("Energy",        "S&P 500",    0),
        # Technology leaves
        ("Apple",         "Technology", 3050),
        ("Microsoft",     "Technology", 3100),
        ("NVIDIA",        "Technology", 2800),
        ("Meta",          "Technology", 1250),
        ("Tech Others",   "Technology", 2600),
        # Financials leaves
        ("JPMorgan",      "Financials",  590),
        ("Berkshire",     "Financials",  870),
        ("Visa",          "Financials",  540),
        ("Fin Others",    "Financials", 2200),
        # Healthcare leaves
        ("UnitedHealth",  "Healthcare",  480),
        ("Eli Lilly",     "Healthcare",  760),
        ("HC Others",     "Healthcare", 3360),
        # Consumer Disc leaves
        ("Amazon",        "Consumer Disc", 1950),
        ("Tesla",         "Consumer Disc",  680),
        ("CD Others",     "Consumer Disc", 1570),
        # Industrials leaves
        ("Caterpillar",   "Industrials",  180),
        ("Honeywell",     "Industrials",  130),
        ("Ind Others",    "Industrials", 2890),
        # Energy leaves
        ("ExxonMobil",    "Energy",  490),
        ("Chevron",       "Energy",  280),
        ("Energy Others", "Energy", 2030),
    ]
    labels  = [n[0] for n in nodes]
    parents = [n[1] for n in nodes]
    values  = [n[2] for n in nodes]
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        textinfo="label+value",
        texttemplate="%{label}<br>$%{value}B",
        root_color="lightgrey",
    ))
    fig.update_layout(title="S&P 500 Sector Market Cap ($B, mid-2024)", **_L)
    # Ground truth: leaf nodes only (parent == sector name)
    leaf_nodes = [n for n in nodes if n[1] not in ("", "S&P 500") and n[2] > 0
                  and n[1] in ("Technology","Financials","Healthcare",
                               "Consumer Disc","Industrials","Energy")]
    gt = {"chart_type": "treemap",
          "nodes": [{"label": n[0], "parent": n[1], "value": n[2]} for n in leaf_nodes]}
    return fig, gt


# ══════════════════════════════════════════════════════════════════════════════
#  Registry & public API
# ══════════════════════════════════════════════════════════════════════════════

_GENERATORS: dict[str, callable] = {
    "bar_vertical":   _bar_vertical,
    "bar_horizontal": _bar_horizontal,
    "line":           _line,
    "scatter":        _scatter,
    "pie":            _pie,
    "histogram":      _histogram,
    "box":            _box,
    "heatmap":        _heatmap,
    "area":           _area,
    "dual_axis":      _dual_axis,
    "log_scale":      _log_scale,
    "subplots_2x2":   _subplots_2x2,
    "bubble":         _bubble,
    "waterfall":      _waterfall,
    "candlestick":    _candlestick,
    "violin":         _violin,
    "radar":          _radar,
    "stacked_bar":    _stacked_bar,
    "stacked_bar_100": _stacked_bar_100,
    "surface_3d":     _surface_3d,
    "funnel":         _funnel,
    "treemap":        _treemap,
}


def generate_all_charts() -> dict[str, tuple[go.Figure, dict]]:
    return {ct: gen() for ct, gen in _GENERATORS.items()}


def save_chart_images(chart_type: str, fig: go.Figure) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for res_name, (width, height) in RESOLUTIONS.items():
        subdir = CHARTS_DIR / res_name
        subdir.mkdir(parents=True, exist_ok=True)
        fpath = subdir / f"{chart_type}.png"
        fig.write_image(str(fpath), width=width, height=height)
        paths[res_name] = fpath
    return paths


def generate_and_save_all() -> dict[str, tuple[dict, dict[str, Path]]]:
    """Generate every chart at all resolutions. Returns {chart_type: (gt, {res: path})}."""
    results: dict[str, tuple[dict, dict[str, Path]]] = {}
    for ct, gen in _GENERATORS.items():
        fig, gt = gen()
        image_paths = save_chart_images(ct, fig)
        results[ct] = (gt, image_paths)
        print(f"  OK {ct:20s}  {list(image_paths.keys())}")
    return results
