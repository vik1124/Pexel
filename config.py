from pathlib import Path

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
]

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "low":    (400,  300),
    "medium": (800,  600),
    "high":   (1200, 900),
}

CHART_TYPES = [
    # Single-axis basics
    "bar_vertical",
    "bar_horizontal",
    "line",
    "scatter",
    "pie",
    "histogram",
    "box",
    "heatmap",
    "area",
    # Extended types
    "dual_axis",
    "log_scale",
    "subplots_2x2",
    "bubble",
    "waterfall",
    "candlestick",
    "violin",
    "radar",
    "stacked_bar",
    "stacked_bar_100",
    "surface_3d",
    "funnel",
    "treemap",
]

# Relative tolerance for numeric value comparison (10%)
NUMERIC_TOLERANCE = 0.10

# Minimum fuzzy-string similarity to consider two labels the same
LABEL_SIMILARITY_THRESHOLD = 0.70

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "graph_data_extraction"
LLM_CALLS_EXPERIMENT = "llm_calls"

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Max concurrent LLM requests (avoid rate-limit errors)
MAX_CONCURRENT = 3
