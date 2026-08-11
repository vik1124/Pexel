"""
Prefect-based orchestrator for the graph extraction benchmark.

Usage
-----
    python run_experiments.py                          # full 264-experiment run
    python run_experiments.py                          # idempotent — skips existing images

Environment overrides
---------------------
    OPENAI_API_KEY   required
    MODELS           comma-separated subset, e.g. "gpt-4o-mini,gpt-4o"
    CHART_TYPES      comma-separated subset, e.g. "bar_vertical,pie"
    RESOLUTIONS      comma-separated subset, e.g. "low,high"

Prefect UI
----------
    prefect server start          # http://localhost:4200
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

import config as cfg
from chart import Chart
from chart_generator import generate_all_charts, generate_and_save_all
from evaluator import evaluate
from llm_extractor import extract_chart_data
from mlflow_tracker import log_experiment_run, log_llm_call, setup_mlflow

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: OPENAI_API_KEY not set in environment or .env file.")

_MODELS      = [m.strip() for m in os.environ.get("MODELS",      ",".join(cfg.OPENAI_MODELS)).split(",") if m.strip()]
_CHART_TYPES = [c.strip() for c in os.environ.get("CHART_TYPES", ",".join(cfg.CHART_TYPES)).split(",")  if c.strip()]
_RESOLUTIONS = [r.strip() for r in os.environ.get("RESOLUTIONS", ",".join(cfg.RESOLUTIONS)).split(",")  if r.strip()]

_sem: asyncio.Semaphore | None = None


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="generate-charts", log_prints=True)
def generate_charts_task(
    chart_types: list[str],
    resolutions: list[str],
) -> dict[str, Chart]:
    """
    Ensure PNG images exist for every (chart_type, resolution) pair.
    Returns {chart_type: Chart} with fig and ground_truth populated.
    """
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    need_generate = any(
        not (cfg.CHARTS_DIR / res / f"{ct}.png").exists()
        for ct in chart_types
        for res in resolutions
    )

    if need_generate:
        print("Generating chart images ...")
        all_charts = generate_and_save_all()
    else:
        print("All chart images already exist — loading ground truth only.")
        all_charts = generate_all_charts()

    return {ct: all_charts[ct] for ct in chart_types if ct in all_charts}


@task(
    name="extract-and-evaluate",
    retries=2,
    retry_delay_seconds=30,
    log_prints=True,
)
async def run_one_task(
    *,
    model:      str,
    chart:      Chart,
    resolution: str,
) -> dict:
    """
    Single experiment: call the vision LLM, evaluate the extraction, log to MLFlow.
    Retried up to 2 times on failure (handles transient API errors / rate limits).
    """
    image_path        = cfg.CHARTS_DIR / resolution / f"{chart.chart_type}.png"
    width, height     = cfg.RESOLUTIONS[resolution]

    async with _sem:
        print(f"-> {model:20s} | {chart.chart_type:16s} | {resolution:6s} ...", end="", flush=True)
        extracted, llm_meta = await extract_chart_data(
            image_path=image_path,
            chart_type=chart.chart_type,
            model_name=model,
            api_key=API_KEY,
        )

    if llm_meta["success"] and extracted is not None:
        metrics = evaluate(chart.chart_type, chart.ground_truth, extracted)
        print(
            f" P={metrics['precision']:.2f}  R={metrics['recall']:.2f}"
            f"  F1={metrics['f1']:.2f}  tokens={llm_meta['total_tokens']}"
        )
    else:
        metrics = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "total_gt": 0,
        }
        print(f" FAILED: {llm_meta.get('error', '?')[:80]}")

    log_llm_call(model=model, chart_type=chart.chart_type, resolution=resolution, llm_meta=llm_meta)
    run_id = log_experiment_run(
        model=model, chart_type=chart.chart_type, resolution=resolution,
        width=width, height=height,
        metrics=metrics, llm_meta=llm_meta,
        ground_truth=chart.ground_truth, extracted=extracted,
        image_path=image_path,
    )

    return {
        "model":         model,
        "chart_type":    chart.chart_type,
        "resolution":    resolution,
        "mlflow_run_id": run_id,
        **metrics,
        "latency_s":    llm_meta.get("latency_seconds", 0),
        "total_tokens": llm_meta.get("total_tokens", 0),
        "success":      llm_meta["success"],
    }


@task(name="save-results", log_prints=True)
def save_results_task(rows: list[dict]) -> Path:
    """Write results_summary.csv and print model × resolution and chart-type pivot tables."""
    results_dir = cfg.OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df       = pd.DataFrame(rows)
    csv_path = results_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    pivot = (
        df.groupby(["model", "resolution"])[["precision", "recall", "f1", "accuracy"]]
        .mean()
        .round(3)
    )
    print("\nMean metrics by model x resolution:")
    print(pivot.to_string())

    pivot2 = (
        df.groupby(["chart_type"])[["precision", "recall", "f1", "accuracy"]]
        .mean()
        .round(3)
    )
    print("\nMean metrics by chart type:")
    print(pivot2.to_string())

    return csv_path


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(
    name="graph-extraction-benchmark",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True,
)
async def benchmark_flow(
    models:      list[str] | None = None,
    chart_types: list[str] | None = None,
    resolutions: list[str] | None = None,
) -> Path:
    """
    Full benchmark flow. All LLM extractions run concurrently, capped at
    MAX_CONCURRENT simultaneous API calls.

    Parameters override the env-var / config.py defaults when provided:

        from run_experiments import benchmark_flow
        await benchmark_flow(models=["gpt-4o-mini"], chart_types=["pie", "bar_vertical"])
    """
    global _sem

    _models      = models      or _MODELS
    _chart_types = chart_types or _CHART_TYPES
    _resolutions = resolutions or _RESOLUTIONS

    n = len(_models) * len(_chart_types) * len(_resolutions)
    print(f"Models:            {_models}")
    print(f"Chart types ({len(_chart_types):2d}):  {_chart_types}")
    print(f"Resolutions:       {_resolutions}")
    print(f"Total experiments: {n}")

    setup_mlflow()

    charts: dict[str, Chart] = generate_charts_task(_chart_types, _resolutions)

    _sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)

    futures = [
        run_one_task.submit(model=model, chart=chart, resolution=res)
        for ct, chart in charts.items()
        for res in _resolutions
        if (cfg.CHARTS_DIR / res / f"{ct}.png").exists()
        for model in _models
    ]

    rows     = [f.result() for f in futures]
    csv_path = save_results_task(rows)

    print("\nLaunch MLFlow UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("Launch Prefect UI: prefect server start  →  http://localhost:4200")

    return csv_path


if __name__ == "__main__":
    asyncio.run(benchmark_flow())
