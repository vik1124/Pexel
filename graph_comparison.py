"""Main flow class"""

from chart_generator import CHART_MAPPINGS, chart_factory, generate_and_save_chart
from chart import Chart
from pathlib import Path
from prefect import flow, task
from prefect.artifacts import create_table_artifact
import config as cfg

import pandas as pd

import os

from dotenv import load_dotenv

from llm_extractor import extract_chart_data
from evaluator import evaluate_detail
from mlflow_tracker import log_experiment_run, log_llm_call, setup_mlflow

@task(
    retries=1,
    retry_delay_seconds=5,
    log_prints=True,
)
async def extract_and_evaluate(
    chart:      Chart = None,
    model:      str = "openai:gpt-5.5",
    resolution: str = "high",
) -> dict:
    """
    Single experiment: call the vision LLM, evaluate the extraction, log to MLFlow.
    Retried up to 2 times on failure (handles transient API errors / rate limits).
    """
    image_path        = cfg.CHARTS_DIR / resolution / f"{chart.chart_type}.png"
    width, height     = cfg.RESOLUTIONS[resolution]

    print(f"-> {model:20s} | {chart.chart_type:16s} | {resolution:6s} ...", end="", flush=True)
    extracted, llm_meta = await extract_chart_data(
        image_path=Path(image_path),
        chart_type=chart.chart_type,
       # model_name=model,
    )


    if llm_meta["success"] and extracted is not None:
        metrics, comparison_rows = evaluate_detail(chart.chart_type, chart.ground_truth, extracted)
        print(
            f" P={metrics['precision']:.2f}  R={metrics['recall']:.2f}"
            f"  F1={metrics['f1']:.2f}  tokens={llm_meta['total_tokens']}"
        )
    else:
        metrics = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "total_gt": 0,
        }
        comparison_rows = []
        print(f" FAILED: {llm_meta.get('error', '?')[:80]}")

    artifact_key = f"{chart.chart_type}-{resolution}".replace("_", "-")
    create_table_artifact(
        key=artifact_key,
        table=comparison_rows,
        description=(
            f"{chart.chart_type} | {resolution} | {model} — "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
            f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})"
        ),
    )
    pd.DataFrame(comparison_rows).to_csv(cfg.CHARTS_DIR / resolution / f"{chart.chart_type}.csv",index=False)
    

    # log_llm_call(model=model, chart_type=chart.chart_type, resolution=resolution, llm_meta=llm_meta)
    # run_id = log_experiment_run(
    #     model=model, chart_type=chart.chart_type, resolution=resolution,
    #     width=width, height=height,
    #     metrics=metrics, llm_meta=llm_meta,
    #     ground_truth=chart.ground_truth, extracted=extracted,
    #     image_path=image_path,
    # )

    result = {
        "model":         model,
        "chart_type":    chart.chart_type,
        "resolution":    resolution,
        # "mlflow_run_id": run_id,
        **metrics,
        "latency_s":    llm_meta.get("latency_seconds", 0),
        "total_tokens": llm_meta.get("total_tokens", 0),
        "success":      llm_meta["success"],
    }
    print(result)
    return result



@flow(log_prints=True)
def evaluate_extraction():
    print(f"Running mappings")
    futures = generate_and_save_chart.map(CHART_MAPPINGS.keys())
    #print(futures.result())
    result = extract_and_evaluate.map(futures.result())
    print(result.result())
    create_table_artifact(
        key="result",
        table=result.result(),
        description="Evaluation Results"
    )




if __name__ == "__main__":
    load_dotenv()
    evaluate_extraction()