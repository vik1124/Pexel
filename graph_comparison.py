"""Main flow class"""

from chart_generator import CHART_MAPPINGS, chart_factory, generate_and_save_chart
from pathlib import Path
from prefect import flow, task, unmapped
from prefect.artifacts import create_table_artifact
import config as cfg

from pydantic import BaseModel
from typing import Dict

import pandas as pd

import os

from dotenv import load_dotenv

from llm_extractor import extract_chart_data
from evaluator import evaluate_detail
from mlflow_tracker import log_experiment_run, log_llm_call, setup_mlflow

def _extract_cache_key(_context, parameters: dict) -> str:
    model = parameters["model"].replace(":", "_")
    return f"extract/{parameters['chart_type']}/{model}/{parameters['resolution']}"

def _evaluate_cache_key(_context, parameters: dict) -> str:
    model = parameters["model"].replace(":", "_")
    return f"evaluate/{parameters["extract_data"]['chart_type']}/{model}/{parameters['resolution']}"


@task(
    retries=1,
    name="Extract_Evaluate",
    retry_delay_seconds=5,
    log_prints=True,
    cache_key_fn=_extract_cache_key,
)
async def extract_and_evaluate(
    chart_type: str,
    model:      str = "openai:gpt-5.5",
    resolution: str = "high",
) -> dict:
    """
    Single experiment: call the vision LLM, evaluate the extraction, log to MLFlow.
    Retried up to 2 times on failure (handles transient API errors / rate limits).
    """
    chart             = chart_factory(chart_type).generate_plot()
    image_path        = cfg.CHARTS_DIR / resolution / f"{chart_type}.png"
    width, height     = cfg.RESOLUTIONS[resolution]

    print(f"-> {model:20s} | {chart_type:16s} | {resolution:6s} ...", end="", flush=True)
    extracted, llm_meta = await extract_chart_data(
        image_path=Path(image_path),
        chart_type=chart_type,
        extraction_class=chart.extraction_class,
       # model_name=model,
    )


    if llm_meta["success"] and extracted is not None:
        metrics, comparison_rows = evaluate_detail(chart_type, chart.ground_truth, extracted)
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

    artifact_key = f"{chart_type}-{resolution}-{model}".replace("_", "-")
    await create_table_artifact(
        key=artifact_key,
        table=comparison_rows,
        description=(
            f"{chart_type} | {resolution} | {model} — "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
            f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})"
        ),
    )
    pd.DataFrame(comparison_rows).to_csv(cfg.CHARTS_DIR / resolution / f"{chart_type}.csv", index=False)
    

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


@task(
    retries=1,
    name="extract",
    retry_delay_seconds=5,
    log_prints=True,
    cache_key_fn=_extract_cache_key,
)
async def extract(
    chart_type: str,
    model:      str = "openai:gpt-5.5",
    resolution: str = "high",
) -> dict:
    """
    Single experiment: call the vision LLM, evaluate the extraction, log to MLFlow.
    Retried up to 2 times on failure (handles transient API errors / rate limits).
    """
    chart             = chart_factory(chart_type).generate_plot()
    image_path        = cfg.CHARTS_DIR / resolution / f"{chart_type}.png"
    width, height     = cfg.RESOLUTIONS[resolution]

    print(f"-> {model:20s} | {chart_type:16s} | {resolution:6s} ...", end="", flush=True)
    extracted, llm_meta = await extract_chart_data(
        image_path=Path(image_path),
        chart_type=chart_type,
        extraction_class=chart.extraction_class,
       # model_name=model,
    )

    return {"extracted": extracted, "llm_meta": llm_meta, "chart_type": chart_type}


@task(
    retries=1,
    name="evaluate",
    retry_delay_seconds=5,
    log_prints=True,
    cache_key_fn=_evaluate_cache_key,
)
async def evaluate(
    extract_data: Dict,
    model:      str = "openai:gpt-5.5",
    resolution: str = "high",
) -> dict:

    chart_type = extract_data["chart_type"]
    extracted = extract_data["extracted"]
    llm_meta = extract_data["llm_meta"]
    chart = chart_factory(chart_type).generate_plot()

    if llm_meta["success"] and extracted is not None:
        metrics, comparison_rows = evaluate_detail(chart_type, chart.ground_truth, extracted)
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

    artifact_key = f"{chart_type}-{resolution}-{model}".replace("_", "-").replace(":","-").replace(".","-")
    await create_table_artifact(
        key=artifact_key,
        table=comparison_rows,
        description=(
            f"{chart_type} | {resolution} | {model} — "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
            f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})"
        ),
    )
    pd.DataFrame(comparison_rows).to_csv(cfg.CHARTS_DIR / resolution / f"{chart_type}.csv", index=False)
    

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
    results = []
    futures = generate_and_save_chart.map(CHART_MAPPINGS.keys())
    #print(futures.result())
    # for model_name in ["openai:gpt-5.5", "openai:gpt-5.6-luna", "openai:gpt-5.6-terra", "openai:gpt-5.6-sol"]:
    #     print(f"Running {model_name}")
    #     result = extract_and_evaluate.map(chart_type=futures.result(), model=model_name)
    #     results.extend(result.result())
    #     print(results)
    for model_name in ["openai:gpt-5.5", "openai:gpt-5.6-luna", "openai:gpt-5.6-terra", "openai:gpt-5.6-sol"]:
            print(f"Running {model_name}")
            res_extract = extract.map(chart_type=futures.result(), model=model_name)
            result = evaluate.map(extract_data=res_extract.result(), model=model_name)
            results.extend(result.result())
            print(results)
    create_table_artifact(
        key="result",
        table=results,
        description="Evaluation Results"
    )




if __name__ == "__main__":
    load_dotenv()
    evaluate_extraction()