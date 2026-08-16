"""Main flow class"""

from chart_generator import CHART_MAPPINGS, chart_factory, generate_and_save_chart
from pathlib import Path
from prefect import flow, task, unmapped
from prefect.artifacts import create_table_artifact
from prefect.tasks import exponential_backoff
import config as cfg
import json
import itertools

import hashlib

from pydantic import BaseModel
from typing import Dict

import pandas as pd

import os

from dotenv import load_dotenv

from llm_extractor import extract_chart_data
from evaluator import evaluate_detail
from llm_evaluator import evaluate_with_llm
from mlflow_tracker import log_experiment_run, log_llm_call, setup_mlflow

_HERE = Path(__file__).parent

def _extract_cache_key(_context, parameters: dict) -> str:
    model = parameters["model"].replace(":", "_")
    h = hashlib.md5()
    h.update((_HERE / "llm_extractor.py").read_bytes())
    h.update((_HERE / "schemas.py").read_bytes())
    h.update((cfg.CHARTS_DIR / parameters["resolution"] / f"{parameters['chart_type']}.png").read_bytes())
    return f"extract/{parameters['chart_type']}/{model}/{parameters['resolution']}-{h.hexdigest()}"

def _evaluate_cache_key(_context, parameters: dict) -> str:
    model = parameters["extract_data"]["model"].replace(":", "_")
    extracted = parameters['extract_data']
    encoded_str = extracted["extracted"].model_dump_json().encode('utf-8')
    data_hash = hashlib.md5(encoded_str).hexdigest()
    return f"evaluate/{data_hash}/{model}/{parameters['resolution']}"



@task(
    retries=3,
    name="extract",
    retry_delay_seconds=exponential_backoff(backoff_factor=3),
    retry_jitter_factor=1.0,
    log_prints=True,
    cache_key_fn=_extract_cache_key,
)
async def extract(
    chart_type: str,
    model:      str = "openai:gpt-5.6-luna",
    resolution: str = "high",
) -> dict:
    """
    Single experiment: call the vision LLM, evaluate the extraction, log to MLFlow.
    Retried up to 2 times on failure (handles transient API errors / rate limits).
    """
    chart      = chart_factory(chart_type)
    image_path = cfg.CHARTS_DIR / resolution / f"{chart_type}.png"

    print(f"-> {model:20s} | {chart_type:16s} | {resolution:6s} ...", end="", flush=True)
    extracted, llm_meta = await extract_chart_data(
        image_path=Path(image_path),
        chart_type=chart_type,
        extraction_class=chart.extraction_class,
        model_name=model,
    )

    return {
        "extracted": extracted, 
        "llm_meta": llm_meta, 
        "chart_type": chart_type,
        "model": model,
        "resolution": resolution
    }


@task(
    retries=1,
    name="evaluate",
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=1.0,
    log_prints=True,
    cache_key_fn=_evaluate_cache_key,
)
async def evaluate(
    extract_data: Dict,
) -> dict:

    chart_type = extract_data["chart_type"]
    extracted = extract_data["extracted"]
    llm_meta = extract_data["llm_meta"]
    model = extract_data["model"]
    resolution = extract_data["resolution"]
    chart = chart_factory(chart_type)

    evaluator = "heuristic"

    if llm_meta["success"] and extracted is not None:
        metrics, comparison_rows = evaluate_detail(chart_type, chart.ground_truth, extracted)
        print(
            f" P={metrics['precision']:.2f}  R={metrics['recall']:.2f}"
            f"  F1={metrics['f1']:.2f}  tokens={llm_meta['total_tokens']}"
        )
        if metrics['precision']!=1.0:
            metrics, comparison_rows = await evaluate_with_llm(
                        chart_type=chart_type,
                        ground_truth=chart.ground_truth,
                        extracted=extracted,
                        model_name=model,
                    )
            evaluator = "llm"
    else:
        metrics = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "total_gt": 0,
        }
        comparison_rows = []
        print(f" FAILED: {llm_meta.get('error', '?')[:80]}")

    artifact_key = f"comp-{chart_type}-{resolution}-{model}".replace("_", "-").replace(":","-").replace(".","-")
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
        "evaluator":     evaluator,
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
    models = [
        "openai:gpt-5.5", 
        "openai:gpt-5.6-luna", 
        "openai:gpt-5.6-terra", 
        "openai:gpt-5.6-sol", 
        "anthropic:claude-sonnet-5", 
        "anthropic:claude-opus-5"
    ]
    # for model_name in models:
    #         print(f"Running {model_name}")
    #         res_extract = extract.map(chart_type=futures.result(), model=model_name)
    #         extracted_results = res_extract.result()
    #         heuristic = evaluate.map(extract_data=extracted_results, model=model_name)
    #         results.extend(heuristic.result())
    #         print(results)

    inputs = list(itertools.product(futures.result(), models))
    extracts = extract.map(chart_type=[x[0] for x in inputs], model=[x[1] for x in inputs])
    evaluation = evaluate.map(extract_data=extracts.result())
    results.extend(evaluation.result())
    print(results)
    
    create_table_artifact(
        key="result",
        table=results,
        description="Evaluation Results"
    )

    pd.DataFrame(results).to_csv(
        cfg.CHARTS_DIR / f"high/results.csv", index=False
    )




if __name__ == "__main__":
    load_dotenv()
    evaluate_extraction()