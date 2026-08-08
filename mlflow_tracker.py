"""MLFlow helpers — experiment tracking and per-call LLM telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import RunStatus

from config import EXPERIMENT_NAME, LLM_CALLS_EXPERIMENT, MLFLOW_TRACKING_URI


# ── Initialisation ────────────────────────────────────────────────────────────

def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for exp_name in (EXPERIMENT_NAME, LLM_CALLS_EXPERIMENT):
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name)


# ── Experiment run ─────────────────────────────────────────────────────────────

def log_experiment_run(
    *,
    model:       str,
    chart_type:  str,
    resolution:  str,
    width:       int,
    height:      int,
    metrics:     dict,      # precision, recall, f1, accuracy, tp, fp, fn, total_gt
    llm_meta:    dict,      # latency, tokens, success
    ground_truth: dict,
    extracted:   Any,
    image_path:  Path,
) -> str:
    """Log one (model × chart_type × resolution) run. Returns the MLFlow run_id."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"{model}__{chart_type}__{resolution}"
    with mlflow.start_run(run_name=run_name) as run:
        # Parameters
        mlflow.log_params({
            "model":      model,
            "chart_type": chart_type,
            "resolution": resolution,
            "width":      width,
            "height":     height,
            "tolerance":  0.10,
            "total_gt_slots": metrics.get("total_gt", 0),
        })

        # Evaluation metrics
        mlflow.log_metrics({
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1":        metrics["f1"],
            "accuracy":  metrics["accuracy"],
            "tp":        float(metrics["tp"]),
            "fp":        float(metrics["fp"]),
            "fn":        float(metrics["fn"]),
        })

        # LLM call telemetry
        if llm_meta.get("success"):
            mlflow.log_metrics({
                "latency_seconds":   llm_meta["latency_seconds"],
                "prompt_tokens":     float(llm_meta["prompt_tokens"]),
                "completion_tokens": float(llm_meta["completion_tokens"]),
                "total_tokens":      float(llm_meta["total_tokens"]),
            })
        else:
            mlflow.set_tag("extraction_error", llm_meta.get("error", "unknown"))

        # Artifacts
        mlflow.log_artifact(str(image_path), artifact_path="chart_images")

        gt_path = image_path.parent / f"gt_{chart_type}.json"
        gt_path.write_text(json.dumps(ground_truth, indent=2))
        mlflow.log_artifact(str(gt_path), artifact_path="ground_truth")

        if extracted is not None:
            ext_path = image_path.parent / f"extracted_{chart_type}_{model}.json"
            ext_path.write_text(extracted.model_dump_json(indent=2))
            mlflow.log_artifact(str(ext_path), artifact_path="extracted")

        return run.info.run_id


# ── LLM call log (separate experiment) ───────────────────────────────────────

def log_llm_call(
    *,
    model:      str,
    chart_type: str,
    resolution: str,
    llm_meta:   dict,
) -> None:
    """Log a single LLM API call into the llm_calls experiment."""
    mlflow.set_experiment(LLM_CALLS_EXPERIMENT)
    run_name = f"{model}__{chart_type}__{resolution}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model":      model,
            "chart_type": chart_type,
            "resolution": resolution,
        })
        mlflow.log_metrics({
            "latency_seconds":   llm_meta.get("latency_seconds", 0.0),
            "prompt_tokens":     float(llm_meta.get("prompt_tokens", 0)),
            "completion_tokens": float(llm_meta.get("completion_tokens", 0)),
            "total_tokens":      float(llm_meta.get("total_tokens", 0)),
            "success":           float(llm_meta.get("success", False)),
        })
        if not llm_meta.get("success"):
            mlflow.set_tag("error", llm_meta.get("error", "unknown"))
