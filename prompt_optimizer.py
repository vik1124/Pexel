"""
Pydantic-AI agent that iteratively optimizes the chart-extraction system prompt
to maximize mean F1 across a representative sample of chart types.

Usage:
    python prompt_optimizer.py

The optimizer:
  1. Evaluates the current system prompt against SAMPLE_CHARTS (extract + evaluate)
  2. Passes failure details to a durable Pydantic-AI agent that proposes a better prompt
  3. Repeats for MAX_ITERATIONS, tracking the best-ever prompt
  4. Saves full history + best prompt to output/prompt_optimizer_results.json

Constraint: the proposed prompt must be generic — no specific data values,
labels, or chart instances may appear in it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, task
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

from chart_generator import chart_factory
from evaluator import evaluate_detail
from llm_extractor import extract, _PROMPTS, _SYSTEM_PROMPT
import config as cfg

load_dotenv()

MODEL = "openai:gpt-5.6-luna"
RESOLUTION = "high"
MAX_ITERATIONS = 6
CONVERGE_F1 = 0.95

SAMPLE_CHARTS = [
    "bar_vertical",
    "line",
    "pie",
    "heatmap",
    "scatter",
    "waterfall",
    "box",
    "stacked_bar",
]


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChartEval(BaseModel):
    chart_type: str
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    missed_labels: list[str]
    extra_labels: list[str]


class IterationResult(BaseModel):
    iteration: int
    system_prompt: str
    chart_evals: list[ChartEval]
    mean_f1: float


class PromptProposal(BaseModel):
    reasoning: str = Field(
        description=(
            "Step-by-step analysis: which chart types failed, what error patterns you see "
            "(missed labels, hallucinated points, value inaccuracy), and how the new prompt addresses them."
        )
    )
    system_prompt: str = Field(
        description=(
            "Improved generic system prompt for a vision LLM extracting data from charts. "
            "Must work for ANY chart with ANY data — no specific values, labels, domain names, "
            "or chart instances. Maximum 250 words."
        )
    )


# ── Optimizer agent ───────────────────────────────────────────────────────────

_OPTIMIZER_SYSTEM = (
    "You are an expert prompt engineer. Your job is to improve a generic system prompt "
    "that instructs a vision LLM to extract structured data from chart images.\n\n"
    "The extraction model receives:\n"
    "  1. This system prompt (what you write)\n"
    "  2. A chart image\n"
    "  3. A short chart-type-specific instruction (e.g. 'extract every bar…')\n\n"
    "Extracted data is compared to ground truth via slot-matching (fuzzy label similarity + "
    "10% numeric tolerance). Precision, recall, and F1 are computed per chart.\n\n"
    "Your goal: propose a system prompt that maximises mean F1 across all chart types.\n\n"
    "Hard constraints:\n"
    "- Generic: no specific data values, labels, axis units, or chart instances\n"
    "- Concise: under 250 words\n"
    "- Builds on previous iterations — don't regress on charts that already score well\n"
)

optimizer_agent = Agent(
    MODEL,
    output_type=PromptProposal,
    system_prompt=_OPTIMIZER_SYSTEM,
)


# ── Evaluation helpers ────────────────────────────────────────────────────────

async def _eval_one(chart_type: str, system_prompt: str) -> ChartEval:
    chart = chart_factory(chart_type)
    image_path = cfg.CHARTS_DIR / RESOLUTION / f"{chart_type}.png"

    extracted, meta = await extract(
        image_path=image_path,
        extraction_class=chart.extraction_class,
        system_prompt=system_prompt,
        prompt=_PROMPTS.get(chart_type, ""),
        model_name=MODEL,
    )

    if not meta["success"] or extracted is None:
        return ChartEval(
            chart_type=chart_type, f1=0.0, precision=0.0, recall=0.0,
            tp=0, fp=0, fn=0, missed_labels=[], extra_labels=[],
        )

    metrics, rows = evaluate_detail(chart_type, chart.ground_truth, extracted)
    missed = [r["label"] for r in rows if r["status"] == "missed (GT only)"]
    extra  = [r["label"] for r in rows if r["status"] == "extra (pred only)"]

    return ChartEval(
        chart_type=chart_type,
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        tp=metrics["tp"],
        fp=metrics["fp"],
        fn=metrics["fn"],
        missed_labels=missed[:10],
        extra_labels=extra[:10],
    )


@task(name="evaluate_prompt", retries=2, log_prints=True)
async def evaluate_prompt(iteration: int, system_prompt: str) -> IterationResult:
    evals = await asyncio.gather(*[_eval_one(ct, system_prompt) for ct in SAMPLE_CHARTS])
    mean_f1 = round(sum(e.f1 for e in evals) / len(evals), 4)
    return IterationResult(
        iteration=iteration,
        system_prompt=system_prompt,
        chart_evals=list(evals),
        mean_f1=mean_f1,
    )


@task(name="propose_next_prompt", retries=2, log_prints=True)
async def propose_next_prompt(feedback: str, history_bytes: bytes | None) -> tuple[str, bytes]:
    """
    Durable: deserializes the agent's full conversation history from the previous
    call and passes it back in, so the agent retains context across all iterations
    rather than starting fresh each time.
    """
    msg_history = ModelMessagesTypeAdapter.validate_json(history_bytes) if history_bytes else None
    result = await optimizer_agent.run(feedback, message_history=msg_history)
    print(f"Reasoning: {result.output.reasoning[:300].strip()}…")
    return result.output.system_prompt, ModelMessagesTypeAdapter.dump_json(result.all_messages())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_feedback(history: list[IterationResult]) -> str:
    lines: list[str] = []
    for r in history[-3:]:  # last 3 iterations keeps the message lean
        lines.append(f"\n=== Iteration {r.iteration}  Mean F1: {r.mean_f1:.4f} ===")
        lines.append(f"System prompt:\n{r.system_prompt}\n")
        for ev in r.chart_evals:
            lines.append(
                f"  {ev.chart_type:20s}  F1={ev.f1:.3f}  "
                f"P={ev.precision:.3f}  R={ev.recall:.3f}  "
                f"TP={ev.tp}  FP={ev.fp}  FN={ev.fn}"
            )
            if ev.missed_labels:
                lines.append(f"    missed     : {ev.missed_labels}")
            if ev.extra_labels:
                lines.append(f"    hallucinated: {ev.extra_labels}")
    return "\n".join(lines)


def _save(path: Path, history: list[IterationResult], best_prompt: str, best_f1: float) -> None:
    path.write_text(json.dumps({
        "best_f1":     best_f1,
        "best_prompt": best_prompt,
        "iterations":  [r.model_dump() for r in history],
    }, indent=2))


def _generate_chart(chart_type: str) -> None:
    chart = chart_factory(chart_type)
    chart.generate_plot()
    chart.save_images()


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="prompt_optimizer", log_prints=True)
async def run_optimizer() -> None:
    print("Flow running")
    output_path = Path("output/prompt_optimizer_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate any missing chart images synchronously before async work begins
    # res_dir = cfg.CHARTS_DIR / RESOLUTION
    # for ct in SAMPLE_CHARTS:
    #     if not (res_dir / f"{ct}.png").exists():
    #         print(f"Generating missing image: {ct}")
    #         _generate_chart(ct)

    current_prompt = _SYSTEM_PROMPT
    history: list[IterationResult] = []
    best_f1 = 0.0
    best_prompt = current_prompt
    history_bytes: bytes | None = None  # serialized optimizer agent conversation

    print(f"Prompt optimizer — model: {MODEL}  charts: {SAMPLE_CHARTS}")
    print(f"Max iterations: {MAX_ITERATIONS}  converge target: {CONVERGE_F1}\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"─── Iteration {iteration}/{MAX_ITERATIONS} ───────────────────────────")
        print(f"Prompt preview: {current_prompt[:90].strip()}…\n")

        result = await evaluate_prompt(iteration, current_prompt)
        history.append(result)

        for ev in result.chart_evals:
            marker = "★" if ev.f1 >= 0.90 else ("△" if ev.f1 >= 0.70 else "✗")
            print(f"  {marker} {ev.chart_type:20s}  F1={ev.f1:.3f}  P={ev.precision:.3f}  R={ev.recall:.3f}")

        print(f"\n  Mean F1: {result.mean_f1:.4f}", end="")

        if result.mean_f1 > best_f1:
            best_f1 = result.mean_f1
            best_prompt = current_prompt
            print("  ← new best")
        else:
            print()

        _save(output_path, history, best_prompt, best_f1)

        if result.mean_f1 >= CONVERGE_F1 or iteration == MAX_ITERATIONS:
            break

        feedback = _format_feedback(history)
        current_prompt, history_bytes = await propose_next_prompt(feedback, history_bytes)

    print(f"\n{'='*60}")
    print(f"Optimization complete — best mean F1: {best_f1:.4f}")
    print(f"\nBest system prompt:\n{best_prompt}")
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_optimizer())
