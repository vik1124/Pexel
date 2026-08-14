"""Semantic chart-data evaluator using a pydantic-ai agent instead of fuzzy heuristics.

Diagnostic use: if this gives high scores where the heuristic evaluator gives low scores,
the problem is in label/value matching — not in chart extraction.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

from evaluator import _pred_slots, _metrics, Slot


class SlotComparison(BaseModel):
    status: Literal["match", "missed", "extra"]
    label: str = Field(default="", description="Label for this data point")
    gt_value: Optional[float] = Field(None, description="Ground truth value; None if extra")
    extracted_value: Optional[float] = Field(None, description="Extracted value; None if missed")
    diff: Optional[float] = Field(None, description="extracted_value - gt_value for matches")
    comment: Optional[str] = Field(default="", description="if diff, Reason why this discrepancy happened (best guess)")


class LLMMatchResult(BaseModel):
    comparisons: list[SlotComparison]
    tp: int = Field(description="Correct matches")
    fp: int = Field(description="Extracted points with no GT match")
    fn: int = Field(description="GT points not found in extraction")
    comment: Optional[str] = Field(default="", description="if tp, fp or fn !=1, Reason why this discrepancy happened (best guess)")


_SYSTEM_PROMPT = (
    "You are evaluating chart data extraction accuracy. "
    "Match extracted data points to ground truth data points. "
    "A match requires both: (1) the label is semantically equivalent — be lenient on formatting "
    "('Jan' matches 'January', '2020' matches '2020.0', 'Q1' matches 'Quarter 1'); "
    "AND (2) the numeric value is within 10% of the ground truth. "
    "Output every GT point as 'match' or 'missed', and every unmatched extracted point as 'extra'. "
    "tp = number of matches, fp = number of extra, fn = number of missed."
)


def _format_slots(slots: list[Slot]) -> str:
    if not slots:
        return "  (empty)"
    lines = [f"  [{i}] label={repr(label) if label is not None else '(none)'}  value={value}"
             for i, (label, value) in enumerate(slots, 1)]
    return "\n".join(lines)


async def evaluate_with_llm(
    chart_type: str,
    ground_truth: BaseModel,
    extracted: BaseModel,
    model_name: str = "openai:gpt-5.5",
) -> tuple[dict, list[dict]]:
    """Semantically match GT vs extracted slots using an LLM.

    Returns (metrics_dict, comparison_rows) in the same format as evaluator.evaluate_detail().
    """
    gt_slots   = _pred_slots(chart_type, ground_truth)
    pred_slots = _pred_slots(chart_type, extracted)

    if not gt_slots:
        rows = [
            {"status": "extra (pred only)", "label": pl or "", "gt_value": "", "extracted_value": pv, "diff": ""}
            for pl, pv in pred_slots
        ]
        return _metrics(0, len(pred_slots), 0), rows

    if not pred_slots:
        rows = [
            {"status": "missed (GT only)", "label": gl or "", "gt_value": gv, "extracted_value": "", "diff": ""}
            for gl, gv in gt_slots
        ]
        return _metrics(0, 0, len(gt_slots)), rows

    prompt = (
        f"Chart type: {chart_type}\n\n"
        f"Ground truth :\n{ground_truth.model_dump_json()}\n\n"
        f"Extracted :\n{extracted.model_dump_json()}\n\n"
        "Match each extracted point to the closest Ground Truth point. "
        "Output every GT point (match or missed) and any unmatched extracted points (extra). "
        "Set tp/fp/fn counts accordingly."
    )

    agent = Agent(model_name, output_type=LLMMatchResult, system_prompt=_SYSTEM_PROMPT, capabilities=[Thinking(effort="high")])
    result = await agent.run(prompt)
    llm_result = result.output

    rows: list[dict] = []
    for cmp in llm_result.comparisons:
        if cmp.status == "match":
            rows.append({
                "status":          "match",
                "label":           cmp.label,
                "gt_value":        cmp.gt_value,
                "extracted_value": cmp.extracted_value,
                "diff":            cmp.diff,
                "comment":         cmp.comment,
            })
        elif cmp.status == "missed":
            rows.append({
                "status":          "missed (GT only)",
                "label":           cmp.label,
                "gt_value":        cmp.gt_value,
                "extracted_value": "",
                "diff":            "",
                "comment":         cmp.comment,
            })
        else:
            rows.append({
                "status":          "extra (pred only)",
                "label":           cmp.label,
                "gt_value":        "",
                "extracted_value": cmp.extracted_value,
                "diff":            "",
                "comment":         cmp.comment,
            })

    return _metrics(llm_result.tp, llm_result.fp, llm_result.fn), rows
