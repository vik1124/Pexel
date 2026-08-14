"""
Compute precision, recall, F1, and accuracy for LLM chart-data extraction.

Strategy
--------
Every chart type is reduced to a flat list of "slots":
    (label: str | None, value: float)

Ground-truth slots are compared to extracted slots.
Matching uses:
  - label similarity (fuzzy string ratio) when labels exist
  - numeric proximity fallback for unlabelled slots

A match is a True Positive only if the numeric value is also within TOLERANCE.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from config import LABEL_SIMILARITY_THRESHOLD, NUMERIC_TOLERANCE
from schemas import (
    AreaChartExtraction,
    BarChartExtraction,
    BoxPlotExtraction,
    BubbleChartExtraction,
    CandlestickExtraction,
    DualAxisExtraction,
    FunnelExtraction,
    HeatmapExtraction,
    HistogramExtraction,
    LineChartExtraction,
    PieChartExtraction,
    RadarExtraction,
    ScatterChartExtraction,
    StackedBarExtraction,
    SubplotsExtraction,
    SurfaceExtraction,
    TreemapExtraction,
    ViolinExtraction,
    WaterfallExtraction,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _within(gt: float, pred: float, tol: float = NUMERIC_TOLERANCE) -> bool:
    if gt == 0 and pred == 0:
        return True
    denom = abs(gt) if gt != 0 else abs(pred)
    return abs(gt - pred) / denom <= tol


def _metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    total = tp + fp + fn
    accuracy  = tp / total if total > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "accuracy":  round(accuracy,  4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_gt": tp + fn,
    }


# ── Slot converters ───────────────────────────────────────────────────────────

Slot = tuple[str | None, float]   # (label_or_None, numeric_value)



def _pred_slots(chart_type: str, extracted: Any) -> list[Slot]:
    if extracted is None:
        return []

    if chart_type in ("bar_vertical", "bar_horizontal"):
        assert isinstance(extracted, BarChartExtraction)
        return [(dp.label, dp.value) for dp in extracted.data_points]

    if chart_type == "pie":
        assert isinstance(extracted, PieChartExtraction)
        return [(s.label, s.percentage) for s in extracted.slices]

    if chart_type in ("line", "area"):
        model = extracted
        return [(f"{dp.series}|{dp.x}", dp.y) for dp in model.data_points]

    if chart_type == "scatter":
        assert isinstance(extracted, ScatterChartExtraction)
        return [(p.label, p.y) for p in extracted.points]

    if chart_type == "histogram":
        assert isinstance(extracted, HistogramExtraction)
        return [(f"{b.bin_start}-{b.bin_end}", b.count) for b in extracted.bins]

    if chart_type == "box":
        assert isinstance(extracted, BoxPlotExtraction)
        slots: list[Slot] = []
        for g in extracted.groups:
            for stat, val in [
                ("median", g.median), ("q1", g.q1), ("q3", g.q3),
                ("lower", g.lower_whisker), ("upper", g.upper_whisker),
            ]:
                if val is not None:
                    slots.append((f"{g.group}|{stat}", val))
        return slots

    if chart_type == "heatmap":
        assert isinstance(extracted, HeatmapExtraction)
        return [(f"{c.row}|{c.col}", c.value) for c in extracted.cells]

    if chart_type == "dual_axis":
        assert isinstance(extracted, DualAxisExtraction)
        slots: list[Slot] = []
        for dp in extracted.data_points:
            if dp.bar_value is not None:
                slots.append((f"{dp.x_label}|bar",  dp.bar_value))
            if dp.line_value is not None:
                slots.append((f"{dp.x_label}|line", dp.line_value))
        return slots

    if chart_type == "log_scale":
        assert isinstance(extracted, LineChartExtraction)
        return [(f"{dp.series}|{dp.x}", dp.y) for dp in extracted.data_points]

    if chart_type == "subplots_2x2":
        assert isinstance(extracted, SubplotsExtraction)
        slots = []
        for panel in extracted.panels:
            for dp in panel.data_points:
                slots.append((f"{panel.panel_title}|{dp.label}", dp.value))
        return slots

    if chart_type == "bubble":
        assert isinstance(extracted, BubbleChartExtraction)
        slots = []
        for p in extracted.points:
            lbl = p.label or ""
            slots.append((f"{lbl}|x",    p.x))
            slots.append((f"{lbl}|y",    p.y))
            slots.append((f"{lbl}|size", p.size))
        return slots

    if chart_type == "waterfall":
        assert isinstance(extracted, WaterfallExtraction)
        return [(b.label, b.value) for b in extracted.bars]

    if chart_type == "candlestick":
        assert isinstance(extracted, CandlestickExtraction)
        slots = []
        for c in extracted.candles:
            slots += [
                (f"{c.date}|open",  c.open),
                (f"{c.date}|high",  c.high),
                (f"{c.date}|low",   c.low),
                (f"{c.date}|close", c.close),
            ]
        return slots

    if chart_type == "violin":
        assert isinstance(extracted, ViolinExtraction)
        slots = []
        for g in extracted.groups:
            for stat, val in [
                ("median", g.median), ("q1", g.q1), ("q3", g.q3),
                ("lower", g.lower_whisker), ("upper", g.upper_whisker),
            ]:
                if val is not None:
                    slots.append((f"{g.group}|{stat}", val))
        return slots

    if chart_type == "radar":
        assert isinstance(extracted, RadarExtraction)
        slots = []
        for series in extracted.series:
            for dp in series.data_points:
                slots.append((f"{series.series_name}|{dp.metric}", dp.value))
        return slots

    if chart_type in ("stacked_bar", "stacked_bar_100"):
        assert isinstance(extracted, StackedBarExtraction)
        return [(f"{dp.series}|{dp.category}", dp.value) for dp in extracted.data_points]

    if chart_type == "surface_3d":
        assert isinstance(extracted, SurfaceExtraction)
        return [(None, p.z) for p in extracted.sample_points]

    if chart_type == "funnel":
        assert isinstance(extracted, FunnelExtraction)
        return [(s.stage, s.value) for s in extracted.stages]

    if chart_type == "treemap":
        assert isinstance(extracted, TreemapExtraction)
        return [(f"{n.parent}|{n.label}" if n.parent else n.label, n.value)
                for n in extracted.nodes]

    return []


# ── Greedy label-aware matching ───────────────────────────────────────────────

def _match(
    gt_slots:   list[Slot],
    pred_slots: list[Slot],
) -> tuple[int, int, int]:
    """
    Return (tp, fp, fn) via greedy label-aware matching.

    Only pairs where label similarity >= threshold AND value within tolerance
    are treated as True Positives.  A label match with wrong value is a
    near-miss that counts as both FP (bad prediction) and FN (missed GT).
    """
    matched_gt   = [False] * len(gt_slots)
    matched_pred = [False] * len(pred_slots)

    tp_pairs: list[tuple[float, int, int]] = []
    for pi, (pl, pv) in enumerate(pred_slots):
        for gi, (gl, gv) in enumerate(gt_slots):
            if pl is not None and gl is not None:
                sim = _sim(pl, gl)
            else:
                denom = max(abs(gv), abs(pv), 1e-9)
                sim = max(0.0, 1.0 - abs(gv - pv) / denom)

            if sim >= LABEL_SIMILARITY_THRESHOLD and _within(gv, pv):
                tp_pairs.append((sim, pi, gi))

    tp_pairs.sort(reverse=True)
    tp = 0
    for _, pi, gi in tp_pairs:
        if matched_pred[pi] or matched_gt[gi]:
            continue
        tp += 1
        matched_pred[pi] = True
        matched_gt[gi]   = True

    fp = sum(1 for m in matched_pred if not m)
    fn = sum(1 for m in matched_gt   if not m)
    return tp, fp, fn


def _match_detail(
    gt_slots:   list[Slot],
    pred_slots: list[Slot],
) -> tuple[int, int, int, list[dict]]:
    """Same as _match but also returns per-slot comparison rows for human review."""
    matched_gt   = [False] * len(gt_slots)
    matched_pred = [False] * len(pred_slots)

    tp_pairs: list[tuple[float, int, int]] = []
    for pi, (pl, pv) in enumerate(pred_slots):
        for gi, (gl, gv) in enumerate(gt_slots):
            if pl is not None and gl is not None:
                sim = _sim(pl, gl)
            else:
                denom = max(abs(gv), abs(pv), 1e-9)
                sim = max(0.0, 1.0 - abs(gv - pv) / denom)

            if sim >= LABEL_SIMILARITY_THRESHOLD and _within(gv, pv):
                tp_pairs.append((sim, pi, gi))

    tp_pairs.sort(reverse=True)
    tp = 0
    matched_pairs: list[tuple[int, int]] = []
    for _, pi, gi in tp_pairs:
        if matched_pred[pi] or matched_gt[gi]:
            continue
        tp += 1
        matched_pred[pi] = True
        matched_gt[gi]   = True
        matched_pairs.append((gi, pi))

    rows: list[dict] = []

    for gi, pi in matched_pairs:
        gl, gv = gt_slots[gi]
        pl, pv = pred_slots[pi]
        rows.append({
            "status":          "match",
            "label":           gl or pl or "",
            "gt_value":        gv,
            "extracted_value": pv,
            "diff":            round(pv - gv, 4),
            "comment":         "",
        })

    for gi, matched in enumerate(matched_gt):
        if not matched:
            gl, gv = gt_slots[gi]
            rows.append({
                "status":          "missed (GT only)",
                "label":           gl or "",
                "gt_value":        gv,
                "extracted_value": "",
                "diff":            "",
                "comment":         "",
            })

    for pi, matched in enumerate(matched_pred):
        if not matched:
            pl, pv = pred_slots[pi]
            rows.append({
                "status":          "extra (pred only)",
                "label":           pl or "",
                "gt_value":        "",
                "extracted_value": pv,
                "diff":            "",
                "comment":         "",
            })

    fp = sum(1 for m in matched_pred if not m)
    fn = sum(1 for m in matched_gt   if not m)
    return tp, fp, fn, rows


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(chart_type: str, ground_truth: Any, extracted: Any) -> dict:
    """
    Compare extracted Pydantic model against ground_truth dict.

    Returns a dict with keys: precision, recall, f1, accuracy, tp, fp, fn, total_gt
    """
    gt_s   = _pred_slots(chart_type, ground_truth)
    pred_s = _pred_slots(chart_type, extracted)

    if not gt_s:
        return _metrics(0, len(pred_s), 0)

    tp, fp, fn = _match(gt_s, pred_s)
    return _metrics(tp, fp, fn)


def evaluate_detail(
    chart_type: str, ground_truth: Any, extracted: Any
) -> tuple[dict, list[dict]]:
    """
    Like evaluate() but also returns per-slot comparison rows for human review.

    Returns (metrics_dict, rows) where rows is a list of dicts suitable for
    create_table_artifact — each row has: status, label, gt_value, extracted_value, diff.
    """
    gt_s   = _pred_slots(chart_type, ground_truth)
    pred_s = _pred_slots(chart_type, extracted)

    if not gt_s:
        rows = [
            {"status": "extra (pred only)", "label": pl or "", "gt_value": "", "extracted_value": pv, "diff": ""}
            for pl, pv in pred_s
        ]
        return _metrics(0, len(pred_s), 0), rows

    tp, fp, fn, rows = _match_detail(gt_s, pred_s)
    return _metrics(tp, fp, fn), rows
