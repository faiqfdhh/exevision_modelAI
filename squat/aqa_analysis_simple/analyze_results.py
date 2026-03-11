from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUALITY_FOLDERS = ("excellent", "good", "fair")
SCORE_TIERS = ("good", "acceptable", "poor")


def get_view_thresholds(view: str) -> dict[str, dict[str, float | bool]]:
    view_lower = str(view).lower()

    default_thresholds = {
        "knee_valgus": {"good": 0.95, "bad": 0.75, "higher_is_better": True},
        "forward_lean": {"good": 20.0, "bad": 45.0, "higher_is_better": False},
        "depth": {"good": 95.0, "bad": 125.0, "higher_is_better": False},
        "squat_depth": {"good": 0.1, "bad": -0.1, "higher_is_better": True},
    }

    if "side" in view_lower and "front" not in view_lower and "back" not in view_lower:
        return {
            "knee_valgus": {"good": 0.95, "bad": 0.70, "higher_is_better": True},
            "forward_lean": {"good": 30.0, "bad": 60.0, "higher_is_better": False},
            "depth": {"good": 70.0, "bad": 120.0, "higher_is_better": False},
            "squat_depth": {"good": 0.15, "bad": -0.05, "higher_is_better": True},
        }

    if view_lower in ["front", "back"]:
        return {
            "knee_valgus": {"good": 0.97, "bad": 0.80, "higher_is_better": True},
            "forward_lean": {"good": 25.0, "bad": 50.0, "higher_is_better": False},
            "depth": {"good": 100.0, "bad": 130.0, "higher_is_better": False},
            "squat_depth": {"good": 0.08, "bad": -0.08, "higher_is_better": True},
        }

    if "front_side" in view_lower or "front-side" in view_lower:
        return {
            "knee_valgus": {"good": 0.95, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 18.0, "bad": 40.0, "higher_is_better": False},
            "depth": {"good": 95.0, "bad": 122.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.05, "higher_is_better": True},
        }

    if "back_side" in view_lower or "back-side" in view_lower:
        return {
            "knee_valgus": {"good": 1.2, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 35.0, "bad": 50.0, "higher_is_better": False},
            "depth": {"good": 95.0, "bad": 122.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.05, "higher_is_better": True},
        }

    return default_thresholds


def metric_specs() -> dict[str, dict[str, Any]]:
    return {
        "knee_valgus": {
            "label": "Knee tracking",
            "source_key": "knee_valgus",
            "unit": "ratio",
            "evaluation": "direct",
            "good_direction": "higher",
        },
        "forward_lean": {
            "label": "Forward lean",
            "source_key": "forward_lean",
            "unit": "deg",
            "evaluation": "absolute",
            "good_direction": "lower",
        },
        "depth": {
            "label": "Depth by knee angle",
            "source_key": "min_knee_angle",
            "unit": "deg",
            "evaluation": "direct",
            "good_direction": "lower",
        },
        "squat_depth": {
            "label": "Bottom depth",
            "source_key": "squat_depth",
            "unit": "normalized",
            "evaluation": "direct",
            "good_direction": "higher",
        },
    }


def severity_from_score(metric_score: float) -> str:
    if metric_score >= 90:
        return "Good"
    if metric_score >= 75:
        return "Slight"
    if metric_score >= 50:
        return "Moderate"
    if metric_score >= 25:
        return "Major"
    return "Severe"


def format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "N/A"
    if unit == "deg":
        return f"{value:.2f} deg"
    if unit == "ratio":
        return f"{value:.3f}"
    return f"{value:.3f}"


def threshold_text(good: float, bad: float, higher_is_better: bool, unit: str) -> str:
    good_text = format_value(good, unit)
    bad_text = format_value(bad, unit)
    if higher_is_better:
        return f"good if >= {good_text}; bad if <= {bad_text}"
    return f"good if <= {good_text}; bad if >= {bad_text}"


def evaluate_metric(rep: dict[str, Any], metric_name: str, view: str) -> dict[str, Any] | None:
    specs = metric_specs()[metric_name]
    metrics = rep.get("metrics", {})
    score = rep.get("score", {})
    metric_scores = score.get("metric_scores", {})
    weights = score.get("weights_used", {})
    raw_value = metrics.get(specs["source_key"])
    metric_score = metric_scores.get(metric_name)

    if raw_value is None or metric_score is None:
        return None

    thresholds = get_view_thresholds(view)[metric_name]
    evaluated_value = abs(raw_value) if specs["evaluation"] == "absolute" else raw_value
    higher_is_better = bool(thresholds["higher_is_better"])
    good_value = float(thresholds["good"])
    bad_value = float(thresholds["bad"])
    violated = evaluated_value < good_value if higher_is_better else evaluated_value > good_value
    evaluation_text = (
        f"abs({raw_value:.2f}) = {evaluated_value:.2f} deg"
        if specs["evaluation"] == "absolute"
        else format_value(evaluated_value, specs["unit"])
    )

    detail = {
        "metric_key": metric_name,
        "label": specs["label"],
        "raw_value": raw_value,
        "evaluated_value": evaluated_value,
        "metric_score": float(metric_score),
        "weight": float(weights.get(metric_name, 0.0)),
        "severity": severity_from_score(float(metric_score)),
        "violated": violated,
        "evaluation_mode": specs["evaluation"],
        "evaluation_text": evaluation_text,
        "threshold_text": threshold_text(good_value, bad_value, higher_is_better, specs["unit"]),
        "higher_is_better": higher_is_better,
        "good_threshold": good_value,
        "bad_threshold": bad_value,
        "user_message": (
            f"{specs['label']}: {severity_from_score(float(metric_score))}. "
            f"Detected {evaluation_text}. Stage 8 evaluated this as "
            f"{'higher is better' if higher_is_better else 'lower is better'} for {view.replace('_', ' ')} view, "
            f"with {threshold_text(good_value, bad_value, higher_is_better, specs['unit'])}. "
            f"Metric score: {float(metric_score):.1f}/100. Weight used: {float(weights.get(metric_name, 0.0)):.2f}."
        ),
    }
    return detail


def build_rep_summary(rep: dict[str, Any], view: str) -> dict[str, Any]:
    diagnostics = []
    for metric_name in ("knee_valgus", "forward_lean", "depth", "squat_depth"):
        detail = evaluate_metric(rep, metric_name, view)
        if detail is not None:
            diagnostics.append(detail)

    diagnostics.sort(key=lambda item: item["metric_score"])
    causes = [item for item in diagnostics if item["violated"]]
    headline = "All scored metrics were within the target range."
    if causes:
        top = causes[0]
        headline = (
            f"Main issue: {top['label']} ({top['severity'].lower()}) with value "
            f"{top['evaluation_text']} and metric score {top['metric_score']:.1f}/100."
        )

    return {
        "rep_id": rep.get("rep_id"),
        "overall_score": rep.get("score", {}).get("overall_score", 0.0),
        "headline": headline,
        "diagnostics": diagnostics,
        "causes": causes,
    }


def build_video_summary(score_data: dict[str, Any]) -> dict[str, Any]:
    video_id = score_data.get("video_id", "unknown")
    view = score_data.get("view", "unknown")
    repetitions = score_data.get("repetitions", [])
    rep_summaries = [build_rep_summary(rep, view) for rep in repetitions]

    summary_lines = [
        f"Video {video_id} scored {float(score_data.get('overall_score', 0.0)):.1f}/100 from {len(repetitions)} repetition(s).",
        f"View used by scoring: {str(view).replace('_', ' ')}.",
    ]
    if not rep_summaries:
        summary_lines.append("No repetitions were available to analyze.")
    else:
        for rep in rep_summaries:
            summary_lines.append(
                f"Rep {rep['rep_id']}: {float(rep['overall_score']):.1f}/100. {rep['headline']}"
            )

    return {
        "video_id": video_id,
        "overall_score": score_data.get("overall_score", 0.0),
        "view": view,
        "rep_count": len(repetitions),
        "summary_lines": summary_lines,
        "repetitions": rep_summaries,
    }


def detect_base_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if any((script_dir / quality).exists() for quality in QUALITY_FOLDERS):
        return script_dir
    return Path("./squat/aqa_analysis_simple").resolve()


def iter_score_files(base_dir: Path, video_id: str | None = None) -> list[Path]:
    matches: list[Path] = []
    for quality in QUALITY_FOLDERS:
        for tier in SCORE_TIERS:
            tier_dir = base_dir / quality / tier
            if not tier_dir.exists():
                continue
            if video_id is None:
                matches.extend(sorted(tier_dir.glob("*_aqa_simple.json")))
            else:
                target = tier_dir / f"{video_id}_aqa_simple.json"
                if target.exists():
                    matches.append(target)
    return matches


def save_summary(base_dir: Path, summary: dict[str, Any]) -> Path:
    output_dir = base_dir / "analysis_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{summary['video_id']}_analysis_summary.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain what caused squat scores to drop.")
    parser.add_argument("--video-id", help="Analyze only one video id.")
    args = parser.parse_args()

    base_dir = detect_base_dir()
    score_files = iter_score_files(base_dir, args.video_id)
    if not score_files:
        print("No scoring JSON files found to analyze.")
        return 1

    written = 0
    for score_file in score_files:
        with open(score_file, "r", encoding="utf-8") as handle:
            score_data = json.load(handle)
        summary = build_video_summary(score_data)
        output_path = save_summary(base_dir, summary)
        written += 1
        print(f"Saved analysis summary: {output_path}")

    print(f"Generated {written} analysis summary file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())