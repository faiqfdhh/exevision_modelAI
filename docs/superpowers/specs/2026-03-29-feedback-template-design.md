# Feedback Template System Design
**Date:** 2026-03-29
**Status:** Approved
**Scope:** ExeVision AI — Modular Slot-Based Feedback Engine

---

## 1. Overview

The Feedback Engine produces **structured, tone-first coaching text** for each rep and a session summary. It uses **Dynamic Template Slot-Filling** (no generative AI) to ensure deterministic, hallucination-free feedback that aligns with the user's actual numerical scores.

**Core principle:** The overall score dictates the tone. Issues are only surfaced if they score below a threshold. Positive reinforcement is included using relative language (no raw numbers).

---

## 2. Architecture: Modular (Approach 2)

Two independent layers:

- **Exercise Config Layer** (`core/exevision/config/exercises/*.json`) — defines thresholds, issue groups, score brackets, and metric targets per exercise
- **Template Library Layer** (`core/exevision/config/templates/feedback_templates.json`) — defines reusable phrasing templates, improvement phrases, and win phrases

The `core/exevision/feedback/` module contains the Python logic that combines both layers to produce feedback.

---

## 3. Folder Structure

```
core/exevision/
├── feedback/
│   ├── __init__.py
│   ├── engine.py               # FeedbackEngine — main entry point
│   ├── rep_comparator.py       # Computes rep-over-rep deltas
│   ├── session_aggregator.py   # Builds session summary
│   └── template_renderer.py   # Slot-filling renderer
│
├── config/
│   ├── exercises/
│   │   ├── squat.json          # Squat-specific config
│   │   └── deadlift.json       # (future)
│   └── templates/
│       └── feedback_templates.json  # Shared template library
```

---

## 4. Score Brackets & Tone Tiers

Brackets are **exercise-specific** (defined per `exercises/*.json`). Default squat brackets:

| Score Range | Tier       | Opener Example                                   |
|-------------|------------|--------------------------------------------------|
| 90–100      | excellent  | "Excellent squat form!"                          |
| 75–89       | good       | "Good squat overall."                            |
| 60–74       | fair       | "Decent effort, but a few things to address."    |
| 40–59       | poor       | "Your form needs significant attention."         |
| 0–39        | critical   | "Let's work on the fundamentals."                |

---

## 5. Issue Surfacing Rules

### Threshold
- Only mention a metric if its subscore is **< 75** (universal threshold, per exercise config)

### Severity Banding (±5 rule)
- Issues are **sorted by severity** (lowest score = highest priority)
- Issues within **±5 points** of each other are treated with **the same urgency/tone**
- Example: scores 62, 65, 72 → first two are same urgency band; 72 is softer band

### Issue Grouping (Predefined Taxonomy)
- Related metrics are **combined into one coaching cue** using the exercise's `issue_groups` taxonomy
- Example for squat:
  - `descent_quality`: groups `forward_lean` + `hip_depth`
  - `knee_stability`: groups `knee_valgus` + `knee_tracking`
- If both metrics in a group score < 75 → combined cue; if only one → single metric cue

### All Issues Mentioned
- No cap on number of issues mentioned — **all issues below threshold are surfaced**
- Grouped issues count as one mention

---

## 6. Positive Reinforcement

- If a metric scores **≥ 75** AND has **improved relative to the previous rep**, it is called out as a win
- Language is **relative, no exact numbers**: e.g., "Great depth this rep — notably better than last time!" not "Great depth (82)!"
- Wins are mentioned **before** issues in per-rep feedback

---

## 7. Rep-Relative Progress Tracking

- `rep_comparator.py` computes **percentage improvement** between each rep and the previous
- Improvement phrases are tiered:
  - **Significant (≥15% improvement):** "notably better than last rep"
  - **Moderate (8–14%):** "getting better", "improving from last rep"
  - **Slight (1–7%):** "a little better than last rep"
- Rep 1 has no comparison (no "from last rep" language)
- Comparison is applied both to **wins** (positive reinforcement) and **issue improvement** (e.g., "forward lean improving!")

---

## 8. Per-Rep Feedback Structure

```
[TONE_OPENER]                     ← from score bracket
[WIN_1] [WIN_2] (if any)          ← relative language, improving metrics ≥75
[ISSUE_GROUP_1] (highest severity band)
[ISSUE_GROUP_2] (next severity band, if different)
...
```

**Example — Rep 2, Score 78, improving from Rep 1 (Score 72):**
> "Good squat overall. Your depth is coming along — improving from last rep! Work on your descent: lean back slightly and drop your hips lower. Watch your knee stability — keep your knees tracking over your toes."

---

## 9. Session-Level Feedback

After all reps, a session summary contains **two parts**:

### Part A — Aggregate Metrics (data-driven)
- Average score across all reps
- Most improved metric (highest positive delta: first rep → last rep of session)
- Most persistent issue (metric below threshold in the most reps)
- Session trajectory rule:
  - **Improving:** last rep score > first rep score by ≥5 points
  - **Declining:** last rep score < first rep score by ≥5 points
  - **Stable:** difference < 5 points either direction

### Part B — Coach Summary (template-driven)
- Overall session trajectory: improving / stable / declining
- Forward-looking coaching cue tied to the most persistent issue

**Example:**
> **Session Summary**
> *Your average score this session was solid. Most improved: depth control. Keep working on: forward lean.*
> "Good session! Your depth control came a long way. Next time, focus on keeping your torso more upright throughout each rep."

---

## 10. Exercise Config Schema (`exercises/squat.json`)

```json
{
  "exercise": "squat",
  "score_brackets": {
    "90-100": { "tier": "excellent", "opener": "Excellent squat form!" },
    "75-89":  { "tier": "good",      "opener": "Good squat overall." },
    "60-74":  { "tier": "fair",      "opener": "Decent effort, but a few things to address." },
    "40-59":  { "tier": "poor",      "opener": "Your form needs significant attention." },
    "0-39":   { "tier": "critical",  "opener": "Let's work on the fundamentals." }
  },
  "improvement_threshold": 75,
  "severity_band": 5,
  "issue_groups": {
    "descent_quality": {
      "metrics": ["forward_lean", "hip_depth"],
      "label": "descent quality",
      "single_cues": {
        "forward_lean": "Focus on keeping your torso more upright.",
        "hip_depth": "Drop your hips lower — aim for parallel or below."
      },
      "combined_cue": "Work on your descent: lean back slightly and drop your hips lower."
    },
    "knee_stability": {
      "metrics": ["knee_valgus", "knee_tracking"],
      "label": "knee stability",
      "single_cues": {
        "knee_valgus": "Keep your knees from caving inward.",
        "knee_tracking": "Keep your knees tracking directly over your toes."
      },
      "combined_cue": "Keep your knees tracking over your toes throughout the movement."
    }
  },
  "metrics": {
    "forward_lean":  { "good_threshold": 20,   "bad_threshold": 45,   "unit": "degrees" },
    "hip_depth":     { "good_threshold": 0.1,  "bad_threshold": -0.1, "unit": "normalized" },
    "knee_valgus":   { "good_threshold": 0.95, "bad_threshold": 0.75, "unit": "ratio" },
    "knee_tracking": { "good_threshold": 0.95, "bad_threshold": 0.75, "unit": "ratio" }
  }
}
```

---

## 11. Template Library Schema (`templates/feedback_templates.json`)

```json
{
  "improvement_phrases": {
    "significant": [
      "notably better than last rep",
      "much improved from last rep"
    ],
    "moderate": [
      "getting better",
      "improving from last rep"
    ],
    "slight": [
      "a little better than last rep"
    ]
  },
  "win_phrases": {
    "excellent_metric": [
      "Great [METRIC_LABEL] this rep!",
      "Solid [METRIC_LABEL] this rep!"
    ],
    "improving_metric": [
      "Your [METRIC_LABEL] is coming along — [IMPROVEMENT_PHRASE]!",
      "[METRIC_LABEL] improving — [IMPROVEMENT_PHRASE]!"
    ]
  },
  "session_summary": {
    "trajectory_openers": {
      "improving":  "Good session! You improved consistently.",
      "stable":     "Consistent session. Solid foundation.",
      "declining":  "Tough session — it happens."
    },
    "metric_summary": "Most improved this session: [TOP_IMPROVEMENT]. Keep working on: [PERSISTENT_ISSUE].",
    "coach_cue": "Next time, focus on [PERSISTENT_ISSUE_CUE]."
  }
}
```

---

## 12. FeedbackResult Output Contract

```python
@dataclass
class RepFeedback:
    rep_id: int
    score: float
    tier: str               # "excellent", "good", "fair", "poor", "critical"
    text: str               # Full feedback string
    wins: list[str]         # Metrics that scored well or improved
    issues: list[str]       # Issue groups surfaced (grouped labels)

@dataclass
class SessionSummary:
    avg_score: float
    most_improved_metric: str
    persistent_issue: str
    aggregate_text: str     # "Most improved: depth. Keep working on: forward lean."
    coach_text: str         # "Good session! Next time, focus on..."

@dataclass
class FeedbackResult:
    exercise: str
    reps: list[RepFeedback]
    session: SessionSummary
```

---

## 13. Integration Points

- **Input:** `neural_fusion_inference.py` output — `neural_score`, `bilstm_score`, `stgcn_score`, `heuristic_score`, `sub_scores`, `metrics` per rep
- **Output:** `FeedbackResult` returned alongside existing `AnalysisResult` in `apps/api/pipeline.py` `collect_results()`
- **Web app:** Renders `FeedbackResult` — per-rep text under each rep card, session summary at bottom of results page
- **Desktop UI:** Not yet wired (feedback is API-first for now)
- **Disclaimer:** Handled by web app UI globally — not part of feedback text

---

## 14. Extensibility

Adding a new exercise requires:
1. Create `core/exevision/config/exercises/deadlift.json` with exercise-specific brackets, groups, and metric thresholds
2. No code changes needed in `engine.py` — config-driven

Adding new templates:
1. Add entries to `feedback_templates.json`
2. No code changes needed in template_renderer.py if slot names match

---

## 15. Deterministic Phrase Selection

Template libraries contain multiple phrases per slot (e.g., 2 win phrases, 3 improvement phrases). To keep text **varied but repeatable** without generative AI:

**Rule:** Select phrase index using a hash of the composite key modulo the phrase count.

```python
import hashlib

def select_phrase(phrases: list[str], video_id: str, rep_id: int, metric_key: str) -> str:
    key = f"{video_id}:{rep_id}:{metric_key}"
    hash_int = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return phrases[hash_int % len(phrases)]
```

**Properties:**
- Same `video_id + rep_id + metric_key` always produces the same phrase (repeatable — no surprise on page refresh)
- Different videos, reps, or metrics produce different phrases (varied across context)
- No randomness, no generative AI, fully deterministic
- Applied in `template_renderer.py` for all phrase lists: `win_phrases`, `improvement_phrases`, `trajectory_openers`

---

## 16. Quality Checks & Fallback Mode

Handles **tone-content mismatch** when overall score and sub-metrics disagree.

### Case 1: Overall score low, all sub-metrics ≥ 75
- Neural fusion detected a quality issue the rule-based metrics didn't capture
- **Behaviour:** Use the overall score tier for tone opener (poor/critical), but suppress specific issue coaching cues since sub-metrics don't support them
- **Fallback text:** Append: *"The system detected concerns with the overall movement quality it couldn't pinpoint to a specific metric. Consider reviewing the full rep video."*
- **Signal that wins:** Overall score (trust the fusion)

### Case 2: Overall score ≥ 75, sub-metrics below threshold
- Rules detected specific faults the fusion didn't penalise heavily
- **Behaviour:** Use the overall score tier for tone opener (good/excellent), but still surface the sub-metric issues with **softened urgency**
- **Softened language:** Replace "Work on..." with "Something to keep in mind:..." for issues when overall score ≥ 75
- **Signal that wins:** Sub-metrics win for content; overall score wins for tone

### Case 3: Both agree (normal case)
- Overall score and sub-metrics consistently indicate same quality level
- Standard feedback flow applies; no fallback needed

### Mismatch Detection
```python
def detect_mismatch(overall_score: float, sub_scores: dict[str, float], threshold: float = 75) -> str:
    any_issue = any(v < threshold for v in sub_scores.values())
    if overall_score < threshold and not any_issue:
        return "low_overall_no_issues"
    if overall_score >= threshold and any_issue:
        return "high_overall_has_issues"
    return "normal"
```

---

## 17. Schema Versioning

All config files and the output payload carry a `schema_version` field to ensure web app and API compatibility as configs evolve.

### Exercise Config (`exercises/squat.json`)
```json
{
  "schema_version": "1.0",
  "exercise": "squat",
  ...
}
```

### Template Library (`templates/feedback_templates.json`)
```json
{
  "schema_version": "1.0",
  "improvement_phrases": { ... },
  ...
}
```

### FeedbackResult Payload
```python
@dataclass
class FeedbackResult:
    schema_version: str          # e.g., "1.0"
    exercise: str
    reps: list[RepFeedback]
    session: SessionSummary
```

### Versioning Rules
- **Patch (1.0 → 1.1):** New phrase variants added, no structural change — backwards compatible
- **Minor (1.0 → 1.1 → 2.0 breaking):** New required fields in config or output — bump to `2.0`
- Web app should check `schema_version` on `FeedbackResult` and display a graceful degradation message if it receives an unexpected version

---

## 18. Out of Scope

- Visual evidence GIFs (3-frame extraction) — separate concern, deferred
- Edge mode real-time feedback — deferred
- Cross-session progress tracking (across multiple sessions, not just within one) — future feature
- LLM/RAG-based feedback generation — explicitly excluded
