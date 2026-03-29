# Metric-Agnostic Tier Language — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make feedback narratives mention ALL metric scores using unified tier language (excellent/strong/okay for ≥75; needs_work/focus_here for <75), so every metric appears in the coaching text with tone appropriate to its score.

**Architecture:** Three config-only changes (templates JSON + exercise JSON) plus three engine.py logic changes (tier-aware win phrases, severity-aware issue cues, new "stable metric" mentions for ≥75 metrics that didn't improve). No new files created. Template renderer unchanged.

**Tech Stack:** Python 3.10+, JSON config files

**Key insight — current gap:** The engine currently only mentions two categories:
1. **Wins** — metrics ≥75 that improved vs previous rep
2. **Issues** — metrics <75

Metrics ≥75 that did NOT improve (or rep 1 with no comparison) are **silently skipped**. This plan adds a third category: **stable mentions** — brief tier-appropriate notes like "Your depth is okay" or "Strong knee tracking."

**Metric-level tier mapping** (based on individual metric score, not overall rep score):

| Metric Score | Phrase Tier | Language Style |
|---|---|---|
| 90–100 | `excellent` | "Excellent [metric]!", "Outstanding [metric]!" |
| 85–89 | `strong` | "Strong [metric].", "Solid [metric]." |
| 75–84 | `okay` | "Your [metric] is okay.", "[metric] is acceptable." |
| 60–74 | `needs_work` | "Work on [metric].", directive coaching cues |
| 0–59 | `focus_here` | "Let's address [metric].", urgent coaching cues |

---

## File Map

| File | Action | What Changes |
|---|---|---|
| `core/exevision/config/templates/feedback_templates.json` | Modify | Add tier-based `win_phrases`, `stable_phrases`, expand `improvement_phrases` |
| `core/exevision/config/exercises/squat.json` | Modify | Change `single_cues` and `combined_cue` from flat strings to `{needs_work, focus_here}` dicts |
| `core/exevision/feedback/engine.py` | Modify | Tier-aware win selection, severity-aware issue cues, new stable-metric mentions, fix `_resolve_issue_cue_text` for dict cues |

---

### Task 1: Update `feedback_templates.json`

**File:** `core/exevision/config/templates/feedback_templates.json`

The current file has 39 lines. Replace the entire contents with the expanded version below.

- [ ] **Step 1: Replace full contents of `feedback_templates.json`**

```json
{
  "schema_version": "1.0",
  "improvement_phrases": {
    "significant": [
      "notably better than last rep",
      "much improved from last rep",
      "a major step up from before"
    ],
    "moderate": [
      "getting better",
      "improving from last rep",
      "showing steady progress"
    ],
    "slight": [
      "a little better than last rep",
      "slightly improved from before",
      "moving in the right direction"
    ]
  },
  "win_phrases": {
    "improving_metric_excellent": [
      "Excellent [METRIC_LABEL] — [IMPROVEMENT_PHRASE]!",
      "Outstanding [METRIC_LABEL] — [IMPROVEMENT_PHRASE]!",
      "[METRIC_LABEL] is superb — [IMPROVEMENT_PHRASE]!"
    ],
    "improving_metric_strong": [
      "Strong [METRIC_LABEL] — [IMPROVEMENT_PHRASE]!",
      "Solid [METRIC_LABEL] — [IMPROVEMENT_PHRASE]!",
      "[METRIC_LABEL] is impressive — [IMPROVEMENT_PHRASE]!"
    ],
    "improving_metric_okay": [
      "Your [METRIC_LABEL] is okay and [IMPROVEMENT_PHRASE].",
      "[METRIC_LABEL] is acceptable — [IMPROVEMENT_PHRASE].",
      "[METRIC_LABEL] is reasonable and [IMPROVEMENT_PHRASE]."
    ]
  },
  "stable_phrases": {
    "excellent": [
      "Excellent [METRIC_LABEL]!",
      "Outstanding [METRIC_LABEL]!",
      "[METRIC_LABEL] is superb!"
    ],
    "strong": [
      "Strong [METRIC_LABEL].",
      "Solid [METRIC_LABEL].",
      "[METRIC_LABEL] is looking good."
    ],
    "okay": [
      "Your [METRIC_LABEL] is okay.",
      "[METRIC_LABEL] is acceptable.",
      "[METRIC_LABEL] is reasonable."
    ]
  },
  "issue_templates": {
    "single_issue": "[ISSUE_CUE]",
    "combined_issue": "[COMBINED_CUE]"
  },
  "session_summary": {
    "trajectory_openers": {
      "improving": "Good session! You improved consistently.",
      "stable": "Consistent session. Solid foundation.",
      "declining": "Tough session — it happens."
    },
    "metric_summary": "Most improved this session: [TOP_IMPROVEMENT]. Keep working on: [PERSISTENT_ISSUE].",
    "coach_cue": "Next time, focus on [PERSISTENT_ISSUE_CUE]."
  }
}
```

**What changed vs current:**
- `win_phrases`: removed flat `"excellent_metric"` and `"improving_metric"` keys. Added three tier-keyed variants: `improving_metric_excellent`, `improving_metric_strong`, `improving_metric_okay`. Each has 3 phrase options.
- `stable_phrases`: **entirely new section**. Three tiers (`excellent`, `strong`, `okay`) each with 3 phrases. These are for metrics ≥75 that did NOT improve or have no comparison (rep 1).
- `improvement_phrases`: added one more variant per tier for slightly more variety.
- `session_summary.trajectory_openers.declining`: changed hyphen to em dash for consistency.
- Everything else unchanged.

---

### Task 2: Update `squat.json` Issue Cues

**File:** `core/exevision/config/exercises/squat.json`

Change `single_cues` values and `combined_cue` values from flat strings to `{needs_work, focus_here}` dicts. The rest of the file stays identical.

- [ ] **Step 1: Replace the `issue_groups` section in `squat.json`**

Find the current `issue_groups` block (lines 28–47) and replace with:

```json
  "issue_groups": {
    "descent_quality": {
      "metrics": ["forward_lean", "hip_depth"],
      "label": "descent quality",
      "single_cues": {
        "forward_lean": {
          "needs_work": "Work on keeping your torso more upright.",
          "focus_here": "Your forward lean needs serious attention — stay upright throughout the movement."
        },
        "hip_depth": {
          "needs_work": "Work on getting lower — aim for parallel or below.",
          "focus_here": "Hip depth is insufficient — you need to go significantly deeper."
        }
      },
      "combined_cue": {
        "needs_work": "Work on your descent: lean back slightly and drop your hips lower.",
        "focus_here": "Your descent needs significant work — stay upright and drop much deeper."
      }
    },
    "knee_stability": {
      "metrics": ["knee_valgus", "knee_tracking"],
      "label": "knee stability",
      "single_cues": {
        "knee_valgus": {
          "needs_work": "Work on keeping your knees from caving inward.",
          "focus_here": "Knee valgus is critical — your knees are collapsing inward and need immediate attention."
        },
        "knee_tracking": {
          "needs_work": "Work on keeping your knees tracking over your toes.",
          "focus_here": "Knee tracking needs significant work — they're drifting off alignment."
        }
      },
      "combined_cue": {
        "needs_work": "Work on knee stability: keep your knees tracking over your toes throughout.",
        "focus_here": "Knee stability is a priority — both alignment and inward collapse need addressing."
      }
    }
  },
```

**What changed:**
- `single_cues.forward_lean`: was `"Focus on keeping your torso more upright."` → now `{needs_work: "...", focus_here: "..."}`
- `single_cues.hip_depth`: was `"Drop your hips lower..."` → now dict with two severity tiers
- `single_cues.knee_valgus`: was `"Keep your knees from caving inward."` → now dict
- `single_cues.knee_tracking`: was `"Keep your knees tracking..."` → now dict
- `combined_cue` for both groups: was flat string → now dict with `needs_work` and `focus_here`
- `metrics`, `label` fields unchanged
- Everything outside `issue_groups` (score_brackets, metrics, etc.) unchanged

---

### Task 3: Update `engine.py` — Tier-Aware Win Phrases

**File:** `core/exevision/feedback/engine.py`

Three changes in this task: (A) tier-aware win phrase selection in `_build_win_texts`, (B) new `_build_stable_texts` method, (C) wire stable texts into `_build_rep_feedback`.

- [ ] **Step 1: Add `_metric_phrase_tier` helper method**

Add this new method after `_get_wins` (after line 181):

```python
    @staticmethod
    def _metric_phrase_tier(score: float) -> str:
        """Map an individual metric score to a phrase tier key."""
        if score >= 90:
            return "excellent"
        if score >= 85:
            return "strong"
        return "okay"
```

- [ ] **Step 2: Update `_build_win_texts` to use tier-based win phrase key**

In `_build_win_texts`, replace lines 214–216:

```python
            win_templates = win_catalog.get("improving_metric", [])
            if not win_templates:
                win_templates = ["Your [METRIC_LABEL] is improving — [IMPROVEMENT_PHRASE]!"]
```

With:

```python
            phrase_tier = self._metric_phrase_tier(sub_scores.get(metric_key, 75.0))
            win_key = f"improving_metric_{phrase_tier}"
            win_templates = win_catalog.get(win_key, win_catalog.get("improving_metric_okay", []))
            if not win_templates:
                win_templates = ["Your [METRIC_LABEL] is improving — [IMPROVEMENT_PHRASE]!"]
```

This requires `sub_scores` to be passed into `_build_win_texts`. Update the method signature (line 183–188) from:

```python
    def _build_win_texts(
        self,
        video_id: str,
        rep_id: int,
        wins: list[str],
        comparison: dict[str, Any] | None,
    ) -> list[str]:
```

To:

```python
    def _build_win_texts(
        self,
        video_id: str,
        rep_id: int,
        wins: list[str],
        sub_scores: dict[str, float],
        comparison: dict[str, Any] | None,
    ) -> list[str]:
```

- [ ] **Step 3: Add `_build_stable_texts` method**

Add this new method after `_build_win_texts` (after line 228):

```python
    def _build_stable_texts(
        self,
        video_id: str,
        rep_id: int,
        sub_scores: dict[str, float],
        wins: list[str],
        issue_keys: list[str],
        threshold: float,
    ) -> list[str]:
        """Generate brief tier-appropriate mentions for metrics ≥ threshold that aren't wins."""
        stable_catalog = self.templates.get("stable_phrases", {})
        if not stable_catalog:
            return []

        texts: list[str] = []
        for metric_key, score in sub_scores.items():
            if score < threshold:
                continue
            if metric_key in wins:
                continue

            phrase_tier = self._metric_phrase_tier(score)
            phrases = stable_catalog.get(phrase_tier, [])
            if not phrases:
                continue

            template = self._renderer.select_phrase(
                phrases, video_id, rep_id, f"{metric_key}_stable"
            )
            text = self._renderer.fill_slots(
                template,
                {"METRIC_LABEL": self._renderer.humanize_metric(metric_key)},
            )
            texts.append(text)

        return texts
```

- [ ] **Step 4: Wire both changes into `_build_rep_feedback`**

In `_build_rep_feedback` (line 278), update the method signature to also accept `sub_scores` and `threshold`:

```python
    def _build_rep_feedback(
        self,
        video_id: str,
        rep_id: int,
        tier: str,
        wins: list[str],
        sub_scores: dict[str, float],
        issue_scores: dict[str, float],
        mismatch_type: str,
        comparison: dict[str, Any] | None,
        threshold: float,
    ) -> str:
```

Update the body (lines 288–308). Replace:

```python
        win_texts = self._build_win_texts(video_id, rep_id, wins, comparison)
        parts.extend(win_texts)

        soften_issues = mismatch_type == "high_overall_has_issues"
        issue_cues = self._group_issue_cues(issue_scores, soften=soften_issues)
        parts.extend(issue_cues)
```

With:

```python
        win_texts = self._build_win_texts(video_id, rep_id, wins, sub_scores, comparison)
        parts.extend(win_texts)

        stable_texts = self._build_stable_texts(
            video_id, rep_id, sub_scores, wins, list(issue_scores.keys()), threshold,
        )
        parts.extend(stable_texts)

        soften_issues = mismatch_type == "high_overall_has_issues"
        issue_cues = self._group_issue_cues(issue_scores, soften=soften_issues)
        parts.extend(issue_cues)
```

- [ ] **Step 5: Update the call site in `generate_feedback`**

In `generate_feedback` (lines 108–116), update the call to `_build_rep_feedback` to pass the new args:

Replace:

```python
            rep_text = self._build_rep_feedback(
                video_id=video_id,
                rep_id=rep_id,
                tier=tier,
                wins=wins,
                issue_scores=issue_scores,
                mismatch_type=mismatch_type,
                comparison=comparison,
            )
```

With:

```python
            rep_text = self._build_rep_feedback(
                video_id=video_id,
                rep_id=rep_id,
                tier=tier,
                wins=wins,
                sub_scores=sub_scores,
                issue_scores=issue_scores,
                mismatch_type=mismatch_type,
                comparison=comparison,
                threshold=threshold,
            )
```

---

### Task 4: Update `engine.py` — Severity-Aware Issue Cues

**File:** `core/exevision/feedback/engine.py`

Two changes: (A) `_group_issue_cues` reads tier-aware dict cues, (B) `_resolve_issue_cue_text` handles dict cues.

- [ ] **Step 1: Update `_group_issue_cues` cue selection (lines 246–251)**

Replace:

```python
            severity = min(issue_scores[metric] for metric in low_metrics)
            if len(low_metrics) > 1:
                cue = str(group_info.get("combined_cue", "")).strip()
            else:
                metric = low_metrics[0]
                cue = str(group_info.get("single_cues", {}).get(metric, "")).strip()
```

With:

```python
            severity = min(issue_scores[metric] for metric in low_metrics)
            cue_tier = "focus_here" if severity < 60 else "needs_work"

            if len(low_metrics) > 1:
                combined = group_info.get("combined_cue", "")
                if isinstance(combined, dict):
                    cue = str(combined.get(cue_tier, combined.get("needs_work", ""))).strip()
                else:
                    cue = str(combined).strip()
            else:
                metric = low_metrics[0]
                single = group_info.get("single_cues", {}).get(metric, "")
                if isinstance(single, dict):
                    cue = str(single.get(cue_tier, single.get("needs_work", ""))).strip()
                else:
                    cue = str(single).strip()
```

**Why `isinstance` checks:** Backwards-compatible. If someone hasn't migrated their exercise config to dict-based cues yet, the old flat string still works.

- [ ] **Step 2: Update fallback cue generation (line 260)**

Replace:

```python
            entries.append((f"Focus on {self._renderer.humanize_metric(metric)}.", score))
```

With:

```python
            label = self._renderer.humanize_metric(metric)
            if score < 60:
                entries.append((f"Let's address {label} — this needs significant work.", score))
            else:
                entries.append((f"Work on {label} — there's room for improvement.", score))
```

- [ ] **Step 3: Update `_resolve_issue_cue_text` to handle dict cues (line 364–373)**

Replace:

```python
    def _resolve_issue_cue_text(self, metric_key: str) -> str:
        if not metric_key:
            return "overall consistency"

        issue_groups = self.exercise_config.get("issue_groups", {})
        for _, group_info in issue_groups.items():
            if metric_key in group_info.get("metrics", []):
                return str(group_info.get("single_cues", {}).get(metric_key, self._renderer.humanize_metric(metric_key))).rstrip(".")

        return self._renderer.humanize_metric(metric_key)
```

With:

```python
    def _resolve_issue_cue_text(self, metric_key: str) -> str:
        if not metric_key:
            return "overall consistency"

        issue_groups = self.exercise_config.get("issue_groups", {})
        for _, group_info in issue_groups.items():
            if metric_key in group_info.get("metrics", []):
                cue = group_info.get("single_cues", {}).get(metric_key, "")
                if isinstance(cue, dict):
                    cue = cue.get("needs_work", "")
                if cue:
                    return str(cue).rstrip(".")
                return self._renderer.humanize_metric(metric_key)

        return self._renderer.humanize_metric(metric_key)
```

**Why `needs_work` default for session coach cue:** The session coach cue is forward-looking advice ("next time, focus on..."). The `needs_work` tier language is coaching-oriented by nature and fits this slot regardless of how severe the persistent issue was.

---

## Summary of All Changes

| File | Lines Changed | What |
|---|---|---|
| `feedback_templates.json` | Full rewrite (39→62 lines) | Added `stable_phrases` section; split `win_phrases` into 3 tier keys; expanded `improvement_phrases` |
| `squat.json` | Lines 28–47 | `single_cues` and `combined_cue` → `{needs_work, focus_here}` dicts |
| `engine.py` | ~8 locations | New `_metric_phrase_tier()`, new `_build_stable_texts()`, tier-aware win selection, severity-aware issue cues, dict-safe `_resolve_issue_cue_text` |

**Narrative structure after changes (per rep):**

```
[TONE_OPENER]                         ← from score bracket (unchanged)
[WIN_1] [WIN_2]                       ← tier-aware: "Excellent depth!" or "Solid depth!"
[STABLE_1] [STABLE_2]                 ← NEW: "Your knee tracking is okay." for ≥75, non-improving
[ISSUE_GROUP_1]                       ← severity-aware: "Work on..." or "Let's address..."
[ISSUE_GROUP_2]
```

**Example output — Rep 2, Score 78:**
> "Good squat overall. Strong depth — getting better! Your knee tracking is okay. Work on keeping your torso more upright."

**Example output — Rep 3, Score 48:**
> "Your form needs significant attention. Your descent needs significant work — stay upright and drop much deeper. Knee valgus is critical — your knees are collapsing inward and need immediate attention."
