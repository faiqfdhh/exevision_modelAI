# ExeVision Web App — Feedback Narrative Integration

> Pass this to the web app agent. Integrate new `feedback` payload with tier-language coaching narratives.

---

## What Changed in the API

The API now returns a `feedback` object alongside `result` in the job response:

```json
{
  "feedback": {
    "schema_version": "1.0",
    "exercise": "squat",
    "reps": [
      {
        "rep_id": 1,
        "score": 78.0,
        "tier": "good",
        "text": "Good squat overall. Strong depth - improving from last rep! Your knee tracking is okay. Work on keeping your torso more upright.",
        "wins": ["depth"],
        "issues": ["forward_lean"]
      }
    ],
    "session": {
      "avg_score": 75.2,
      "trajectory": "improving",
      "most_improved_metric": "depth",
      "persistent_issue": "forward_lean",
      "aggregate_text": "Most improved this session: depth. Keep working on: forward lean.",
      "coach_text": "Good session! You improved consistently. Next time, focus on Work on keeping your torso more upright."
    }
  }
}
```

## UI Changes Required

**Replace the rep results card layout.** Old layout (remove):
```
Sub-scores (bars): Smoothness ██████░░ 80, Control ███████░░ 70, etc.
Biomechanical Metrics: (table of exact values)
Safety caps: (warning banner)
```

**New layout (add):**
```
Rep N - Neural: 78/100  Heuristic: 74/100

[NARRATIVE TEXT FROM feedback.reps[N].text]

Expand "Judges & Metrics" to see:
  - BiLSTM: 80, ST-GCN: 70, Heuristic: 74
  - (existing metrics table, now collapsible)
```

## Required Changes

1. **Rep card component:** Display `feedback.reps[N].text` as primary coaching content (large, prominent, readable)
2. **Session summary:** Show `feedback.session.coach_text` at bottom of results page in a coached section
3. **Judges panel:** Keep three-judge scores (BiLSTM/ST-GCN/Heuristic) but move to collapsible "Details" section
4. **Metrics table:** Move to same collapsible section (users can expand if they want raw angles/ratios)
5. **No changes needed:** Overall score display, rep selector tabs, view/quality badges all remain

## Example Rep Narrative Display

```
Rep 2 — Neural: 78/100  Heuristic: 74/100

Good squat overall. Strong depth - improving from last rep! Your knee
tracking is okay. Work on keeping your torso more upright.

[Expand Details ▼]
  BiLSTM: 80 | ST-GCN: 70 | Heuristic: 74
  Metrics: forward_lean 22.5°, knee_valgus 0.94, ...
```

## Fallback

If `feedback` is null or missing:
- Show heuristic score only with note: "Coaching feedback unavailable (neural pipeline skipped)"
- Display existing metrics table as fallback

## Files to Update

- `app/analyze/results/[jobId]/page.tsx` — results layout
- `components/analysis/RepCard.tsx` (or equivalent) — rep display component
- `types/analysis.ts` — add `FeedbackResult` type if missing

No API changes needed. No backend changes needed.
