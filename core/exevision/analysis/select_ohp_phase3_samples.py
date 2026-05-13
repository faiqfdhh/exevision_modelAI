import argparse
import json
import random
from pathlib import Path

STRATA = [
    ("calibration", lambda r: r["heuristic"] < 40 or r["heuristic"] > 85, 25),
    ("uncertainty", lambda r: 40 <= r["heuristic"] <= 70, 125),
    ("error",       lambda r: r.get("knee_error_prob", 0) > 0.5 or r.get("knee_error_prob", 1) < 0.2, 63),
    ("view_balance", None, 38),   # fill remainder capped per-view at 30% of final pool
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-dir", type=Path, required=True)
    parser.add_argument("--aqa-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=250)
    args = parser.parse_args()

    # Read heuristic scores
    reps = []
    aqa_files = list(args.aqa_dir.rglob("*.json"))
    for f in aqa_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for rep in data.get("reps", []):
                rep_id = rep.get("rep_id")
                score = rep.get("score")
                if score is not None:
                    reps.append({
                        "video_id": data.get("video_id"),
                        "rep_id": rep_id,
                        "heuristic": score,
                        "view": data.get("view", "unknown")
                    })
        except Exception:
            continue

    # Read neural scores
    neural_files = list(args.neural_dir.rglob("*.json"))
    neural_map = {}
    for f in neural_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            v_id = data.get("video_id")
            for rep in data.get("reps", []):
                r_id = rep.get("rep_id")
                neural_map[(v_id, r_id)] = rep.get("knee_error_prob")
        except Exception:
            continue

    # Merge
    for r in reps:
        key = (r["video_id"], r["rep_id"])
        if key in neural_map and neural_map[key] is not None:
            r["knee_error_prob"] = neural_map[key]

    random.seed(42)
    random.shuffle(reps)

    selected = []
    used_keys = set()
    
    for stratum_name, condition, count in STRATA:
        if condition is None:
            # view_balance
            view_counts = {}
            added = 0
            for r in reps:
                if added >= count:
                    break
                key = (r["video_id"], r["rep_id"])
                if key in used_keys:
                    continue
                v = r["view"]
                if view_counts.get(v, 0) >= (args.n * 0.3):
                    continue
                view_counts[v] = view_counts.get(v, 0) + 1
                r["stratum"] = stratum_name
                r["heuristic_score"] = r.pop("heuristic")
                selected.append(r)
                used_keys.add(key)
                added += 1
        else:
            added = 0
            for r in reps:
                if added >= count:
                    break
                key = (r["video_id"], r["rep_id"])
                if key in used_keys:
                    continue
                if condition(r):
                    r["stratum"] = stratum_name
                    r["heuristic_score"] = r.pop("heuristic")
                    selected.append(r)
                    used_keys.add(key)
                    added += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} reps. Output to {args.output}")

if __name__ == "__main__":
    main()
