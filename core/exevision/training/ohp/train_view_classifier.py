"""Train the OHP neural view classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

_REPO = Path(__file__).resolve().parents[4]
_NEURAL = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from view_classifier import (
    ViewClassifierMLP,
    VIEW_LABELS,
    VIEW_TO_IDX,
    extract_frame_features,
)


def _load_dataset(annotation_dir: Path, features_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (X, y, vid_rows, video_labels)."""
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    vid_rows: list[str] = []
    video_labels: dict[str, int] = {}

    for anno_path in sorted(annotation_dir.glob("*.json")):
        try:
            anno = json.loads(anno_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_id = anno.get("video_id", anno_path.stem)
        view_label = None
        for rep in anno.get("reps", []):
            v = rep.get("annotated_view") or rep.get("view") or anno.get("view")
            if v and v in VIEW_TO_IDX:
                view_label = v
                break
        if view_label is None:
            view_label = anno.get("view", "")
        if view_label not in VIEW_TO_IDX:
            continue

        class_idx = VIEW_TO_IDX[view_label]
        video_labels[video_id] = class_idx

        feat_path = features_dir / f"{video_id}.json"
        if not feat_path.exists():
            continue
        try:
            feat_data = json.loads(feat_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        keypoints_img = feat_data.get("keypoints_img", [])
        face_detected_list = feat_data.get("face_detected", [])

        for i, frame in enumerate(keypoints_img):
            face_detected = face_detected_list[i] if i < len(face_detected_list) else False
            feat = extract_frame_features(frame, face_detected)
            if feat is not None:
                X_rows.append(feat)
                y_rows.append(class_idx)
                vid_rows.append(video_id)

    if not X_rows:
        raise RuntimeError("No training samples extracted. Check annotation/features paths.")

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64)
    vid_arr = np.array(vid_rows)

    print(f"Dataset: {X.shape[0]} frame samples from {len(set(vid_rows))} videos")
    for i, lbl in enumerate(VIEW_LABELS):
        count = int((y == i).sum())
        print(f"  {lbl.ljust(12)}: {count}")

    return X, y, vid_arr, video_labels


def _build_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.arange(len(VIEW_LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    label: str,
) -> tuple[ViewClassifierMLP, float | None]:
    model = ViewClassifierMLP().to(device)
    class_w = _build_class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = _build_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _build_loader(X_val, y_val, batch_size, shuffle=False) if X_val is not None else None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if ep % 30 == 0 or ep == epochs:
            denom = max(len(train_loader), 1)
            print(f"  [{label}] ep {ep:3d}/{epochs} loss={total_loss / denom:.4f}")

    if val_loader is None:
        return model, None

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy().tolist())

    acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if all_labels else 0.0
    return model, acc


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    vid_arr: np.ndarray,
    video_labels: dict,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> None:
    vids = np.array(sorted(video_labels.keys()))
    vid_y = np.array([video_labels[v] for v in vids])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(vids, vid_y)):
        tr_vids = set(vids[tr_idx])
        val_vids = set(vids[val_idx])
        mask_tr = np.isin(vid_arr, list(tr_vids))
        mask_val = np.isin(vid_arr, list(val_vids))

        _, acc = _train_model(
            X[mask_tr], y[mask_tr], X[mask_val], y[mask_val],
            epochs, lr, batch_size, device, label=f"fold{fold}",
        )
        acc = acc or 0.0
        accs.append(acc)
        print(f"  Fold {fold} val accuracy: {acc:.3f}")

    print(f"\n5-fold CV mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


def run_final(
    X: np.ndarray,
    y: np.ndarray,
    vid_arr: np.ndarray,
    video_labels: dict,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    output: Path,
) -> None:
    vids = np.array(sorted(video_labels.keys()))
    vid_y = np.array([video_labels[v] for v in vids])

    tr_vids, val_vids = train_test_split(
        vids, test_size=0.1, stratify=vid_y, random_state=42,
    )
    mask_tr = np.isin(vid_arr, tr_vids)
    mask_val = np.isin(vid_arr, val_vids)

    model, acc = _train_model(
        X[mask_tr], y[mask_tr], X[mask_val], y[mask_val],
        epochs, lr, batch_size, device, label="final",
    )
    acc = acc or 0.0
    print(f"\nFinal model held-out accuracy: {acc:.3f}")

    model_full, _ = _train_model(
        X, y, None, None, epochs, lr, batch_size, device, label="full",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_full.state_dict(), output)
    print(f"\nSaved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OHP view classifier MLP")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("training_dataset/ohp_phase3_annotations/videos"),
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("training_dataset/ohp_phase3_annotations/extracted_features"),
    )
    parser.add_argument("--output", type=Path, default=Path("models/view_classifier_ohp.pt"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run 5-fold CV only, skip final retrain",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, vid_arr, video_labels = _load_dataset(args.annotation_dir, args.features_dir)
    run_cv(X, y, vid_arr, video_labels, args.epochs, args.lr, args.batch_size, device)

    if not args.cv_only:
        run_final(X, y, vid_arr, video_labels, args.epochs, args.lr, args.batch_size, device, args.output)


if __name__ == "__main__":
    main()
