"""Phase 3 OHP fine-tuning: multi-task heads on manually annotated reps.

Progressive unfreezing:
  Stage 1 (20 epochs): encoder frozen, only heads + fusion trained.
  Stage 2 (60 epochs): encoder unfrozen at differential LRs, SGDR schedule, SWA from epoch 51.

CLI:
  python finetune_phase3.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --pretrain-bilstm models/bilstm_ohp_phase2.pt \\
      --pretrain-stgcn  models/stgcn_ohp_phase2.pt \\
      --output-dir      models/ \\
      [--final]   # retrain on full dataset with 5 seeds instead of CV
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR

_REPO      = Path(__file__).resolve().parents[4]
_NEURAL    = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL = _NEURAL / "ohp"
_TRAIN_OHP = Path(__file__).resolve().parent
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer
from ohp.fusion import build_ohp_fusion
from data_phase3 import OHPPhase3Dataset, build_phase3_dataloaders
from losses import compute_phase3_loss

# ── Hyper-parameters (match plan exactly) ──────────────────────────────────────
STAGE1_EPOCHS = 20      # freeze encoder, train heads only
STAGE2_EPOCHS = 60      # unfreeze encoder at low LR
TOTAL_EPOCHS  = STAGE1_EPOCHS + STAGE2_EPOCHS   # = 80

STAGE1_HEAD_LR          = 5e-4
STAGE2_HEAD_LR          = 3e-4
STAGE2_ENCODER_LR_LAST  = 5e-5   # last LSTM/block layers
STAGE2_ENCODER_LR_EARLY = 1e-5   # early LSTM/block layers

SWA_START_STAGE2_EPOCH = 51   # Stage 2 epoch 51 = global epoch 71
BATCH_SIZE = 16
FINAL_SEEDS = [42, 7, 19, 3, 99]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_optimizer_stage1(bilstm, stgcn, fusion):
    """Stage 1: encoder frozen. Only heads + fusion trained."""
    params = (
        list(bilstm.quality_head.parameters()) +
        list(bilstm.smoothness_head.parameters()) +
        list(bilstm.control_head.parameters()) +
        list(stgcn.quality_head.parameters()) +
        list(stgcn.lockout_head.parameters()) +
        list(stgcn.elbow_flare_head.parameters()) +
        list(stgcn.grip_ratio_head.parameters()) +
        list(stgcn.rom_top_head.parameters()) +
        list(stgcn.rom_bottom_head.parameters()) +
        list(fusion.parameters())
    )
    # knee_error_head NOT in optimizer (frozen, requires_grad=False)
    return torch.optim.Adam(params, lr=STAGE1_HEAD_LR, weight_decay=1e-3)


def _make_optimizer_stage2(bilstm, stgcn, fusion):
    """Stage 2: encoder unfrozen at low LR. Heads at higher LR. Knee still frozen."""
    param_groups = [
        {"params": list(bilstm.lstm1.parameters()) + list(stgcn.block1.parameters()) +
                   list(stgcn.block2.parameters()),
         "lr": STAGE2_ENCODER_LR_EARLY},
        {"params": list(bilstm.lstm2.parameters()) + list(bilstm.temporal_attention.parameters()) +
                   list(stgcn.block3.parameters()) + list(stgcn.block4.parameters()) +
                   list(stgcn.block5.parameters()),
         "lr": STAGE2_ENCODER_LR_LAST},
        {"params": (
            list(bilstm.quality_head.parameters()) +
            list(bilstm.smoothness_head.parameters()) +
            list(bilstm.control_head.parameters()) +
            list(stgcn.quality_head.parameters()) +
            list(stgcn.lockout_head.parameters()) +
            list(stgcn.elbow_flare_head.parameters()) +
            list(stgcn.grip_ratio_head.parameters()) +
            list(stgcn.rom_top_head.parameters()) +
            list(stgcn.rom_bottom_head.parameters()) +
            list(fusion.parameters())
         ), "lr": STAGE2_HEAD_LR},
    ]
    return torch.optim.Adam(param_groups, weight_decay=1e-3)


def _freeze_encoder(bilstm, stgcn) -> None:
    for m in [bilstm.lstm1, bilstm.lstm2, bilstm.temporal_attention,
              stgcn.block1, stgcn.block2, stgcn.block3, stgcn.block4, stgcn.block5]:
        for p in m.parameters():
            p.requires_grad = False


def _unfreeze_encoder(bilstm, stgcn) -> None:
    for m in [bilstm.lstm1, bilstm.lstm2, bilstm.temporal_attention,
              stgcn.block1, stgcn.block2, stgcn.block3, stgcn.block4, stgcn.block5]:
        for p in m.parameters():
            p.requires_grad = True


def _run_epoch(bilstm, stgcn, fusion, loader, optimizer, device, train: bool) -> float:
    bilstm.train(train)
    stgcn.train(train)
    fusion.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            # Move all tensor values to device; keep str 'view' on CPU
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            bilstm_out = bilstm(batch_dev["bilstm_input"])
            stgcn_out  = stgcn(batch_dev["stgcn_input"], batch_dev["view_vec"])
            fusion_score, _ = fusion(
                batch_dev["heuristic_vec"],
                stgcn_out["embedding"],
                bilstm_out["embedding"],
            )
            loss = compute_phase3_loss(bilstm_out, stgcn_out, fusion_score, batch_dev)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def _compute_val_metrics(bilstm, stgcn, fusion, loader, device) -> tuple[float, float]:
    """Return (val_loss, lockout_auc) over the full validation set."""
    bilstm.eval(); stgcn.eval(); fusion.eval()
    total_loss = 0.0
    all_lockout_preds: list[float] = []
    all_lockout_labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            bilstm_out = bilstm(batch_dev["bilstm_input"])
            stgcn_out  = stgcn(batch_dev["stgcn_input"], batch_dev["view_vec"])
            fusion_score, _ = fusion(
                batch_dev["heuristic_vec"],
                stgcn_out["embedding"],
                bilstm_out["embedding"],
            )
            total_loss += compute_phase3_loss(bilstm_out, stgcn_out, fusion_score, batch_dev).item()
            all_lockout_preds.extend(stgcn_out["lockout"].cpu().numpy().tolist())
            all_lockout_labels.extend(batch_dev["lockout"].cpu().numpy().tolist())

    val_loss = total_loss / max(len(loader), 1)
    arr_l = np.array(all_lockout_labels)
    arr_p = np.array(all_lockout_preds)
    lockout_auc = (
        float(roc_auc_score(arr_l, arr_p))
        if arr_l.sum() > 0 and arr_l.sum() < len(arr_l)
        else 0.5
    )
    return val_loss, lockout_auc


def _train_one_fold(
    train_paths: list[Path],
    val_paths: list[Path],
    pretrain_bilstm: Path,
    pretrain_stgcn: Path,
    output_dir: Path,
    checkpoint_suffix: str,
    seed: int = 42,
) -> None:
    """Train one fold / one seed run with the full stage1 → stage2 schedule."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn  = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)

    # Load Phase 2 weights; freeze knee head
    bilstm.load_phase2_for_phase3(str(pretrain_bilstm))
    stgcn.load_phase2_for_phase3(str(pretrain_stgcn))

    # Build data loaders
    train_ds = OHPPhase3Dataset(train_paths)
    val_ds   = OHPPhase3Dataset(val_paths) if val_paths else None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = (
        torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        if val_ds and len(val_ds) > 0 else None
    )

    # ── Stage 1: freeze encoder ─────────────────────────────────────────────
    _freeze_encoder(bilstm, stgcn)
    opt1 = _make_optimizer_stage1(bilstm, stgcn, fusion)

    print(f"[{checkpoint_suffix}] Stage 1 — encoder frozen ({STAGE1_EPOCHS} epochs)")
    for ep in range(1, STAGE1_EPOCHS + 1):
        train_loss = _run_epoch(bilstm, stgcn, fusion, train_loader, opt1, device, train=True)
        print(f"  ep {ep:2d}/{STAGE1_EPOCHS} | train_loss={train_loss:.4f}")

    # ── Stage 2: unfreeze encoder at differential LRs, SGDR, SWA ───────────
    _unfreeze_encoder(bilstm, stgcn)
    opt2 = _make_optimizer_stage2(bilstm, stgcn, fusion)
    scheduler = CosineAnnealingWarmRestarts(opt2, T_0=20, T_mult=1, eta_min=1e-6)

    swa_bilstm = AveragedModel(bilstm)
    swa_stgcn  = AveragedModel(stgcn)

    best_val_loss = float("inf")
    best_lockout_auc = 0.0

    print(f"[{checkpoint_suffix}] Stage 2 — encoder unfrozen ({STAGE2_EPOCHS} epochs, SGDR+SWA)")
    for s2_ep in range(1, STAGE2_EPOCHS + 1):
        train_loss = _run_epoch(bilstm, stgcn, fusion, train_loader, opt2, device, train=True)
        scheduler.step()

        # SWA: start averaging from stage2 epoch 51 (= global epoch 71)
        if s2_ep >= SWA_START_STAGE2_EPOCH:
            swa_bilstm.update_parameters(bilstm)
            swa_stgcn.update_parameters(stgcn)

        val_loss = float("inf")
        lockout_auc = 0.0
        if val_loader is not None:
            val_loss, lockout_auc = _compute_val_metrics(bilstm, stgcn, fusion, val_loader, device)

        global_ep = STAGE1_EPOCHS + s2_ep
        if val_loader is not None:
            print(
                f"  ep {global_ep:2d}/{TOTAL_EPOCHS} | s2_ep {s2_ep:2d} "
                f"| train={train_loss:.4f} | val_loss={val_loss:.4f} | lockout_auc={lockout_auc:.4f}"
            )
        else:
            print(
                f"  ep {global_ep:2d}/{TOTAL_EPOCHS} | s2_ep {s2_ep:2d} "
                f"| train={train_loss:.4f} | val=N/A (full-data run)"
            )

        # Save best on val_loss; no val set → save only at last Stage 2 epoch (SWA overwrites at end)
        improved = (val_loss < best_val_loss) if val_loader else (s2_ep == STAGE2_EPOCHS)
        if improved:
            best_val_loss    = val_loss
            best_lockout_auc = lockout_auc
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bilstm.state_dict(), output_dir / f"bilstm_{checkpoint_suffix}.pt")
            torch.save(stgcn.state_dict(),  output_dir / f"stgcn_{checkpoint_suffix}.pt")
            torch.save(fusion.state_dict(), output_dir / f"fusion_{checkpoint_suffix}.pt")
            ckpt_note = f"val_loss={best_val_loss:.4f} lockout_auc={best_lockout_auc:.4f}" if val_loader else "full-data epoch"
            print(f"  → checkpoint saved ({ckpt_note})")

    if val_loader is not None:
        print(f"[{checkpoint_suffix}] Done. Best val_loss={best_val_loss:.4f} lockout_auc={best_lockout_auc:.4f}")
    else:
        print(f"[{checkpoint_suffix}] Done. (full-data run — no val metrics)")

    # ── Finalise SWA: update batch norm stats, then save averaged weights ────
    swa_updated = (STAGE2_EPOCHS >= SWA_START_STAGE2_EPOCH)
    if swa_updated and len(train_loader) > 0:
        print(f"[{checkpoint_suffix}] Updating SWA batch norm stats...")
        # update_bn expects tensors; run forward manually since loader yields dicts
        swa_bilstm.train()
        swa_stgcn.train()
        with torch.no_grad():
            for batch in train_loader:
                bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
                swa_bilstm(bd["bilstm_input"])
                swa_stgcn(bd["stgcn_input"], bd["view_vec"])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(swa_bilstm.module.state_dict(), output_dir / f"bilstm_{checkpoint_suffix}.pt")
        torch.save(swa_stgcn.module.state_dict(),  output_dir / f"stgcn_{checkpoint_suffix}.pt")
        print(f"[{checkpoint_suffix}] SWA weights saved.")
    else:
        print(f"[{checkpoint_suffix}] SWA not triggered — best-epoch checkpoint retained.")


def run_cv(
    annotation_dir: Path,
    pretrain_bilstm: Path,
    pretrain_stgcn: Path,
    output_dir: Path,
) -> None:
    """5-fold stratified CV. Stratify key: view + lockout label."""
    all_paths = sorted(Path(annotation_dir).glob("*.json"))
    if not all_paths:
        raise RuntimeError(f"No annotation JSONs found in {annotation_dir}")

    # Build a flat list of (path, stratify_key) for all annotated reps
    import json as _json
    strat_keys = []
    path_per_rep = []
    for p in all_paths:
        try:
            anno = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        view = anno.get("view", "unknown")
        for rep in anno.get("reps") or []:
            if rep.get("human_score") is None:
                continue
            lockout = int(bool(rep.get("human_flags", {}).get("lockout", True)))
            strat_keys.append(f"{view}_{lockout}")
            path_per_rep.append(p)

    if not strat_keys:
        raise RuntimeError("No annotated reps found for CV")

    paths_arr = np.array(path_per_rep)
    keys_arr  = np.array(strat_keys)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k, (train_idx, val_idx) in enumerate(skf.split(paths_arr, keys_arr)):
        train_paths = list(dict.fromkeys(paths_arr[train_idx].tolist()))
        val_paths   = list(dict.fromkeys(paths_arr[val_idx].tolist()))
        suffix = f"ohp_phase3_fold{k}"
        print(f"\n=== Fold {k} | train_files={len(train_paths)} val_files={len(val_paths)} ===")
        _train_one_fold(
            [Path(p) for p in train_paths],
            [Path(p) for p in val_paths],
            pretrain_bilstm, pretrain_stgcn,
            output_dir, suffix, seed=42,
        )


def run_final(
    annotation_dir: Path,
    pretrain_bilstm: Path,
    pretrain_stgcn: Path,
    output_dir: Path,
    seeds: list[int] | None = None,
) -> None:
    """Final 5-seed retrain. Excludes fitnessaqa_split='test' videos if stamped."""
    import json as _json
    all_paths = list(sorted(Path(annotation_dir).glob("*.json")))
    if not all_paths:
        raise RuntimeError(f"No annotation JSONs found in {annotation_dir}")

    # Exclude held-out test videos if split field is stamped
    train_paths = []
    test_paths  = []
    for p in all_paths:
        try:
            split = _json.loads(p.read_text(encoding="utf-8")).get("fitnessaqa_split")
        except Exception:
            split = None
        if split == "test":
            test_paths.append(p)
        else:
            train_paths.append(p)

    if test_paths:
        print(f"[--final] Excluding {len(test_paths)} held-out test video(s) — using {len(train_paths)} for training.")
    else:
        print(f"[--final] No fitnessaqa_split field found — training on all {len(all_paths)} videos.")
        train_paths = all_paths

    seeds_to_run = seeds if seeds is not None else FINAL_SEEDS
    for seed in seeds_to_run:
        suffix = f"ohp_phase3_seed{seed}"
        # Skip if all three checkpoints already exist
        if all((output_dir / f"{m}_{suffix}.pt").exists()
               for m in ["bilstm", "stgcn", "fusion"]):
            print(f"\n=== Final seed={seed} — checkpoints exist, skipping ===")
            continue
        print(f"\n=== Final seed={seed} ===")
        _train_one_fold(
            train_paths, [],
            pretrain_bilstm, pretrain_stgcn,
            output_dir, suffix, seed=seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 OHP multi-task fine-tuning")
    parser.add_argument("--annotation-dir",  type=Path, required=True,
                        help="Path to training_dataset/ohp_phase3_annotations")
    parser.add_argument("--pretrain-bilstm", type=Path, required=True,
                        help="Path to bilstm_ohp_phase2.pt")
    parser.add_argument("--pretrain-stgcn",  type=Path, required=True,
                        help="Path to stgcn_ohp_phase2.pt")
    parser.add_argument("--output-dir",      type=Path, required=True,
                        help="Directory to save checkpoints (e.g. models/)")
    parser.add_argument("--final", action="store_true",
                        help="Retrain on full dataset with 5 seeds (no CV)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Only run these seeds, e.g. --seeds 7 19 3 99 (default: all 5)")
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir) / "videos"

    if args.final:
        run_final(annotation_dir, args.pretrain_bilstm, args.pretrain_stgcn, args.output_dir,
                  seeds=args.seeds)
    else:
        run_cv(annotation_dir, args.pretrain_bilstm, args.pretrain_stgcn, args.output_dir)


if __name__ == "__main__":
    main()
