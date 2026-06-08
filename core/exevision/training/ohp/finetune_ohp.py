"""OHP Phase 3 fine-tuning — squat methodology.

4 sequential phases, single model output, upfront train/val/test split.

  Phase 1 — BiLSTM temporal heads (smoothness, control, quality)
  Phase 2 — ST-GCN spatial heads  (lockout, elbow_flare, grip_ratio, rom_top, rom_bottom, quality)
  Phase 3 — Fusion layer only     (encoders frozen, quality score)
  Phase 4 — Joint fine-tune       (all unfrozen, optional --joint flag)

CLI:
  python finetune_ohp.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --pretrain-bilstm models/bilstm_ohp_phase2.pt \\
      --pretrain-stgcn  models/stgcn_ohp_phase2.pt \\
      --output-dir      models/ \\
      [--joint]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

_REPO        = Path(__file__).resolve().parents[4]
_NEURAL      = _REPO / "core" / "exevision" / "neural"
_OHP_NEURAL  = _NEURAL / "ohp"
_TRAIN_OHP   = Path(__file__).resolve().parent
_TRAIN_OHP_PRETRAIN = _REPO / "core" / "exevision" / "training" / "overhead_press"
for _p in [str(_NEURAL), str(_OHP_NEURAL), str(_TRAIN_OHP_PRETRAIN), str(_TRAIN_OHP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import build_adjacency_matrix_ohp
from models import OHPBiLSTMScorer, OHPSTGCNScorer
from fusion import build_ohp_fusion
from data_phase3 import OHPPhase3Dataset
from losses import masked_mse, weighted_bce, compute_phase3_loss

# ── Hyper-parameters ───────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 16

P1_EPOCHS = 40;  P1_PATIENCE = 10;  P1_LR = 5e-4
P2_EPOCHS = 40;  P2_PATIENCE = 10;  P2_LR = 5e-4
P3_EPOCHS = 60;  P3_PATIENCE = 12;  P3_LR = 5e-4;  P3_SWA_START = 0.67
P4_EPOCHS = 20;  P4_PATIENCE = 8;   P4_LR = 1e-5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_split(path: Path) -> str:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fitnessaqa_split", "train")
    except Exception:
        return "train"


def _make_loader(paths: list[Path], shuffle: bool) -> torch.utils.data.DataLoader | None:
    ds = OHPPhase3Dataset(paths)
    if len(ds) == 0:
        return None
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        drop_last=(shuffle and len(ds) >= BATCH_SIZE),
        num_workers=0,
    )


def _freeze(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def _freeze_encoder_bilstm(bilstm: OHPBiLSTMScorer) -> None:
    for m in [bilstm.lstm1, bilstm.lstm2, bilstm.temporal_attention]:
        _freeze(m)


def _freeze_encoder_stgcn(stgcn: OHPSTGCNScorer) -> None:
    for m in [stgcn.block1, stgcn.block2, stgcn.block3, stgcn.block4, stgcn.block5]:
        _freeze(m)


# ── Phase runners ──────────────────────────────────────────────────────────────

def _run_bilstm_epoch(bilstm, loader, opt, device, train: bool) -> float:
    bilstm.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = bilstm(bd["bilstm_input"])
            loss = (
                masked_mse(out["smoothness"] / 100.0, bd["smoothness"] / 100.0)
                + masked_mse(out["control"] / 100.0, bd["control"] / 100.0)
                + 0.3 * masked_mse(out["quality"] / 100.0, bd["quality"] / 100.0)
            )
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return total / max(len(loader), 1)


def _run_stgcn_epoch(stgcn, loader, opt, device, train: bool) -> float:
    stgcn.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = stgcn(bd["stgcn_input"], bd["view_vec"])
            loss = (
                0.7 * weighted_bce(out["lockout"], bd["lockout"])
                + 0.5 * masked_mse(out["elbow_flare"] / 100.0, bd["elbow_flare"] / 100.0)
                + 0.5 * masked_mse(out["grip_ratio"] / 100.0, bd["grip_ratio"] / 100.0)
                + 0.5 * masked_mse(out["rom_top"] / 100.0, bd["rom_top"] / 100.0)
                + 0.5 * masked_mse(out["rom_bottom"] / 100.0, bd["rom_bottom"] / 100.0)
                + 0.3 * masked_mse(out["quality"] / 100.0, bd["quality"] / 100.0)
            )
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return total / max(len(loader), 1)


def _run_fusion_epoch(bilstm, stgcn, fusion, loader, opt, device, train: bool) -> float:
    bilstm.eval(); stgcn.eval()   # always eval — encoders frozen in P3
    fusion.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            b_out = bilstm(bd["bilstm_input"])
            s_out = stgcn(bd["stgcn_input"], bd["view_vec"])
            fscore, _ = fusion(bd["heuristic_vec"], s_out["embedding"], b_out["embedding"])
            loss = F.mse_loss(fscore / 100.0, bd["quality"] / 100.0)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return total / max(len(loader), 1)


def _run_joint_epoch(bilstm, stgcn, fusion, loader, opt, device, train: bool) -> float:
    bilstm.train(train); stgcn.train(train); fusion.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            b_out = bilstm(bd["bilstm_input"])
            s_out = stgcn(bd["stgcn_input"], bd["view_vec"])
            fscore, _ = fusion(bd["heuristic_vec"], s_out["embedding"], b_out["embedding"])
            loss = compute_phase3_loss(b_out, s_out, fscore, bd)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return total / max(len(loader), 1)


def _early_stop(val_losses: list[float], patience: int) -> bool:
    if len(val_losses) <= patience:
        return False
    return all(v >= val_losses[-(patience + 1)] for v in val_losses[-patience:])


# ── Main training function ─────────────────────────────────────────────────────

def train(
    annotation_dir: Path,
    pretrain_bilstm: Path,
    pretrain_stgcn: Path,
    output_dir: Path,
    run_joint: bool = False,
    seed: int = SEED,
    suffix: str = "",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={seed}  suffix='{suffix}'")

    # Output checkpoint names — suffix keeps multi-seed ensemble runs from
    # overwriting each other (e.g. "_seed7" → bilstm_ohp_finetuned_seed7.pt).
    bilstm_name = f"bilstm_ohp_finetuned{suffix}.pt"
    stgcn_name  = f"stgcn_ohp_finetuned{suffix}.pt"
    fusion_name = f"fusion_ohp_finetuned{suffix}.pt"

    all_paths = sorted((annotation_dir / "videos").glob("*.json"))
    train_paths = [p for p in all_paths if _get_split(p) == "train"]
    val_paths   = [p for p in all_paths if _get_split(p) == "val"]
    test_paths  = [p for p in all_paths if _get_split(p) == "test"]

    print(f"Split — train: {len(train_paths)}  val: {len(val_paths)}  test: {len(test_paths)}")
    if not val_paths:
        print("[WARN] No val split found. Run stamp_phase3_splits.py first.")
        print("       Continuing with train-only (no early stopping / best-ckpt selection).")

    train_loader = _make_loader(train_paths, shuffle=True)
    val_loader   = _make_loader(val_paths,   shuffle=False)

    if train_loader is None:
        raise RuntimeError("No training reps found. Check annotation_dir and fitnessaqa_split fields.")

    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn  = OHPSTGCNScorer(A).to(device)
    fusion = build_ohp_fusion().to(device)

    # Load Phase 2 pretrained encoder weights
    bilstm.load_phase2_for_phase3(str(pretrain_bilstm))
    stgcn.load_phase2_for_phase3(str(pretrain_stgcn))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: BiLSTM temporal heads ────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Phase 1 — BiLSTM temporal heads ({P1_EPOCHS} epochs, lr={P1_LR})")
    print("="*60)
    _freeze(stgcn); _freeze(fusion)
    _unfreeze(bilstm)

    opt1 = torch.optim.Adam(
        [p for p in bilstm.parameters() if p.requires_grad],
        lr=P1_LR, weight_decay=1e-3,
    )
    sched1 = ReduceLROnPlateau(opt1, patience=P1_PATIENCE // 2, factor=0.5, verbose=False)
    best_p1_loss = float("inf")
    p1_val_hist: list[float] = []

    for ep in range(1, P1_EPOCHS + 1):
        tr = _run_bilstm_epoch(bilstm, train_loader, opt1, device, train=True)
        vl = _run_bilstm_epoch(bilstm, val_loader, opt1, device, train=False) if val_loader else float("inf")
        metric = vl if val_loader else tr
        sched1.step(metric)
        improved = metric < best_p1_loss
        flag = " ✓" if improved else ""
        val_str = f"{vl:.4f}" if val_loader else "N/A"
        print(f"  ep {ep:2d}/{P1_EPOCHS} | train={tr:.4f} | val={val_str:>6}{flag}")
        if improved:
            best_p1_loss = metric
            torch.save(bilstm.state_dict(), output_dir / bilstm_name)
        p1_val_hist.append(metric)
        if _early_stop(p1_val_hist, P1_PATIENCE):
            print(f"  Early stop at epoch {ep}.")
            break

    bilstm.load_state_dict(torch.load(output_dir / bilstm_name, map_location="cpu"))
    print(f"Phase 1 done. Best val_loss={best_p1_loss:.4f}")

    # ── Phase 2: ST-GCN spatial heads ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Phase 2 — ST-GCN spatial heads ({P2_EPOCHS} epochs, lr={P2_LR})")
    print("="*60)
    _freeze(bilstm); _freeze(fusion)
    _unfreeze(stgcn)

    opt2 = torch.optim.Adam(
        [p for p in stgcn.parameters() if p.requires_grad],
        lr=P2_LR, weight_decay=1e-3,
    )
    sched2 = ReduceLROnPlateau(opt2, patience=P2_PATIENCE // 2, factor=0.5, verbose=False)
    best_p2_loss = float("inf")
    p2_val_hist: list[float] = []

    for ep in range(1, P2_EPOCHS + 1):
        tr = _run_stgcn_epoch(stgcn, train_loader, opt2, device, train=True)
        vl = _run_stgcn_epoch(stgcn, val_loader, opt2, device, train=False) if val_loader else float("inf")
        metric = vl if val_loader else tr
        sched2.step(metric)
        improved = metric < best_p2_loss
        flag = " ✓" if improved else ""
        val_str = f"{vl:.4f}" if val_loader else "N/A"
        print(f"  ep {ep:2d}/{P2_EPOCHS} | train={tr:.4f} | val={val_str:>6}{flag}")
        if improved:
            best_p2_loss = metric
            torch.save(stgcn.state_dict(), output_dir / stgcn_name)
        p2_val_hist.append(metric)
        if _early_stop(p2_val_hist, P2_PATIENCE):
            print(f"  Early stop at epoch {ep}.")
            break

    stgcn.load_state_dict(torch.load(output_dir / stgcn_name, map_location="cpu"))
    print(f"Phase 2 done. Best val_loss={best_p2_loss:.4f}")

    # ── Phase 3: Fusion (encoders frozen) ─────────────────────────────────────
    print("\n" + "="*60)
    print(f"Phase 3 — Fusion layer ({P3_EPOCHS} epochs, lr={P3_LR}, SWA from {int(P3_SWA_START*100)}%)")
    print("="*60)
    _freeze(bilstm); _freeze(stgcn)
    _unfreeze(fusion)

    opt3 = torch.optim.Adam(fusion.parameters(), lr=P3_LR, weight_decay=1e-4)
    sched3 = ReduceLROnPlateau(opt3, patience=P3_PATIENCE // 2, factor=0.5, verbose=False)
    best_p3_loss = float("inf")
    best_fusion_state = None
    p3_val_hist: list[float] = []
    swa_states: list[dict] = []
    swa_start_ep = max(1, int(P3_EPOCHS * P3_SWA_START))

    for ep in range(1, P3_EPOCHS + 1):
        tr = _run_fusion_epoch(bilstm, stgcn, fusion, train_loader, opt3, device, train=True)
        vl = _run_fusion_epoch(bilstm, stgcn, fusion, val_loader, opt3, device, train=False) if val_loader else float("inf")
        metric = vl if val_loader else tr
        sched3.step(metric)

        if ep >= swa_start_ep:
            swa_states.append({k: v.clone() for k, v in fusion.state_dict().items()})

        improved = metric < best_p3_loss
        flag = " ✓" if improved else ""
        val_str = f"{vl:.4f}" if val_loader else "N/A"
        print(f"  ep {ep:2d}/{P3_EPOCHS} | train={tr:.4f} | val={val_str:>6}{flag}")
        if improved:
            best_p3_loss = metric
            best_fusion_state = {k: v.clone() for k, v in fusion.state_dict().items()}
        p3_val_hist.append(metric)
        if _early_stop(p3_val_hist, P3_PATIENCE):
            print(f"  Early stop at epoch {ep}.")
            break

    # Try SWA averaged weights against best single
    if len(swa_states) >= 5:
        print(f"  Evaluating SWA average ({len(swa_states)} snapshots)…")
        swa_state = {
            k: torch.stack([s[k].float() for s in swa_states]).mean(0)
            for k in swa_states[0]
        }
        fusion.load_state_dict(swa_state)
        swa_val = _run_fusion_epoch(bilstm, stgcn, fusion, val_loader, opt3, device, train=False) if val_loader else float("inf")
        print(f"  SWA val_loss={swa_val:.4f}  best_single={best_p3_loss:.4f}")
        if val_loader and swa_val < best_p3_loss:
            print("  → SWA wins, saving SWA weights.")
            torch.save(swa_state, output_dir / fusion_name)
        else:
            print("  → Best single epoch wins.")
            torch.save(best_fusion_state, output_dir / fusion_name)
            fusion.load_state_dict(best_fusion_state)
    else:
        torch.save(best_fusion_state or fusion.state_dict(), output_dir / fusion_name)

    print(f"Phase 3 done. Best val_loss={best_p3_loss:.4f}")

    # ── Phase 4: Joint fine-tune (optional) ───────────────────────────────────
    if run_joint:
        print("\n" + "="*60)
        print(f"Phase 4 — Joint fine-tune ({P4_EPOCHS} epochs, lr={P4_LR})")
        print("="*60)
        _unfreeze(bilstm); _unfreeze(stgcn); _unfreeze(fusion)

        opt4 = torch.optim.Adam(
            list(bilstm.parameters()) + list(stgcn.parameters()) + list(fusion.parameters()),
            lr=P4_LR, weight_decay=1e-4,
        )
        sched4 = ReduceLROnPlateau(opt4, patience=P4_PATIENCE // 2, factor=0.5, verbose=False)
        best_p4_loss = float("inf")
        p4_val_hist: list[float] = []

        for ep in range(1, P4_EPOCHS + 1):
            tr = _run_joint_epoch(bilstm, stgcn, fusion, train_loader, opt4, device, train=True)
            vl = _run_joint_epoch(bilstm, stgcn, fusion, val_loader, opt4, device, train=False) if val_loader else float("inf")
            metric = vl if val_loader else tr
            sched4.step(metric)
            improved = metric < best_p4_loss
            flag = " ✓" if improved else ""
            val_str = f"{vl:.4f}" if val_loader else "N/A"
            print(f"  ep {ep:2d}/{P4_EPOCHS} | train={tr:.4f} | val={val_str:>6}{flag}")
            if improved:
                best_p4_loss = metric
                torch.save(bilstm.state_dict(), output_dir / bilstm_name)
                torch.save(stgcn.state_dict(),  output_dir / stgcn_name)
                torch.save(fusion.state_dict(), output_dir / fusion_name)
            p4_val_hist.append(metric)
            if _early_stop(p4_val_hist, P4_PATIENCE):
                print(f"  Early stop at epoch {ep}.")
                break

        print(f"Phase 4 done. Best val_loss={best_p4_loss:.4f}")

    print("\n" + "="*60)
    print("Training complete.")
    print(f"  {bilstm_name} → {output_dir / bilstm_name}")
    print(f"  {stgcn_name}  → {output_dir / stgcn_name}")
    print(f"  {fusion_name} → {output_dir / fusion_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OHP Phase 3 fine-tuning — squat methodology")
    parser.add_argument("--annotation-dir",  type=Path, required=True,
                        help="Path to training_dataset/ohp_phase3_annotations")
    parser.add_argument("--pretrain-bilstm", type=Path, required=True,
                        help="Path to bilstm_ohp_phase2.pt")
    parser.add_argument("--pretrain-stgcn",  type=Path, required=True,
                        help="Path to stgcn_ohp_phase2.pt")
    parser.add_argument("--output-dir",      type=Path, required=True,
                        help="Directory for output checkpoints")
    parser.add_argument("--joint", action="store_true",
                        help="Run optional Phase 4 joint fine-tune after fusion")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for this run (default: 42)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Appended to output checkpoint names, e.g. _seed7")
    args = parser.parse_args()
    train(args.annotation_dir, args.pretrain_bilstm, args.pretrain_stgcn,
          args.output_dir, run_joint=args.joint, seed=args.seed, suffix=args.suffix)


if __name__ == "__main__":
    main()
