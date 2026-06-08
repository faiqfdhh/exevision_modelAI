"""Retrain OHP fusion layer only — head-scalar inputs (A) + bucket-weighted loss (B).

Loads FROZEN {bilstm,stgcn}_ohp_finetuned{suffix}.pt encoders+heads — bit-identical
to production, never retrained here — and trains a NEW HeuristicGuidedFusion
(head_dim=7) that additionally consumes the 7 neural head scalars (smoothness,
control, lockout, elbow_flare, grip_ratio, rom_top, rom_bottom) alongside the
heuristic vector and embeddings. Loss is bucket_weighted_mse, which up-weights
quality targets <60 to combat bad-rep blindness (the 40-60 bucket has the worst
quality_mae by far — see results/ohp_ensemble_eval_no7.json).

Saves fusion_ohp_v2{suffix}.pt — a NEW name. The v2 residual_head input dim
(fusion_dim*4) differs from v1 (fusion_dim*3), so v2 cannot load into / be loaded
by v1 architecture. New naming keeps fusion_ohp_finetuned*.pt (v1, locked-in
production ensemble) intact as a rollback path.

CLI (run once per seed used by the production ensemble):
  python retrain_fusion_ohp.py \\
      --annotation-dir training_dataset/ohp_phase3_annotations \\
      --model-dir      models/runtime_neural_ohp \\
      --output-dir     models/runtime_neural_ohp \\
      --suffix _seed42 [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
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
from fusion import build_ohp_fusion, build_ohp_head_vector, OHP_HEAD_DIM
from losses import bucket_weighted_mse
from finetune_ohp import set_seed, _get_split, _make_loader, _freeze, _early_stop

EPOCHS = 60; PATIENCE = 12; LR = 5e-4; SWA_START = 0.67


def _run_epoch(bilstm, stgcn, fusion, loader, opt, device, train: bool) -> float:
    bilstm.eval(); stgcn.eval()   # frozen — never trained here
    fusion.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            b_out = bilstm(bd["bilstm_input"])
            s_out = stgcn(bd["stgcn_input"], bd["view_vec"])
            head_vec = build_ohp_head_vector(b_out, s_out)
            fscore, _ = fusion(bd["heuristic_vec"], s_out["embedding"], b_out["embedding"], head_vec)
            loss = bucket_weighted_mse(fscore / 100.0, bd["quality"] / 100.0)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
    return total / max(len(loader), 1)


def retrain(annotation_dir: Path, model_dir: Path, output_dir: Path, suffix: str = "", seed: int = 42) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={seed}  suffix='{suffix}'")

    bilstm_ckpt = model_dir / f"bilstm_ohp_finetuned{suffix}.pt"
    stgcn_ckpt  = model_dir / f"stgcn_ohp_finetuned{suffix}.pt"
    fusion_name = f"fusion_ohp_v2{suffix}.pt"
    for p in [bilstm_ckpt, stgcn_ckpt]:
        if not p.exists():
            raise RuntimeError(f"Frozen checkpoint not found: {p}. Train it via finetune_ohp.py first.")

    all_paths = sorted((annotation_dir / "videos").glob("*.json"))
    train_paths = [p for p in all_paths if _get_split(p) == "train"]
    val_paths   = [p for p in all_paths if _get_split(p) == "val"]
    print(f"Split — train: {len(train_paths)}  val: {len(val_paths)}")

    train_loader = _make_loader(train_paths, shuffle=True)
    val_loader   = _make_loader(val_paths, shuffle=False)
    if train_loader is None:
        raise RuntimeError("No training reps found. Check annotation_dir and fitnessaqa_split fields.")

    A = torch.tensor(build_adjacency_matrix_ohp(), dtype=torch.float32).to(device)
    bilstm = OHPBiLSTMScorer().to(device)
    stgcn  = OHPSTGCNScorer(A).to(device)
    bilstm.load_state_dict(torch.load(bilstm_ckpt, map_location="cpu"))
    stgcn.load_state_dict(torch.load(stgcn_ckpt, map_location="cpu"))
    _freeze(bilstm); _freeze(stgcn)
    bilstm.eval(); stgcn.eval()
    print(f"Loaded frozen encoders: {bilstm_ckpt.name}, {stgcn_ckpt.name}")

    fusion = build_ohp_fusion(head_dim=OHP_HEAD_DIM).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(fusion.parameters(), lr=LR, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, patience=PATIENCE // 2, factor=0.5, verbose=False)
    best_loss = float("inf")
    best_state = None
    val_hist: list[float] = []
    swa_states: list[dict] = []
    swa_start_ep = max(1, int(EPOCHS * SWA_START))

    print("\n" + "=" * 60)
    print(f"Retraining fusion (head_dim={OHP_HEAD_DIM}, bucket-weighted loss) "
          f"— {EPOCHS} epochs, lr={LR}, SWA from {int(SWA_START*100)}%")
    print("=" * 60)

    for ep in range(1, EPOCHS + 1):
        tr = _run_epoch(bilstm, stgcn, fusion, train_loader, opt, device, train=True)
        vl = _run_epoch(bilstm, stgcn, fusion, val_loader, opt, device, train=False) if val_loader else float("inf")
        metric = vl if val_loader else tr
        sched.step(metric)

        if ep >= swa_start_ep:
            swa_states.append({k: v.clone() for k, v in fusion.state_dict().items()})

        improved = metric < best_loss
        flag = " ✓" if improved else ""
        val_str = f"{vl:.4f}" if val_loader else "N/A"
        print(f"  ep {ep:2d}/{EPOCHS} | train={tr:.4f} | val={val_str:>6}{flag}")
        if improved:
            best_loss = metric
            best_state = {k: v.clone() for k, v in fusion.state_dict().items()}
        val_hist.append(metric)
        if _early_stop(val_hist, PATIENCE):
            print(f"  Early stop at epoch {ep}.")
            break

    if len(swa_states) >= 5:
        print(f"  Evaluating SWA average ({len(swa_states)} snapshots)…")
        swa_state = {k: torch.stack([s[k].float() for s in swa_states]).mean(0) for k in swa_states[0]}
        fusion.load_state_dict(swa_state)
        swa_val = _run_epoch(bilstm, stgcn, fusion, val_loader, opt, device, train=False) if val_loader else float("inf")
        print(f"  SWA val_loss={swa_val:.4f}  best_single={best_loss:.4f}")
        if val_loader and swa_val < best_loss:
            print("  -> SWA wins, saving SWA weights.")
            torch.save(swa_state, output_dir / fusion_name)
        else:
            print("  -> Best single epoch wins.")
            torch.save(best_state, output_dir / fusion_name)
    else:
        torch.save(best_state or fusion.state_dict(), output_dir / fusion_name)

    print(f"\nDone. Best val_loss={best_loss:.4f}")
    print(f"  {fusion_name} -> {output_dir / fusion_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain OHP fusion only (head-scalar inputs + bucket-weighted loss)")
    parser.add_argument("--annotation-dir", type=Path, required=True,
                        help="Path to training_dataset/ohp_phase3_annotations")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Dir containing frozen {bilstm,stgcn}_ohp_finetuned{suffix}.pt")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for fusion_ohp_v2{suffix}.pt output")
    parser.add_argument("--suffix", type=str, default="",
                        help="Seed suffix matching the frozen checkpoints, e.g. _seed42")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for this run")
    args = parser.parse_args()
    retrain(args.annotation_dir, args.model_dir, args.output_dir, suffix=args.suffix, seed=args.seed)


if __name__ == "__main__":
    main()
