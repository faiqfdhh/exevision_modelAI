# MODULE 1: AI EVALUATION MODULE
## Complete Training Methodology Diagram

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                         TRAINING METHODOLOGY — COMPLETE PIPELINE                                           ║
║                        (PRE-TRAINING → FINE-TUNING → INFERENCE)                                           ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PRE-TRAINING STAGE (Self-Supervised Learning)                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  Objective: Initialize encoder weights using self-supervised tasks on FULL DATASET                         │
│            (no ground-truth labels needed; uses reconstruction loss)                                       │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐            │
│  │ INPUT: Raw Features (5250 extracted features) from ALL available squat videos              │            │
│  │        ↓                                                                                   │            │
│  │ SPLIT: Load all extracted rep sequences (temporal signals + skeleton graphs)              │            │
│  │        No stratification needed — just using for representation learning                  │            │
│  └────────────┬───────────────────────────────────────────────────────────────────────────────┘            │
│               │                                                                                            │
│  ┌────────────▼───────────────────────────────────────────────────────────────────────────────┐            │
│  │ ╔════════════════════════════════════════════════════════════════════════════════════════╗ │            │
│  │ ║ PRETRAIN PHASE A: BiLSTM Self-Supervised Learning                                    ║ │            │
│  │ ║ ─────────────────────────────────────────────────────────────────────────────────── ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ Task: Masked Temporal Reconstruction (like BERT masking)                             ║ │            │
│  │ ║ ─────────────────────────────────────────────────────────────────────────────────── ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ BiLSTMPretrainer Architecture:                                                       ║ │            │
│  │ ║ ├─ Input: Temporal sequence with 25% of timesteps randomly masked to 0               ║ │            │
│  │ ║ ├─ LSTM Encoder: 2 layers, bidirectional (hidden=128, output=256 dims)               ║ │            │
│  │ ║ ├─ Temporal Attention: Compress sequence → single vector (256 dims)                  ║ │            │
│  │ ║ ├─ Reconstruction Head: Linear(256) → NUM_BILSTM_CHANNELS to predict masked region   ║ │            │
│  │ ║ └─ Output: Reconstructed full sequence at original shape                             ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ Hyperparameters:                                                                     ║ │            │
│  │ ║ ├─ Mask Ratio: 25% of frames                                                         ║ │            │
│  │ ║ ├─ Min Mask Length: 10 timesteps                                                     ║ │            │
│  │ ║ ├─ Batch Size: 64                                                                    ║ │            │
│  │ ║ ├─ Learning Rate: 1e-3                                                               ║ │            │
│  │ ║ ├─ Epochs: 80                                                                        ║ │            │
│  │ ║ ├─ Loss: MSE(reconstruction_pred, original_sequence)                                 ║ │            │
│  │ ║ └─ Dropout: 0.3 (regularization)                                                     ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ Why This Works:                                                                      ║ │            │
│  │ ║ • Forces encoder to learn meaningful temporal patterns (velocity, acceleration)       ║ │            │
│  │ ║ • Doesn't require hand-labeled scores — just structure in the data                    ║ │            │
│  │ ║ • Creates rich latent representations useful for downstream fine-tuning               ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ╚════════════════════════════════════════════════════════════════════════════════════════╝ │            │
│  │                                         ▼                                               │            │
│  │                                 ✓ Save: bilstm_pretrained.pt                            │            │
│  │                                   (encoder weights only, discard reconstruction head)    │            │
│  └────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                          │                                                             │
│  ┌────────────────────────────────────────▼───────────────────────────────────────────────────┐            │
│  │ ╔════════════════════════════════════════════════════════════════════════════════════════╗ │            │
│  │ ║ PRETRAIN PHASE B: ST-GCN Self-Supervised Learning                                    ║ │            │
│  │ ║ ─────────────────────────────────────────────────────────────────────────────────── ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ Task: Masked Skeleton Reconstruction (mask joints, predict positions)                ║ │            │
│  │ ║ ─────────────────────────────────────────────────────────────────────────────────── ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ STGCNPretrainer Architecture:                                                        ║ │            │
│  │ ║ ├─ Input: Skeleton graph (33 joints, 60 frames, 2 coords) with random joints masked  ║ │            │
│  │ ║ ├─ ST-GCN Encoder: 5 blocks alternating spatial (graph conv) + temporal convolutions  ║ │            │
│  │ ║ │   • Block 1-2: Spatial/temporal with stride 1 (preserve all frames)                ║ │            │
│  │ ║ │   • Block 3: Stride 2 (downsample to 30 frames)                                    ║ │            │
│  │ ║ │   • Block 4: Stride 1 (maintain 30 frames)                                         ║ │            │
│  │ ║ │   • Block 5: Stride 2 (downsample to 15 frames) → output (256 dims)                ║ │            │
│  │ ║ ├─ Decoder: FC(256) → FC(512) → FC(channels×frames×joints) to reconstruct full graph ║ │            │
│  │ ║ └─ Output: Predicted skeleton coordinates at original shape                           ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ Hyperparameters:                                                                     ║ │            │
│  │ ║ ├─ Batch Size: 32                                                                    ║ │            │
│  │ ║ ├─ Learning Rate: 1e-3                                                               ║ │            │
│  │ ║ ├─ Epochs: 60                                                                        ║ │            │
│  │ ║ ├─ Loss: MSE(decoder_output, original_skeleton)                                      ║ │            │
│  │ ║ └─ Dropout: 0.2 (regularization)                                                     ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ║ Why This Works:                                                                      ║ │            │
│  │ ║ • Forces encoder to learn spatial biomechanics (skeleton structure, joint angles)     ║ │            │
│  │ ║ • Graph convolutions preserve skeletal kinematic chains (parent→child joints)         ║ │            │
│  │ ║ • Creates embeddings useful for depth, lean, knee-tracking predictions                ║ │            │
│  │ ║                                                                                      ║ │            │
│  │ ╚════════════════════════════════════════════════════════════════════════════════════════╝ │            │
│  │                                         ▼                                               │            │
│  │                                 ✓ Save: stgcn_pretrained.pt                             │            │
│  │                                   (encoder weights only, discard decoder)                │            │
│  └────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                                                                                             │
│  PRETRAIN SUMMARY:                                                                                        │
│  ├─ Uses all available squat videos (no train/test split)                                                 │
│  ├─ Creates general-purpose movement understanding (no labels needed)                                     │
│  ├─ Initializes encoder weights for downstream fine-tuning                                                │
│  └─ Reduces overfitting on small annotated dataset (121 reps) via transfer learning                      │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA PREPARATION (SUPERVISED FINE-TUNING)                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌────────────────────────────────────────┐                                                               │
│  │ Annotation Index                       │                                                               │
│  │ training_dataset/annotations/          │                                                               │
│  │ index.json (147 reps)                  │                                                               │
│  └─────────────┬──────────────────────────┘                                                               │
│                │                                                                                           │
│                ├─→ Stratified Split (Seed 42)                                                             │
│                │   └─ Bucket edges: [20, 40, 60, 80, 100]                                                 │
│                │   └─ Train: 121 reps | Test: 26 reps                                                     │
│                │                                                                                           │
│  ┌─────────────▼──────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ ╔════════════════════════════════════════════════════════════════════════════════════════════════╗ │   │
│  │ ║  Load & Compute Training Features                                                            ║ │   │
│  │ ║  ─────────────────────────────────────────────────────────────────────────────────────────   ║ │   │
│  │ ║  1. Raw Features: 5250 extracted features from Stage 2.5 (MediaPipe landmarks)              ║ │   │
│  │ ║  2. BiLSTM Input: Temporal sequences (T=60, C=11)                                           ║ │   │
│  │ ║  3. ST-GCN Input: Skeleton graph (N=33 joints, T=60, C=2 coords)                            ║ │   │
│  │ ║  4. Heuristic Score: Rule-based AQA score from Stage 8 (0-100)                              ║ │   │
│  │ ║  5. Human Score: Ground truth from annotations (continuous)                                 ║ │   │
│  │ ║  6. Temporal Targets: Smoothness, Control (sub-metrics)                                     ║ │   │
│  │ ║  7. Spatial Targets: Depth, Forward Lean, Knee Tracking (sub-metrics)                       ║ │   │
│  │ ║  8. Auxiliary Targets: Min knee angle, Lean deg, Knee valgus, Squat depth                   ║ │   │
│  │ ║  9. Stratified Weights: Bucket-based balancing for imbalanced classes                       ║ │   │
│  │ ╚════════════════════════════════════════════════════════════════════════════════════════════════╝ │   │
│  └─────────────┬──────────────────────────────────────────────────────────────────────────────────────┘   │
│                │                                                                                           │
│                ▼                                                                                           │
│        DataLoaders Ready                                                                                  │
│        (train_loader, val_loader)                                                                        │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: BILSTM TEMPORAL FINE-TUNING                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  Input: BiLSTM Temporal Sequences (121 reps × 60 timesteps)                                               │
│                                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐             │
│  │                                                                                          │             │
│  │  ╔══════════════════════════════════════════════════════════════════════════════════╗   │             │
│  │  ║  BiLSTMScorer                                                                   ║   │             │
│  │  ║  ─────────────────────────────────────────────────────────────────────────────  ║   │             │
│  │  ║  • Embedding Layer: Temporal → Hidden (256 dims)                                ║   │             │
│  │  ║  • Bi-LSTM Layers: Forward + Backward (2 layers, 128 each)                      ║   │             │
│  │  ║  • Temporal Head: Predicts [Smoothness, Control]                                ║   │             │
│  │  ║  • Dropout: 0.3 (regularization)                                                ║   │             │
│  │  ║                                                                                  ║   │             │
│  │  ║  Output Shape: (batch_size, 2) → [smoothness_pred, control_pred]                ║   │             │
│  │  ╚══════════════════════════════════════════════════════════════════════════════════╝   │             │
│  │                                                                                          │             │
│  │  Training Configuration:                                                               │             │
│  │  ├─ Optimizer: Adam (lr=1e-3 default)                                                  │             │
│  │  ├─ Loss: Masked Weighted MSE (temporal_target vs temporal_pred)                       │             │
│  │  ├─ Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)                              │             │
│  │  ├─ Gradient Clipping: max_norm=1.0                                                    │             │
│  │  ├─ Early Stopping: Best validation loss checkpoint                                    │             │
│  │  └─ Epochs: 100 (or configurable)                                                      │             │
│  │                                                                                          │             │
│  │  Target: batch["temporal_target"] = [smoothness_human, control_human]                  │             │
│  │  Mask: batch["temporal_mask"] = [valid_smoothness, valid_control]                      │             │
│  │                                                                                          │             │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘             │
│                                                                                                             │
│                              ▼                                                                            │
│                       ✓ Save: bilstm_finetuned.pt                                                         │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: ST-GCN SPATIAL FINE-TUNING                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  Input: Skeleton Graph (121 reps × 33 joints × 60 timesteps × 2 coords)                                  │
│                                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐             │
│  │                                                                                          │             │
│  │  ╔══════════════════════════════════════════════════════════════════════════════════╗   │             │
│  │  ║  STGCNScorer (Spatio-Temporal Graph Convolution Network)                         ║   │             │
│  │  ║  ─────────────────────────────────────────────────────────────────────────────  ║   │             │
│  │  ║  • Input: Skeleton graph (33 joints) + Temporal dimension (60 frames)            ║   │             │
│  │  ║  • Graph: Fixed adjacency matrix (skeletal connections)                          ║   │             │
│  │  ║  • ST-GCN Blocks: 3 layers alternating spatial & temporal convolutions           ║   │             │
│  │  ║  • Graph Embedding: (batch, 256) → spatial representation                        ║   │             │
│  │  ║  • Spatial Head: [Depth, Forward Lean, Knee Tracking] predictions                ║   │             │
│  │  ║  • Auxiliary Head: [Min Knee Angle, Lean Deg, Knee Valgus, Squat Depth]         ║   │             │
│  │  ║  • Dropout: 0.3                                                                  ║   │             │
│  │  ║                                                                                  ║   │             │
│  │  ║  Output Shapes:                                                                  ║   │             │
│  │  ║    - spatial_pred: (batch, 3) → [depth, forward_lean, knee_tracking]             ║   │             │
│  │  ║    - aux_pred: (batch, 4) → [min_knee_angle, lean_deg, knee_valgus, depth]      ║   │             │
│  │  ╚══════════════════════════════════════════════════════════════════════════════════╝   │             │
│  │                                                                                          │             │
│  │  Training Configuration:                                                               │             │
│  │  ├─ Optimizer: Adam (lr=1e-3 default)                                                  │             │
│  │  ├─ Loss: Combined Masked Weighted MSE                                                 │             │
│  │  │        loss = L_spatial + 0.3 × L_aux                                               │             │
│  │  ├─ Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)                              │             │
│  │  ├─ Gradient Clipping: max_norm=1.0                                                    │             │
│  │  ├─ Early Stopping: Best validation loss checkpoint                                    │             │
│  │  └─ Epochs: 100 (or configurable)                                                      │             │
│  │                                                                                          │             │
│  │  Targets:                                                                              │             │
│  │  ├─ spatial_target: [depth_human, forward_lean_human, knee_tracking_human]             │             │
│  │  ├─ aux_target: [knee_angle_human, lean_deg_human, valgus_human, depth_human]         │             │
│  │  └─ Masks: Indicate valid target values per sample                                     │             │
│  │                                                                                          │             │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘             │
│                                                                                                             │
│                              ▼                                                                            │
│                       ✓ Save: stgcn_finetuned.pt                                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: FUSION LAYER TRAINING (WITH ENCODER FINE-TUNING)                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  Inputs: Heuristic Scores + BiLSTM Embedding + ST-GCN Embedding                                           │
│                                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐             │
│  │                                                                                          │             │
│  │  ╔══════════════════════════════════════════════════════════════════════════════════╗   │             │
│  │  ║  Fusion Phase: Combine Heuristic + Learned Representations                       ║   │             │
│  │  ║  ─────────────────────────────────────────────────────────────────────────────  ║   │             │
│  │  ║                                                                                  ║   │             │
│  │  ║  Step 3.1: Load Pre-trained Encoders                                            ║   │             │
│  │  ║  ├─ BiLSTM (from Phase 1): frozen initially, then unfrozen with lr×0.05          ║   │             │
│  │  ║  ├─ ST-GCN (from Phase 2): frozen initially, then unfrozen with lr×0.05          ║   │             │
│  │  ║  └─ Extract embeddings: bilstm_embedding, stgcn_embedding (256 dims each)        ║   │             │
│  │  ║                                                                                  ║   │             │
│  │  ║  Step 3.2: HeuristicGuidedFusion Architecture                                   ║   │             │
│  │  ║  ├─ Input Fusion:                                                                ║   │             │
│  │  ║  │   Concatenate [heuristic_vector (16 dims), stgcn_embed (256), bilstm_embed]  ║   │             │
│  │  ║  │   → (batch, 528)                                                              ║   │             │
│  │  ║  │                                                                               ║   │             │
│  │  ║  ├─ Fusion Network:                                                              ║   │             │
│  │  ║  │   FC(528) → ReLU → FC(128) → ReLU → FC(64) → ReLU                            ║   │             │
│  │  ║  │   Dropout(0.1) applied at each ReLU                                           ║   │             │
│  │  ║  │                                                                               ║   │             │
│  │  ║  ├─ Output Heads:                                                                ║   │             │
│  │  ║  │   • Correction Head: FC(64) → tanh(x) × 40 → residual ∈ [-40, +40]           ║   │             │
│  │  ║  │   • Final Score: clamp(heuristic + residual, 0, 100)                          ║   │             │
│  │  ║  │                                                                               ║   │             │
│  │  ║  ║  KEY DESIGN FIX (2026-03-21):                                                ║   │             │
│  │  ║  ║  ────────────────────────────────                                            ║   │             │
│  │  ║  ║  • REMOVED: L1 regularization penalty on residuals (caused collapse)          ║   │             │
│  │  ║  ║  • UNFROZE: Both encoders (pretrained, now adapting to correction task)       ║   │             │
│  │  ║  ║  • BOUNDED: Residual via tanh×40 (replaces L1, provides natural constraint)   ║   │             │
│  │  ║  ║  • DIFFERENTIAL LR: Encoders at 0.05×lr (slow, preserving representations)    ║   │             │
│  │  ║  ║  • WEIGHT DECAY: L2 only (1e-4) on fusion layer                               ║   │             │
│  │  ║  ║  • SWA: Stochastic Weight Averaging after epoch > epochs/3                    ║   │             │
│  │  ║  ║                                                                               ║   │             │
│  │  ║  ║  RESULT: Residual std now ~14-15 (vs prior collapse to ~0.05)                 ║   │             │
│  │  ║  ║          Per-rep corrections meaningful & varied                               ║   │             │
│  │  ║  ║                                                                               ║   │             │
│  │  ║  └─ Output: neural_score ∈ [0, 100]                                             ║   │             │
│  │  ║                                                                                  ║   │             │
│  │  ╚══════════════════════════════════════════════════════════════════════════════════╝   │             │
│  │                                                                                          │             │
│  │  Optimizer Configuration (Differential LR):                                            │             │
│  │  ├─ BiLSTM params: lr × 0.05 = slow encoder adaptation                                │             │
│  │  ├─ ST-GCN params: lr × 0.05 = slow encoder adaptation                                │             │
│  │  └─ Fusion params: lr (default 1e-3), weight_decay=1e-4                               │             │
│  │                                                                                          │             │
│  │  Loss Function:                                                                        │             │
│  │  ├─ Target: batch["human_score"] (continuous, 0-100)                                   │             │
│  │  ├─ Prediction: clamp(heuristic + tanh(residual)×40, 0, 100)                           │             │
│  │  ├─ Loss: Weighted MSE = mean((pred - target)² × sample_weights)                       │             │
│  │  └─ No regularization penalty on residuals                                             │             │
│  │                                                                                          │             │
│  │  Training Configuration:                                                               │             │
│  │  ├─ Optimizer: Adam (multi-param groups with differential LR)                          │             │
│  │  ├─ Scheduler: ReduceLROnPlateau (patience=8, factor=0.5)                              │             │
│  │  ├─ Gradient Clipping: max_norm=1.0 (all params combined)                              │             │
│  │  ├─ Early Stopping: patience=25 (more tolerance for encoder adaptation)                │             │
│  │  ├─ SWA: Enabled from epoch = epochs/3 onwards                                        │             │
│  │  └─ Epochs: 200 (or configurable; SWA smooths loss landscape)                          │             │
│  │                                                                                          │             │
│  │  Validation Monitoring (per epoch):                                                    │             │
│  │  ├─ Val Loss: MSE on test batch                                                       │             │
│  │  ├─ Mean |Residual|: Average absolute correction                                       │             │
│  │  ├─ Residual Std: Spread of corrections (should be ~14-15 post-fix)                    │             │
│  │  └─ Learning Rates: Logged per parameter group                                         │             │
│  │                                                                                          │             │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘             │
│                                                                                                             │
│                              ▼                                                                            │
│                       ✓ Save: fusion_layer.pt                                                            │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: EVALUATION                                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  Evaluate Trained Fusion on Test Set (26 reps)                                                            │
│                                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐             │
│  │                                                                                          │             │
│  │  Test Set Metrics (2026-03-21 POST-FIX):                                               │             │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐   │             │
│  │  │                                                                                  │   │             │
│  │  │  Post-Clamp Metrics (after applying clamp(0, 100)):                              │   │             │
│  │  │  ├─ Pearson Correlation: 0.8737 ✅ (excellent fit)                              │   │             │
│  │  │  ├─ Mean Absolute Error: 9.04 pts                                               │   │             │
│  │  │  ├─ Heuristic Baseline MAE: 12.08 pts (neural is -3.04 pts better)               │   │             │
│  │  │  └─ Linear Baseline MAE: 10.39 pts (neural is -1.35 pts better) ✅               │   │             │
│  │  │                                                                                  │   │             │
│  │  │  Pre-Clamp Metrics (raw tanh×40 output):                                         │   │             │
│  │  │  ├─ Pearson Correlation: 0.8552 (high temporal fidelity)                        │   │             │
│  │  │  ├─ Mean Absolute Error: 9.28 pts                                               │   │             │
│  │  │  └─ Beats linear baseline ✅ (linear MAE was 10.39)                              │   │             │
│  │  │                                                                                  │   │             │
│  │  │  Residual Statistics:                                                            │   │             │
│  │  │  ├─ Mean: ~0.0 (well-centered; unbiased)                                        │   │             │
│  │  │  ├─ Std Dev: ~14-15 pts (meaningful per-rep corrections)                        │   │             │
│  │  │  ├─ Prior (broken): std ≈ 0.05 pts (collapse → near-constant offset) ✗          │   │             │
│  │  │  └─ Failure cases (|error| > 20): 0 ✅ (vs 4 pre-fix)                            │   │             │
│  │  │                                                                                  │   │             │
│  │  │  Per-Metric MAE (spatial predictions from ST-GCN head):                          │   │             │
│  │  │  ├─ Smoothness: ~10.5 pts (learned well)                                        │   │             │
│  │  │  ├─ Control: ~10.2 pts (learned well)                                           │   │             │
│  │  │  ├─ Depth: ~24 pts (high noise on diagonal views; needs more data/normalization) │   │             │
│  │  │  ├─ Forward Lean: ~12 pts (reasonable)                                          │   │             │
│  │  │  └─ Knee Tracking: ~24 pts (high noise; same Z-drift issue as depth)            │   │             │
│  │  │                                                                                  │   │             │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘   │             │
│  │                                                                                          │             │
│  │  Comparison with Baselines:                                                            │             │
│  │  ┌────────────────────┬──────────┬──────────┬──────────┐                              │             │
│  │  │ Method             │ MAE      │ Pearson  │ Better?  │                              │             │
│  │  ├────────────────────┼──────────┼──────────┼──────────┤                              │             │
│  │  │ Heuristic (Rule)   │ 12.08    │ 0.77     │ baseline │                              │             │
│  │  │ Linear (L2 fit)    │ 10.39    │ 0.84     │ N/A      │                              │             │
│  │  │ Neural (Fusion)    │ 9.04     │ 0.8737   │ ✅ YES   │                              │             │
│  │  └────────────────────┴──────────┴──────────┴──────────┘                              │             │
│  │                                                                                          │             │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘             │
│                                                                                                             │
│                              ▼                                                                            │
│                ✓ Export: results/evaluation_report.json                                                   │
│                         (Full metrics + breakdowns)                                                      │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: INFERENCE (Production Pipeline Integration)                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  Load Saved Checkpoints → Run on New Videos → Output Final Scores                                         │
│                                                                                                             │
│  Pipeline Flow:                                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐             │
│  │                                                                                          │             │
│  │  Input: New Video (not in training set)                                                 │             │
│  │    ▼                                                                                     │             │
│  │  Stage 2.5: Extract Features (MediaPipe pose)                                           │             │
│  │    ├─ Output: Raw Features JSON                                                         │             │
│  │    ├─ Contains: 5250 landmark coords, confidences, velocities                           │             │
│  │    └─ Format: {bilstm_features, stgcn_features, extracted at per-rep}                   │             │
│  │    ▼                                                                                     │             │
│  │  Stage 4: Classify View                                                                 │             │
│  │    ▼                                                                                     │             │
│  │  Stage 5: Segment Reps & Phases (Temporal FSM)                                          │             │
│  │    ▼                                                                                     │             │
│  │  Stage 8: Score via Heuristic Rules (0-100)                                             │             │
│  │    ├─ Output: heuristic_score, metrics (depth, lean, knee_angle, valgus)                │             │
│  │    ▼                                                                                     │             │
│  │  Stage 9: NEURAL FUSION INFERENCE (NEW)                                                 │             │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐    │             │
│  │  │ 1. Load Checkpoints:                                                           │    │             │
│  │  │    ├─ models/bilstm_finetuned.pt → BiLSTM                                      │    │             │
│  │  │    ├─ models/stgcn_finetuned.pt → ST-GCN                                       │    │             │
│  │  │    └─ models/fusion_layer.pt → HeuristicGuidedFusion                           │    │             │
│  │  │                                                                                 │    │             │
│  │  │ 2. For each rep:                                                               │    │             │
│  │  │    a. Extract temporal features → BiLSTM                                       │    │             │
│  │  │       Output: bilstm_embedding (256 dims)                                      │    │             │
│  │  │                                                                                 │    │             │
│  │  │    b. Extract skeleton graph → ST-GCN                                          │    │             │
│  │  │       Output: stgcn_embedding (256 dims)                                       │    │             │
│  │  │                                                                                 │    │             │
│  │  │    c. Build heuristic vector: [score, depth, lean, angle, valgus, metrics...] │    │             │
│  │  │       Shape: (16,)                                                             │    │             │
│  │  │                                                                                 │    │             │
│  │  │    d. Fusion: Concatenate + forward through HeuristicGuidedFusion              │    │             │
│  │  │       Input: [heuristic_vec, stgcn_embed, bilstm_embed]                        │    │             │
│  │  │       Output: residual correction via tanh×40                                  │    │             │
│  │  │                                                                                 │    │             │
│  │  │    e. Compute final neural score:                                              │    │             │
│  │  │       neural_score = clamp(heuristic + residual, 0, 100)                       │    │             │
│  │  │                                                                                 │    │             │
│  │  │ 3. Output per-rep:                                                             │    │             │
│  │  │    ├─ heuristic_score (from rules)                                             │    │             │
│  │  │    ├─ bilstm_score (temporal judgment, 0-100 range)                            │    │             │
│  │  │    ├─ stgcn_score (spatial judgment, 0-100 range)                              │    │             │
│  │  │    └─ neural_score (fused, with tanh×40 bounded correction)                    │    │             │
│  │  │                                                                                 │    │             │
│  │  └────────────────────────────────────────────────────────────────────────────────┘    │             │
│  │    ▼                                                                                     │             │
│  │  Output: api/pipeline.py merges results → Final JSON                                    │             │
│  │    ├─ result.scores (array of neural_score per rep)                                    │             │
│  │    ├─ result.judges (bilstm, stgcn, heuristic per rep)                                 │             │
│  │    ├─ result.neural_available: true (if inference succeeded)                            │             │
│  │    ├─ result.feedback (coaching narratives)                                             │             │
│  │    └─ result.status: "done"                                                             │             │
│  │                                                                                          │             │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘             │
│                                                                                                             │
│  Integration Points:                                                                                      │
│  ├─ Desktop UI: `apps/desktop-ui/app.py` (heuristic-only; Stage 9 not wired yet)                         │
│  └─ FastAPI Server: `apps/api/pipeline.py` (full Stage 9 integration ✅; deployed to GCR)                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ SUMMARY: TRAINING → INFERENCE FLOW                                                                        ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                            ║
║  Training Objectives (Phased Fine-Tuning):                                                                ║
║  ├─ Phase 1 (BiLSTM): Learn temporal quality (smoothness, control) from sequence data                      ║
║  ├─ Phase 2 (ST-GCN): Learn spatial quality (depth, lean, tracking) from skeleton graphs                  ║
║  └─ Phase 3 (Fusion): Learn how to correct heuristic scores via neural residuals (±40 pts)               ║
║                                                                                                            ║
║  Key Innovation (Fixed 2026-03-21):                                                                       ║
║  ├─ BEFORE: L1 penalty on residuals caused collapse → const +0.4 offset; std ≈ 0.05                       ║
║  └─ AFTER: Tanh×40 bounded residual + differential LR → meaningful per-rep corrections; std ≈ 14-15      ║
║                                                                                                            ║
║  Metrics (Test Set, 26 reps):                                                                             ║
║  ├─ Pearson: 0.8737 (excellent correlation with human judgment)                                           ║
║  ├─ MAE: 9.04 pts (better than heuristic 12.08, linear 10.39)                                             ║
║  ├─ Failure cases: 0 (pre-fix had 4)                                                                      ║
║  └─ Beats all baselines ✅                                                                                 ║
║                                                                                                            ║
║  Deployment:                                                                                              ║
║  ├─ Checkpoints: models/{bilstm,stgcn,fusion}_finetuned.pt                                                ║
║  ├─ Stage 9 Script: core/exevision/stages/neural_fusion_inference.py                                      ║
║  ├─ API Integration: apps/api/pipeline.py (DEFAULT_STAGES includes Stage 9) ✅                            ║
║  └─ Cloud Run: GCR deployment with Stage 9 enabled (asia-southeast1)                                      ║
║                                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Architecture Overview (Text Diagram)

```
RAW VIDEO INPUT
      ↓
   [Stage 2.5: Feature Extraction]
      ↓
   [Stage 4: View Classification]
      ↓
   [Stage 5: Temporal Segmentation] ← Reps & Phases
      ↓
   [Stage 8: Heuristic Scoring] ← Rule-based 0-100
      ├─ Output: heuristic_score + metrics
      ↓
   [Stage 9: Neural Fusion] ← THREE-JUDGE INFERENCE
      ├─ BiLSTM Judge: Temporal quality
      ├─ ST-GCN Judge: Spatial quality
      └─ Fusion Judge: Heuristic + corrections
         Formula: clamp(heuristic + tanh(residual)×40, 0, 100)
      ↓
   FINAL OUTPUT: neural_score [0, 100]
```

---

## Key Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Seed** | 42 | Deterministic split |
| **Train/Test Split** | 121/26 | Stratified by score bucket |
| **BiLSTM LR** | 1e-3 | Phase 1 training |
| **ST-GCN LR** | 1e-3 | Phase 2 training |
| **Fusion LR** | 1e-3 | Phase 3 fusion layer |
| **Encoder LR** | 1e-3 × 0.05 | Phase 3 encoder fine-tuning (differential LR) |
| **Weight Decay** | 1e-4 | L2 on fusion layer only |
| **Dropout** | 0.3 (Phases 1-2), 0.1 (Phase 3) | Regularization |
| **Gradient Clip** | 1.0 | Stability |
| **Scheduler** | ReduceLROnPlateau | patience=5 (P1-2), patience=8 (P3); factor=0.5 |
| **Early Stop Patience** | 25 (Phase 3) | Allow encoder adaptation time |
| **SWA** | Enabled from epoch > epochs/3 | Stochastic Weight Averaging |
| **Residual Bound** | tanh(x) × 40 | [-40, +40] correction range |

---

## Known Limitations & Future Work

1. **Small test set (26 reps)**: No 0-20 score representation; poor-form generalization unvalidated
2. **High spatial MAE on diagonal views**: Depth & knee-tracking ~24 pts due to Z-drift and view coupling
3. **Desktop UI integration pending**: Stage 9 not yet wired into `apps/desktop-ui/app.py`
4. **Phase 4 joint fine-tuning not validated**: Could provide additional gains if all layers retrain together
5. **Dataset expansion needed**: 30+ more annotations in 20-60 range to improve mid-range accuracy
