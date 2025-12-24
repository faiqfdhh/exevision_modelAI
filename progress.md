## 5. Core Modules

### 5.1 Pose Extraction & Normalisation
- Extract 2D skeletal landmarks per frame
- Body-normalise using torso-based scaling
- Output: time-series landmark tensor

### 5.2 View Validation
- Confirm required camera angle (front/side)
- Reject occluded or invalid perspectives

### 5.3 Temporal Segmentation
- Segment motion into **idle / eccentric / concentric**
- Detect individual repetitions via joint velocity

### 5.4 Symbolic Rule Engine (Primary)
- Deterministic biomechanical rules per exercise
- Detect specific form faults
- Assign error severity (percentile)
- Output: error type, severity, peak timestamp

### 5.5 BiLSTM Temporal Scoring (Secondary)
- Input: rep-level pose sequences
- Output: smoothness / rhythm score
- No direct feedback generation

### 5.6 Score Aggregation
- Fuse symbolic + neural signals
- Produce final **0–100 Form Score**
- Deterministic weighting

### 5.7 Feedback & Evidence Engine
- Slot-filled text templates (no generative AI)
- Extract 3-frame GIFs around peak errors
