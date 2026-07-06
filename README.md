# CDR-Net

**Change Detection and Reconstruction Network for Post-operative Glioma Follow-up**

CDR-Net predicts spatially-resolved tumor change maps — where edema (ED) and active tumor (AT) will increase or decrease — directly from a single time-point MRI, without requiring a follow-up scan at inference.

---

## How it works

```
Input:  Time1 MRI  (T1, T1ce, T2, FLAIR)  →  4-channel 2D slice
             │
     Shared Residual Encoder
            ├──────────────────┐
       ED Decoder          AT Decoder
            │                  │
    [ED_INC, ED_DEC]   [AT_INC, AT_DEC]   ←  predicted change maps

Evaluation:  T2_pred = (T1_mask ∪ INC_pred) \ DEC_pred
             Dice(T2_pred, T2_GT)           ←  Recon Dice
```

Ground truth is derived from paired expert segmentation masks:
```
INC = time2_mask & ~time1_mask   (newly appeared region)
DEC = time1_mask & ~time2_mask   (disappeared region)
```

---

## Repository Structure

```
project_root/
├── dataset.py         ← NIfTI slice dataset, patch crop, augmentation
├── utils.py           ← Loss functions, evaluation metrics
└── CDR-Net/
    ├── model.py       ← Dual-decoder U-Net (ED / AT)
    ├── train.py       ← Standard training
    ├── train_ema.py   ← EMA training (recommended)
    ├── run.py         ← CLI for train.py
    ├── run_ema.py     ← CLI for train_ema.py
    ├── test.py        ← Evaluation (standard)
    └── test_ema.py    ← Evaluation (EMA)
```

---

## Setup

**Dataset** — UCSF Post-operative Glioma Dataset:
```
https://imagingdatasets.ucsf.edu/dataset/2
```

Each patient folder should contain:
```
{pid}_time1_t1.nii.gz      {pid}_time1_t1ce.nii.gz
{pid}_time1_t2.nii.gz      {pid}_time1_flair.nii.gz
{pid}_time1_seg.nii.gz     {pid}_time2_seg.nii.gz
```

**Dependencies:**
```bash
pip install torch nibabel numpy tqdm
```

---

## Usage

```bash
# Train with EMA (recommended)
python run_ema.py --root /path/to/dataset

# Train standard
python run.py --root /path/to/dataset

# Evaluate
python test_ema.py   # or test.py for standard checkpoint
```

---

## Key Design Choices

| | Detail |
|---|---|
| Split | Patient-level 8 / 1 / 1 (no slice-level leakage) |
| Sampling | AT-change slices ×10, ED-change slices ×3 |
| Class imbalance | Per-channel BCE pos_weight (cap 100) |
| Loss — ED | BCE + Dice + Tversky |
| Loss — AT | BCE + Focal + Dice + Tversky + overlap penalty |
| Best model | Val Recon Dice (not Delta Dice) |
| EMA decay | 0.995 |

---

