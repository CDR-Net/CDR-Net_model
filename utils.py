# utils.py
import os
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn

# ---------------- Reproducibility ----------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- I/O ----------------
def find_nii(dirpath, stem):
    for ext in [".nii.gz", ".nii"]:
        p = os.path.join(dirpath, stem + ext)
        if os.path.exists(p):
            return p
    return None

def load_nii(path):
    return nib.load(path).get_fdata().astype(np.float32)

def minmax_norm(x, eps=1e-6):
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)

def ensure_divisible_hw(x, div=16):
    # x: (C,H,W)
    C, H, W = x.shape
    newH = ((H + div - 1) // div) * div
    newW = ((W + div - 1) // div) * div
    pad_h, pad_w = newH - H, newW - W
    if pad_h or pad_w:
        x = np.pad(x, ((0,0),(0,pad_h),(0,pad_w)), mode="edge")
    return x

# ---------------- Losses (logits 기반) ----------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2 * (p * targets).sum(dim=(2,3)) + self.eps
        den = p.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.eps
        return (1 - num / den).mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p * targets).sum(dim=(2,3))
        fp = (p * (1 - targets)).sum(dim=(2,3))
        fn = ((1 - p) * targets).sum(dim=(2,3))
        denom = tp + self.alpha * fn + self.beta * fp + self.eps
        return (1 - (tp + self.eps) / denom).mean()

class FocalLoss(nn.Module):
    """Binary focal loss for logits."""
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = w * (1 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# ---------------- Metrics ----------------
def dice_binary(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps=1e-6) -> float:
    # pred_bin, gt_bin: (B,1,H,W) or (B,H,W)
    if pred_bin.dim() == 4:
        pred_bin = pred_bin[:, 0]
    if gt_bin.dim() == 4:
        gt_bin = gt_bin[:, 0]
    inter = (pred_bin * gt_bin).sum().item()
    den = pred_bin.sum().item() + gt_bin.sum().item()
    return float((2.0 * inter + eps) / (den + eps))

@torch.no_grad()
def eval_delta_and_recon(
    model,
    loader,
    device,
    thr_inc=0.5,
    thr_dec=0.5,
):
    """
    논문용 평가:
      A) Δ 평가: inc/dec Dice (ED/AT 각각)
      B) 복원 평가: T2_pred = (T1 ∪ INC_pred) \ DEC_pred 의 Dice (ED/AT 각각)

    loader가 (x, y(4ch), t1_mask(2ch), t2_mask(2ch))를 반환해야 함.
    """
    model.eval()

    # 누적
    delta_ed_inc, delta_ed_dec = [], []
    delta_at_inc, delta_at_dec = [], []
    recon_ed, recon_at = [], []

    for x, y, t1m, t2m in loader:
        x = x.to(device)
        y = y.to(device)       # (B,4,H,W)
        t1m = t1m.to(device)   # (B,2,H,W)
        t2m = t2m.to(device)   # (B,2,H,W)

        out_ed, out_at = model(x)  # out_ed/out_at: (B,2,H,W)

        # pred bin
        p_ed_inc = (torch.sigmoid(out_ed[:, 0:1]) >= thr_inc).float()
        p_ed_dec = (torch.sigmoid(out_ed[:, 1:2]) >= thr_dec).float()
        p_at_inc = (torch.sigmoid(out_at[:, 0:1]) >= thr_inc).float()
        p_at_dec = (torch.sigmoid(out_at[:, 1:2]) >= thr_dec).float()

        # GT delta
        gt_ed_inc = y[:, 0:1]
        gt_ed_dec = y[:, 1:2]
        gt_at_inc = y[:, 2:3]
        gt_at_dec = y[:, 3:4]

        # A) delta dice (batch-wise scalar)
        delta_ed_inc.append(dice_binary(p_ed_inc, gt_ed_inc))
        delta_ed_dec.append(dice_binary(p_ed_dec, gt_ed_dec))
        delta_at_inc.append(dice_binary(p_at_inc, gt_at_inc))
        delta_at_dec.append(dice_binary(p_at_dec, gt_at_dec))

        # B) recon dice
        t1_ed = t1m[:, 0:1]
        t1_at = t1m[:, 1:2]
        t2_ed = t2m[:, 0:1]
        t2_at = t2m[:, 1:2]

        # T2_pred = (T1 ∪ INC) \ DEC  ==  (T1 OR INC) AND (NOT DEC)
        t2p_ed = torch.clamp(t1_ed + p_ed_inc, 0, 1) * (1.0 - p_ed_dec)
        t2p_at = torch.clamp(t1_at + p_at_inc, 0, 1) * (1.0 - p_at_dec)

        # binarize (already 0/1 but keep safe)
        t2p_ed = (t2p_ed > 0.5).float()
        t2p_at = (t2p_at > 0.5).float()

        recon_ed.append(dice_binary(t2p_ed, t2_ed))
        recon_at.append(dice_binary(t2p_at, t2_at))

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    return {
        "Delta": {
            "ED_INC": _mean(delta_ed_inc),
            "ED_DEC": _mean(delta_ed_dec),
            "AT_INC": _mean(delta_at_inc),
            "AT_DEC": _mean(delta_at_dec),
            "Mean": float(np.mean([_mean(delta_ed_inc), _mean(delta_ed_dec), _mean(delta_at_inc), _mean(delta_at_dec)])),
        },
        "Recon_T2": {
            "ED": _mean(recon_ed),
            "AT": _mean(recon_at),
            "Mean": float(np.mean([_mean(recon_ed), _mean(recon_at)])),
        }
    }
