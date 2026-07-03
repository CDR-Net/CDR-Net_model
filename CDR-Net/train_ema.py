# train_cdrnet_ema.py

import os
import re
import json
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import BrainChangeSliceDataset
from model import UNetNoPoolingDualDecoder


# ============================================================
# 0. 설정
# ============================================================

ROOT = r"/Users/amy/PycharmProjects/labresearch/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"

# 기존 best에서 시작해서 EMA fine-tuning
START_CKPT = r"/Users/amy/PycharmProjects/labresearch/CDR-Net/best_change_seg_incdec.pt"


BATCH_SIZE = 4

SAVE_BEST = r"/Users/amy/PycharmProjects/labresearch/CDR-Net/best_change_seg_incdec_ema_bs4.pt"
SAVE_LAST = r"/Users/amy/PycharmProjects/labresearch/last_change_seg_incdec_ema_bs4.pt"
OUT_LOG = r"/Users/amy/PycharmProjects/labresearch/train_log_cdrnet_ema_bs4.json"

SEED = 42
SLICE_AXIS = 2
USE_MINMAX = True

ED_LABEL = 2
AT_LABEL = 4

BASE_CH = 32
#BATCH_SIZE = 8
NUM_WORKERS = 0

EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5

THR_INC = 0.5
THR_DEC = 0.5

# EMA 설정
USE_EMA = True
EMA_DECAY = 0.995

# loss weight
LAMBDA_DICE = 1.0
LAMBDA_TVERSKY = 0.5
LAMBDA_FOCAL_AT = 0.5
LAMBDA_OVERLAP = 0.1

# BCE pos_weight: 기존 실험처럼 sparse change를 고려
# 너무 크면 overprediction이 심해질 수 있어서 일단 50/50/100/100 권장
POS_WEIGHT = {
    "ED_INC": 50.0,
    "ED_DEC": 50.0,
    "AT_INC": 100.0,
    "AT_DEC": 100.0,
}


# ============================================================
# 1. 기본 유틸
# ============================================================

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_patients(root, patient_range=(100001, 100302)):
    pids = []

    for name in os.listdir(root):
        if re.fullmatch(r"\d{6}", name):
            pid = int(name)
            if patient_range[0] <= pid <= patient_range[1]:
                pids.append(pid)

    pids.sort()
    return pids


def split_patients(pids, seed=42, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.RandomState(seed)
    pids = np.array(pids, dtype=int)
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = pids[:n_train].tolist()
    val = pids[n_train:n_train + n_val].tolist()
    test = pids[n_train + n_val:].tolist()

    return train, val, test


def load_ckpt_if_exists(model, ckpt_path, device):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("[Start] Training from scratch")
        return model

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        elif "ema_state_dict" in state:
            state = state["ema_state_dict"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    print(f"[Start] Loaded checkpoint: {ckpt_path}")
    return model


def dice_binary_torch(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    den = pred_bin.sum() + gt_bin.sum()
    return float(((2.0 * inter + eps) / (den + eps)).detach().cpu().item())


# ============================================================
# 2. Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        dims = (0, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        den = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        dims = (0, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - tversky.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = self.alpha * (1.0 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def overlap_penalty(logits):
    """
    같은 branch 안에서 INC와 DEC가 동시에 높아지는 것을 억제.
    logits: (B,2,H,W)
    """
    probs = torch.sigmoid(logits)
    inc = probs[:, 0:1]
    dec = probs[:, 1:2]
    return (inc * dec).mean()


def build_loss_functions(device):
    pos_w = torch.tensor(
        [
            POS_WEIGHT["ED_INC"],
            POS_WEIGHT["ED_DEC"],
            POS_WEIGHT["AT_INC"],
            POS_WEIGHT["AT_DEC"],
        ],
        dtype=torch.float32,
        device=device,
    )

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w.view(1, 4, 1, 1))
    dice = DiceLoss()
    tversky = TverskyLoss(alpha=0.3, beta=0.7)
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    return bce, dice, tversky, focal


def compute_loss(out_ed, out_at, y, bce, dice, tversky, focal):
    """
    out_ed: (B,2,H,W) = ED_INC, ED_DEC
    out_at: (B,2,H,W) = AT_INC, AT_DEC
    y:      (B,4,H,W) = ED_INC, ED_DEC, AT_INC, AT_DEC
    """
    logits4 = torch.cat([out_ed, out_at], dim=1)

    y_ed = y[:, 0:2]
    y_at = y[:, 2:4]

    loss_bce = bce(logits4, y)
    loss_dice = dice(logits4, y)

    loss_tversky_ed = tversky(out_ed, y_ed)
    loss_tversky_at = tversky(out_at, y_at)

    loss_focal_at = focal(out_at, y_at)

    loss_overlap = overlap_penalty(out_ed) + overlap_penalty(out_at)

    total = (
        loss_bce
        + LAMBDA_DICE * loss_dice
        + LAMBDA_TVERSKY * (loss_tversky_ed + loss_tversky_at)
        + LAMBDA_FOCAL_AT * loss_focal_at
        + LAMBDA_OVERLAP * loss_overlap
    )

    parts = {
        "bce": float(loss_bce.detach().cpu().item()),
        "dice": float(loss_dice.detach().cpu().item()),
        "tversky_ed": float(loss_tversky_ed.detach().cpu().item()),
        "tversky_at": float(loss_tversky_at.detach().cpu().item()),
        "focal_at": float(loss_focal_at.detach().cpu().item()),
        "overlap": float(loss_overlap.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }

    return total, parts


# ============================================================
# 3. EMA
# ============================================================

class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(),
                    alpha=1.0 - self.decay,
                )

    def apply_to(self, model):
        self.backup = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name in self.backup:
                param.data.copy_(self.backup[name].data)

        self.backup = {}

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


# ============================================================
# 4. Sampler / Eval
# ============================================================

def build_weighted_sampler(ds):
    """
    change slice를 더 자주 보게 하는 sampler.
    has_ed_change / has_at_change는 ds.index에 있음.
    """
    weights = []

    for item in ds.index:
        _, _, has_ed_change, has_at_change = item

        w = 1.0
        if has_ed_change:
            w += 2.0
        if has_at_change:
            w += 4.0

        weights.append(w)

    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler


@torch.no_grad()
def evaluate_batch_level(model, loader, device, thr_inc=0.5, thr_dec=0.5):
    model.eval()

    delta_ed_inc = []
    delta_ed_dec = []
    delta_at_inc = []
    delta_at_dec = []

    recon_ed = []
    recon_at = []

    for x, y, t1m, t2m in loader:
        x = x.to(device)
        y = y.to(device)
        t1m = t1m.to(device)
        t2m = t2m.to(device)

        out_ed, out_at = model(x)

        p_ed_inc = (torch.sigmoid(out_ed[:, 0:1]) >= thr_inc).float()
        p_ed_dec = (torch.sigmoid(out_ed[:, 1:2]) >= thr_dec).float()
        p_at_inc = (torch.sigmoid(out_at[:, 0:1]) >= thr_inc).float()
        p_at_dec = (torch.sigmoid(out_at[:, 1:2]) >= thr_dec).float()

        gt_ed_inc = y[:, 0:1]
        gt_ed_dec = y[:, 1:2]
        gt_at_inc = y[:, 2:3]
        gt_at_dec = y[:, 3:4]

        delta_ed_inc.append(dice_binary_torch(p_ed_inc, gt_ed_inc))
        delta_ed_dec.append(dice_binary_torch(p_ed_dec, gt_ed_dec))
        delta_at_inc.append(dice_binary_torch(p_at_inc, gt_at_inc))
        delta_at_dec.append(dice_binary_torch(p_at_dec, gt_at_dec))

        t1_ed = t1m[:, 0:1]
        t1_at = t1m[:, 1:2]
        t2_ed = t2m[:, 0:1]
        t2_at = t2m[:, 1:2]

        t2p_ed = torch.clamp(t1_ed + p_ed_inc, 0, 1) * (1.0 - p_ed_dec)
        t2p_at = torch.clamp(t1_at + p_at_inc, 0, 1) * (1.0 - p_at_dec)

        t2p_ed = (t2p_ed > 0.5).float()
        t2p_at = (t2p_at > 0.5).float()

        recon_ed.append(dice_binary_torch(t2p_ed, t2_ed))
        recon_at.append(dice_binary_torch(t2p_at, t2_at))

    def m(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    rep = {
        "Delta": {
            "ED_INC": m(delta_ed_inc),
            "ED_DEC": m(delta_ed_dec),
            "AT_INC": m(delta_at_inc),
            "AT_DEC": m(delta_at_dec),
            "Mean": float(np.mean([
                m(delta_ed_inc),
                m(delta_ed_dec),
                m(delta_at_inc),
                m(delta_at_dec),
            ])),
        },
        "Recon_T2": {
            "ED": m(recon_ed),
            "AT": m(recon_at),
            "Mean": float(np.mean([m(recon_ed), m(recon_at)])),
        },
    }

    return rep


# ============================================================
# 5. Main train
# ============================================================

def main():
    seed_all(SEED)
    device = get_device()

    print("===== DEVICE CHECK =====")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("selected device:", device)
    print("========================")

    pids = list_patients(ROOT)
    train_ids, val_ids, test_ids = split_patients(pids, seed=SEED)

    print(f"[Split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print("[Save best]", SAVE_BEST)
    print("[Save last]", SAVE_LAST)
    print("[EMA]", USE_EMA, "decay=", EMA_DECAY)

    train_ds = BrainChangeSliceDataset(
        root=ROOT,
        slice_axis=SLICE_AXIS,
        patient_ids_keep=train_ids,
        use_minmax=USE_MINMAX,
        cache_patients=1,
        ed_label=ED_LABEL,
        at_label=AT_LABEL,
    )
    train_ds.is_train = True

    val_ds = BrainChangeSliceDataset(
        root=ROOT,
        slice_axis=SLICE_AXIS,
        patient_ids_keep=val_ids,
        use_minmax=USE_MINMAX,
        cache_patients=1,
        ed_label=ED_LABEL,
        at_label=AT_LABEL,
    )
    val_ds.is_train = False

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"[Train] slices={len(train_ds)}, batches/epoch={len(train_loader)}")
    print(f"[Val] slices={len(val_ds)}, batches={len(val_loader)}")

    model = UNetNoPoolingDualDecoder(
        in_channels=4,
        base_channels=BASE_CH,
    ).to(device)

    model = load_ckpt_if_exists(model, START_CKPT, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.1,
    )

    bce, dice, tversky, focal = build_loss_functions(device)

    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    best_val_recon = -1.0
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        loss_meter = []
        part_meter = {
            "bce": [],
            "dice": [],
            "tversky_ed": [],
            "tversky_at": [],
            "focal_at": [],
            "overlap": [],
        }

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")

        for x, y, t1m, t2m in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            out_ed, out_at = model(x)

            loss, parts = compute_loss(
                out_ed=out_ed,
                out_at=out_at,
                y=y,
                bce=bce,
                dice=dice,
                tversky=tversky,
                focal=focal,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            loss_meter.append(float(loss.detach().cpu().item()))

            for k in part_meter:
                part_meter[k].append(parts[k])

            pbar.set_postfix({
                "loss": f"{np.mean(loss_meter):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        # ----------------------------------------------------
        # Validation with EMA weights
        # ----------------------------------------------------
        if ema is not None:
            ema.apply_to(model)

        val_rep = evaluate_batch_level(
            model=model,
            loader=val_loader,
            device=device,
            thr_inc=THR_INC,
            thr_dec=THR_DEC,
        )

        if ema is not None:
            ema.restore(model)

        val_delta = val_rep["Delta"]
        val_recon = val_rep["Recon_T2"]

        elapsed_min = (time.time() - start_time) / 60.0

        train_loss = float(np.mean(loss_meter))

        print(
            f"[{epoch}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} | "
            f"Val ΔDice Mean={val_delta['Mean']:.4f} "
            f"(ED_INC={val_delta['ED_INC']:.4f}, ED_DEC={val_delta['ED_DEC']:.4f}, "
            f"AT_INC={val_delta['AT_INC']:.4f}, AT_DEC={val_delta['AT_DEC']:.4f}) | "
            f"Val Recon(T2) Mean={val_recon['Mean']:.4f} "
            f"(ED={val_recon['ED']:.4f}, AT={val_recon['AT']:.4f}) | "
            f"time={elapsed_min:.1f} min"
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "loss_parts": {
                k: float(np.mean(v)) if len(v) else 0.0
                for k, v in part_meter.items()
            },
            "val": val_rep,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_min": elapsed_min,
        }

        history.append(epoch_log)

        # ----------------------------------------------------
        # Save best based on EMA validation Recon(T2)
        # ----------------------------------------------------
        current_val_recon = val_recon["Mean"]

        if current_val_recon > best_val_recon:
            best_val_recon = current_val_recon

            save_obj = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "best_val_recon": best_val_recon,
                "val_report": val_rep,
                "config": {
                    "root": ROOT,
                    "seed": SEED,
                    "slice_axis": SLICE_AXIS,
                    "use_minmax": USE_MINMAX,
                    "base_channels": BASE_CH,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "ema": USE_EMA,
                    "ema_decay": EMA_DECAY,
                    "start_ckpt": START_CKPT,
                    "thr_inc": THR_INC,
                    "thr_dec": THR_DEC,
                    "pos_weight": POS_WEIGHT,
                },
                "split": {
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids": test_ids,
                },
            }

            torch.save(save_obj, SAVE_BEST)

            print(
                f"  ✅ Saved best EMA model: {SAVE_BEST} "
                f"(epoch={epoch}, best Val Recon Mean={best_val_recon:.4f})"
            )

        # save last
        save_last_obj = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "best_val_recon": best_val_recon,
            "history": history,
            "config": {
                "root": ROOT,
                "seed": SEED,
                "slice_axis": SLICE_AXIS,
                "use_minmax": USE_MINMAX,
                "base_channels": BASE_CH,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "ema": USE_EMA,
                "ema_decay": EMA_DECAY,
                "start_ckpt": START_CKPT,
                "thr_inc": THR_INC,
                "thr_dec": THR_DEC,
                "pos_weight": POS_WEIGHT,
            },
            "split": {
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
        }

        torch.save(save_last_obj, SAVE_LAST)

        with open(OUT_LOG, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": history,
                    "best_val_recon": best_val_recon,
                    "save_best": SAVE_BEST,
                    "save_last": SAVE_LAST,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    total_hour = (time.time() - start_time) / 3600.0

    print("\nTraining finished.")
    print(f"Best Val Recon(T2) Mean: {best_val_recon:.4f}")
    print(f"Total training time: {total_hour:.2f} hours")
    print("Best checkpoint:", SAVE_BEST)
    print("Last checkpoint:", SAVE_LAST)
    print("Train log:", OUT_LOG)


if __name__ == "__main__":
    main()