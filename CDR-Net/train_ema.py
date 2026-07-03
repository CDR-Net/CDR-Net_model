# train_ema.py

import os
import re
import json
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import BrainChangeSliceDataset
from model import UNetNoPoolingDualDecoder
from utils import seed_all, DiceLoss, TverskyLoss, FocalLoss, eval_delta_and_recon


# ============================================================
# Utils
# ============================================================

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
    return pids[:n_train].tolist(), pids[n_train:n_train + n_val].tolist(), pids[n_train + n_val:].tolist()


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
    new_state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)
    print(f"[Start] Loaded checkpoint: {ckpt_path}")
    return model


def overlap_penalty(logits):
    probs = torch.sigmoid(logits)
    return (probs[:, 0:1] * probs[:, 1:2]).mean()


def build_weighted_sampler(ds):
    weights = []
    for _, _, has_ed_change, has_at_change in ds.index:
        w = 1.0
        if has_ed_change:
            w += 2.0
        if has_at_change:
            w += 4.0
        weights.append(w)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


# ============================================================
# EMA
# ============================================================

class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


# ============================================================
# Main training function
# ============================================================

def run_training_ema(
    root,
    start_ckpt=None,
    epochs=20,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-5,
    slice_axis=2,
    base_channels=32,
    use_minmax=True,
    save_best="best_change_seg_incdec_ema_bs4.pt",
    save_last="last_change_seg_incdec_ema_bs4.pt",
    out_log="train_log_cdrnet_ema_bs4.json",
    seed=42,
    num_workers=0,
    thr_inc=0.5,
    thr_dec=0.5,
    ema_decay=0.995,
    lambda_dice=1.0,
    lambda_tversky=0.5,
    lambda_focal_at=0.5,
    lambda_overlap=0.1,
    pos_weight_ed_inc=50.0,
    pos_weight_ed_dec=50.0,
    pos_weight_at_inc=100.0,
    pos_weight_at_dec=100.0,
    ed_label=2,
    at_label=4,
):
    seed_all(seed)
    device = get_device()

    print("===== DEVICE CHECK =====")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("selected device:", device)
    print("========================")

    pids = list_patients(root)
    train_ids, val_ids, test_ids = split_patients(pids, seed=seed)
    print(f"[Split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"[Save best] {save_best} | [Save last] {save_last} | [EMA] decay={ema_decay}")

    train_ds = BrainChangeSliceDataset(
        root=root, slice_axis=slice_axis, patient_ids_keep=train_ids,
        use_minmax=use_minmax, cache_patients=1, ed_label=ed_label, at_label=at_label,
    )
    train_ds.is_train = True

    val_ds = BrainChangeSliceDataset(
        root=root, slice_axis=slice_axis, patient_ids_keep=val_ids,
        use_minmax=use_minmax, cache_patients=1, ed_label=ed_label, at_label=at_label,
    )
    val_ds.is_train = False

    sampler = build_weighted_sampler(train_ds)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    print(f"[Train] slices={len(train_ds)}, batches/epoch={len(train_loader)}")
    print(f"[Val]   slices={len(val_ds)}, batches={len(val_loader)}")

    model = UNetNoPoolingDualDecoder(in_channels=4, base_channels=base_channels).to(device)
    model = load_ckpt_if_exists(model, start_ckpt, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    pos_w = torch.tensor(
        [pos_weight_ed_inc, pos_weight_ed_dec, pos_weight_at_inc, pos_weight_at_dec],
        dtype=torch.float32, device=device,
    )
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w.view(1, 4, 1, 1))
    dice_fn = DiceLoss()
    tversky_fn = TverskyLoss(alpha=0.3, beta=0.7)
    focal_fn = FocalLoss(gamma=2.0, alpha=0.25)

    ema = ModelEMA(model, decay=ema_decay)

    best_val_recon = -1.0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = []
        part_meter = {k: [] for k in ["bce", "dice", "tversky_ed", "tversky_at", "focal_at", "overlap"]}

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]")
        for x, y, t1m, t2m in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out_ed, out_at = model(x)

            logits4 = torch.cat([out_ed, out_at], dim=1)
            y_ed, y_at = y[:, 0:2], y[:, 2:4]

            loss_bce = bce_fn(logits4, y)
            loss_dice = dice_fn(logits4, y)
            loss_tversky_ed = tversky_fn(out_ed, y_ed)
            loss_tversky_at = tversky_fn(out_at, y_at)
            loss_focal_at = focal_fn(out_at, y_at)
            loss_overlap = overlap_penalty(out_ed) + overlap_penalty(out_at)

            loss = (
                loss_bce
                + lambda_dice * loss_dice
                + lambda_tversky * (loss_tversky_ed + loss_tversky_at)
                + lambda_focal_at * loss_focal_at
                + lambda_overlap * loss_overlap
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)

            loss_meter.append(float(loss.detach().cpu().item()))
            part_meter["bce"].append(float(loss_bce.detach().cpu().item()))
            part_meter["dice"].append(float(loss_dice.detach().cpu().item()))
            part_meter["tversky_ed"].append(float(loss_tversky_ed.detach().cpu().item()))
            part_meter["tversky_at"].append(float(loss_tversky_at.detach().cpu().item()))
            part_meter["focal_at"].append(float(loss_focal_at.detach().cpu().item()))
            part_meter["overlap"].append(float(loss_overlap.detach().cpu().item()))

            pbar.set_postfix({"loss": f"{np.mean(loss_meter):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step()

        ema.apply_to(model)
        val_rep = eval_delta_and_recon(model, val_loader, device, thr_inc=thr_inc, thr_dec=thr_dec)
        ema.restore(model)

        val_delta = val_rep["Delta"]
        val_recon = val_rep["Recon_T2"]
        elapsed_min = (time.time() - start_time) / 60.0
        train_loss = float(np.mean(loss_meter))

        print(
            f"[{epoch}/{epochs}] train_loss={train_loss:.4f} | "
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
            "loss_parts": {k: float(np.mean(v)) if v else 0.0 for k, v in part_meter.items()},
            "val": val_rep,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_min": elapsed_min,
        }
        history.append(epoch_log)

        current_val_recon = val_recon["Mean"]
        if current_val_recon > best_val_recon:
            best_val_recon = current_val_recon
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "best_val_recon": best_val_recon,
                "val_report": val_rep,
                "config": {
                    "root": root, "seed": seed, "slice_axis": slice_axis,
                    "use_minmax": use_minmax, "base_channels": base_channels,
                    "batch_size": batch_size, "epochs": epochs,
                    "lr": lr, "weight_decay": weight_decay,
                    "ema_decay": ema_decay, "start_ckpt": start_ckpt,
                    "thr_inc": thr_inc, "thr_dec": thr_dec,
                    "pos_weight": {
                        "ED_INC": pos_weight_ed_inc, "ED_DEC": pos_weight_ed_dec,
                        "AT_INC": pos_weight_at_inc, "AT_DEC": pos_weight_at_dec,
                    },
                },
                "split": {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids},
            }, save_best)
            print(f"  ✅ Saved best EMA model: {save_best} (epoch={epoch}, Val Recon Mean={best_val_recon:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "best_val_recon": best_val_recon,
            "history": history,
        }, save_last)

        with open(out_log, "w", encoding="utf-8") as f:
            json.dump({"history": history, "best_val_recon": best_val_recon,
                       "save_best": save_best, "save_last": save_last}, f, ensure_ascii=False, indent=2)

    total_hour = (time.time() - start_time) / 3600.0
    print(f"\nTraining finished. Best Val Recon(T2) Mean: {best_val_recon:.4f}")
    print(f"Total training time: {total_hour:.2f} hours")
    print(f"Best checkpoint: {save_best} | Last checkpoint: {save_last} | Train log: {out_log}")
