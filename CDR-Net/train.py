# train.py
import os
import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import BrainChangeSliceDataset
from model import UNetNoPoolingDualDecoder
from utils import seed_all, DiceLoss, TverskyLoss, FocalLoss, eval_delta_and_recon


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        info = f"cuda | GPU={torch.cuda.get_device_name(0)}"
        return dev, info
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        return dev, "mps"
    return torch.device("cpu"), "cpu"


def print_device_info(device, info):
    print("===== DEVICE CHECK =====")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU[0]:", torch.cuda.get_device_name(0))
    print("selected device:", device, "|", info)
    print("========================")


def _list_patients(root, patient_range=(100001, 100302)):
    pids = []
    for name in os.listdir(root):
        if re.fullmatch(r"\d{6}", name):
            pid = int(name)
            if patient_range[0] <= pid <= patient_range[1]:
                pids.append(pid)
    pids.sort()
    return pids


def _split_patients(pids, seed=42, train_ratio=0.8, val_ratio=0.1):
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


def _compute_pos_weight(pos_pixels: int, total_pixels: int, cap: float = 100.0):
    if pos_pixels <= 0:
        return 1.0
    neg = max(0, total_pixels - pos_pixels)
    w = float(neg) / float(pos_pixels)
    return float(min(max(w, 1.0), cap))


def run_training(
    root,
    epochs=20,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    slice_axis=2,
    base_channels=32,
    use_minmax=True,
    save_path="best_change_seg_incdec.pt",
    seed=42,
    num_workers=0,

    train_patch_size=192,

    # imbalance/sampling
    sampler_w_at=10.0,
    sampler_w_ed=3.0,
    train_at_center_prob=0.70,
    w_at=2.0,

    pos_weight_cap=100.0,
    grad_clip=1.0,

    # inc/dec overlap penalty (선택)
    overlap_lambda=0.1,

    # label values
    ed_label=2,
    at_label=4,

    # eval thresholds
    thr_inc=0.5,
    thr_dec=0.5,
):
    seed_all(seed)

    device, info = get_device()
    print_device_info(device, info)

    # split (환자 단위)
    pids = _list_patients(root)
    train_ids, val_ids, test_ids = _split_patients(pids, seed=seed)

    # Dataset
    train_ds = BrainChangeSliceDataset(
        root=root,
        slice_axis=slice_axis,
        patient_ids_keep=train_ids,
        use_minmax=use_minmax,
        train_patch_size=train_patch_size,
        train_pos_center_prob=0.95,
        train_at_center_prob=train_at_center_prob,
        cache_patients=2,
        ed_label=ed_label,
        at_label=at_label,
    )
    val_ds = BrainChangeSliceDataset(
        root=root,
        slice_axis=slice_axis,
        patient_ids_keep=val_ids,
        use_minmax=use_minmax,
        cache_patients=1,
        ed_label=ed_label,
        at_label=at_label,
    )

    train_ds.is_train = True
    val_ds.is_train = False

    # 통계 출력
    tr_stat = train_ds.get_imbalance_stats()
    va_stat = val_ds.get_imbalance_stats()

    print("========== Dataset imbalance stats (Δ INC/DEC GT) ==========")
    print(f"[GT] ED_label={ed_label} | AT_label={at_label}")
    print(f"[Train] total_slices={tr_stat['total_slices']}, total_pixels={tr_stat['total_pixels']}")
    print("[Train] pos_slices:", tr_stat["pos_slices"])
    print("[Train] pos_slice_ratio(%):", {k: v*100 for k, v in tr_stat["pos_slice_ratio"].items()})
    print("[Train] pos_pixel_ratio(%):", {k: v*100 for k, v in tr_stat["pos_pixel_ratio"].items()})
    print(f"[Val]   total_slices={va_stat['total_slices']}, total_pixels={va_stat['total_pixels']}")
    print("[Val] pos_slice_ratio(%):", {k: v*100 for k, v in va_stat["pos_slice_ratio"].items()})
    print("===========================================================")

    # 채널별 pos_weight (cap 적용)
    # ch0 ED_INC, ch1 ED_DEC, ch2 AT_INC, ch3 AT_DEC
    posw = {}
    for k in ["ED_INC", "ED_DEC", "AT_INC", "AT_DEC"]:
        pos_pixels = tr_stat["pos_pixels"][k]
        total_pixels = tr_stat["total_pixels"]
        posw[k] = _compute_pos_weight(pos_pixels, total_pixels, cap=pos_weight_cap)

    print(f"[Loss] BCE pos_weight cap={pos_weight_cap} -> {posw}")

    # Sampler: AT change(inc/dec any) > ED change(inc/dec any)
    weights = []
    for (_, _, has_ed_change, has_at_change) in train_ds.index:
        w = 1.0
        if has_at_change:
            w *= float(sampler_w_at)
        elif has_ed_change:
            w *= float(sampler_w_ed)
        weights.append(w)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = UNetNoPoolingDualDecoder(
        in_channels=4,
        base_channels=base_channels,
    ).to(device)

    # Loss building blocks
    dice = DiceLoss()
    # (권장) inc/dec는 둘 다 희소하므로 BCE(pos_weight) + (선택) focal + tversky 조합 유지
    # ED
    bce_ed_inc = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw["ED_INC"]], device=device))
    bce_ed_dec = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw["ED_DEC"]], device=device))
    tversky_ed = TverskyLoss(alpha=0.3, beta=0.7)
    # AT
    bce_at_inc = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw["AT_INC"]], device=device))
    bce_at_dec = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([posw["AT_DEC"]], device=device))
    focal_at = FocalLoss(gamma=2.0, alpha=0.75)
    tversky_at = TverskyLoss(alpha=0.5, beta=0.5)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_score = -1.0

    print(f"[Split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} patients")
    print(f"[Train] steps/epoch = {len(train_loader)} (len(train_ds)={len(train_ds)}, batch={batch_size})")
    print(f"[Train] sampler_w_at={sampler_w_at}, sampler_w_ed={sampler_w_ed}, train_at_center_prob={train_at_center_prob}, w_at={w_at}")
    print(f"[Eval] thr_inc={thr_inc}, thr_dec={thr_dec} | overlap_lambda={overlap_lambda}")

    # Train loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y, t1m, t2m in tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]"):
            x = x.to(device)
            y = y.to(device)  # (B,4,H,W)

            # GT
            y_ed_inc = y[:, 0:1]
            y_ed_dec = y[:, 1:2]
            y_at_inc = y[:, 2:3]
            y_at_dec = y[:, 3:4]

            out_ed, out_at = model(x)  # (B,2,H,W), (B,2,H,W)
            out_ed_inc = out_ed[:, 0:1]
            out_ed_dec = out_ed[:, 1:2]
            out_at_inc = out_at[:, 0:1]
            out_at_dec = out_at[:, 1:2]

            # ED loss
            loss_ed = (
                bce_ed_inc(out_ed_inc, y_ed_inc) + dice(out_ed_inc, y_ed_inc) + tversky_ed(out_ed_inc, y_ed_inc)
                + bce_ed_dec(out_ed_dec, y_ed_dec) + dice(out_ed_dec, y_ed_dec) + tversky_ed(out_ed_dec, y_ed_dec)
            )

            # AT loss (inc/dec 각각)
            loss_at = (
                bce_at_inc(out_at_inc, y_at_inc) + focal_at(out_at_inc, y_at_inc) + dice(out_at_inc, y_at_inc) + tversky_at(out_at_inc, y_at_inc)
                + bce_at_dec(out_at_dec, y_at_dec) + focal_at(out_at_dec, y_at_dec) + dice(out_at_dec, y_at_dec) + tversky_at(out_at_dec, y_at_dec)
            )

            # overlap penalty (같은 픽셀을 inc & dec로 동시에 찍는 것 억제)
            p_ed_inc = torch.sigmoid(out_ed_inc)
            p_ed_dec = torch.sigmoid(out_ed_dec)
            p_at_inc = torch.sigmoid(out_at_inc)
            p_at_dec = torch.sigmoid(out_at_dec)
            overlap_pen = (p_ed_inc * p_ed_dec).mean() + (p_at_inc * p_at_dec).mean()

            loss = loss_ed + float(w_at) * loss_at + float(overlap_lambda) * overlap_pen

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))

        # Validation: Δ + Recon 둘 다
        val_report = eval_delta_and_recon(
            model, val_loader, device,
            thr_inc=thr_inc, thr_dec=thr_dec
        )

        d = val_report["Delta"]
        r = val_report["Recon_T2"]

        print(
            f"[{epoch}/{epochs}] train_loss={train_loss:.4f} | "
            f"Val ΔDice Mean={d['Mean']:.4f} "
            f"(ED_INC={d['ED_INC']:.4f}, ED_DEC={d['ED_DEC']:.4f}, AT_INC={d['AT_INC']:.4f}, AT_DEC={d['AT_DEC']:.4f}) | "
            f"Val Recon(T2) Mean={r['Mean']:.4f} (ED={r['ED']:.4f}, AT={r['AT']:.4f})"
        )

        # ✅ 논문용 “B 점수”를 best 기준으로 저장 (Recon mean)
        score = r["Mean"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best model: {save_path} (best Val Recon Mean={best_score:.4f})")

    print("Training finished.")
