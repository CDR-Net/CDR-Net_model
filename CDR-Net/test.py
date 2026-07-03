# test.py
import os, re, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BrainChangeSliceDataset
from model import UNetNoPoolingDualDecoder
from utils import eval_delta_and_recon, seed_all


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # ======= 설정 =======
    ROOT = r"/Users/amy/PycharmProjects/labresearch/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
    CKPT = r"/Users/amy/PycharmProjects/labresearch/CDR-Net/best_change_seg_incdec.pt"

    seed = 42
    slice_axis = 2
    base_channels = 32
    use_minmax = True
    num_workers = 0

    ed_label = 2
    at_label = 4

    thr_inc = 0.5
    thr_dec = 0.5

    batch_size = 8
    # ===================

    seed_all(seed)
    device = get_device()
    print("Device:", device)

    # patient-wise split (train.py와 동일)
    pids = _list_patients(ROOT)
    train_ids, val_ids, test_ids = _split_patients(pids, seed=seed)
    print(f"[Split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    # test dataset/loader
    test_ds = BrainChangeSliceDataset(
        root=ROOT,
        slice_axis=slice_axis,
        patient_ids_keep=test_ids,
        use_minmax=use_minmax,
        cache_patients=1,
        ed_label=ed_label,
        at_label=at_label,
    )
    test_ds.is_train = False

    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # model
    model = UNetNoPoolingDualDecoder(
        in_channels=4,
        base_channels=base_channels,
    ).to(device)

    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    print("Loaded:", CKPT)

    # eval on test
    report = eval_delta_and_recon(
        model, test_loader, device,
        thr_inc=thr_inc, thr_dec=thr_dec
    )

    print("\n==================== TEST REPORT ====================")
    d = report["Delta"]
    r = report["Recon_T2"]
    print(f"[Test ΔDice] Mean={d['Mean']:.4f} | "
          f"ED_INC={d['ED_INC']:.4f}, ED_DEC={d['ED_DEC']:.4f}, "
          f"AT_INC={d['AT_INC']:.4f}, AT_DEC={d['AT_DEC']:.4f}")
    print(f"[Test Recon(T2)] Mean={r['Mean']:.4f} | ED={r['ED']:.4f}, AT={r['AT']:.4f}")
    print("=====================================================\n")

    # save json
    out_path = os.path.join(os.path.dirname(CKPT), "../test_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
