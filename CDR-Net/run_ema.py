# run_ema.py
import argparse
from train_ema import run_training_ema


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default=r"/Users/amy/PycharmProjects/labresearch/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0")
    ap.add_argument("--start_ckpt", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--slice_axis", type=int, default=2)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--use_minmax", action="store_true", default=True)
    ap.add_argument("--save_best", type=str, default="best_change_seg_incdec_ema_bs4.pt")
    ap.add_argument("--save_last", type=str, default="last_change_seg_incdec_ema_bs4.pt")
    ap.add_argument("--out_log", type=str, default="train_log_cdrnet_ema_bs4.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--thr_inc", type=float, default=0.5)
    ap.add_argument("--thr_dec", type=float, default=0.5)

    # EMA
    ap.add_argument("--ema_decay", type=float, default=0.995)

    # loss weights
    ap.add_argument("--lambda_dice", type=float, default=1.0)
    ap.add_argument("--lambda_tversky", type=float, default=0.5)
    ap.add_argument("--lambda_focal_at", type=float, default=0.5)
    ap.add_argument("--lambda_overlap", type=float, default=0.1)

    # pos_weight per channel
    ap.add_argument("--pos_weight_ed_inc", type=float, default=50.0)
    ap.add_argument("--pos_weight_ed_dec", type=float, default=50.0)
    ap.add_argument("--pos_weight_at_inc", type=float, default=100.0)
    ap.add_argument("--pos_weight_at_dec", type=float, default=100.0)

    # labels
    ap.add_argument("--ed_label", type=int, default=2)
    ap.add_argument("--at_label", type=int, default=4)

    args = ap.parse_args()
    run_training_ema(**vars(args))


if __name__ == "__main__":
    main()
