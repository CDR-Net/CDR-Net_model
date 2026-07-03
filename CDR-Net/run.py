# run.py
import argparse
from train import run_training

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default=r"/Users/amy/PycharmProjects/labresearch/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--slice_axis", type=int, default=2)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--use_minmax", action="store_true", default=True)
    ap.add_argument("--save_path", type=str, default="best_change_seg_incdec.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--train_patch_size", type=int, default=192)

    # imbalance/sampling
    ap.add_argument("--sampler_w_at", type=float, default=10.0)
    ap.add_argument("--sampler_w_ed", type=float, default=3.0)
    ap.add_argument("--train_at_center_prob", type=float, default=0.70)
    ap.add_argument("--w_at", type=float, default=2.0)

    # loss stability
    ap.add_argument("--pos_weight_cap", type=float, default=100.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--overlap_lambda", type=float, default=0.1)

    # labels
    ap.add_argument("--ed_label", type=int, default=2)
    ap.add_argument("--at_label", type=int, default=4)

    # eval thresholds
    ap.add_argument("--thr_inc", type=float, default=0.5)
    ap.add_argument("--thr_dec", type=float, default=0.5)

    args = ap.parse_args()
    run_training(**vars(args))

if __name__ == "__main__":
    main()
