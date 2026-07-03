# dataset.py
import os, re
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

from utils import find_nii, load_nii, minmax_norm, ensure_divisible_hw


class BrainChangeSliceDataset(Dataset):
    """
    ROOT/100001/ 같은 6자리 환자 폴더

    입력(4채널, time1만 사용):
      - time1_t1, time1_t1ce, time1_t2, time1_flair

    라벨(4채널): (INC/DEC 2채널 x (ED/AT))
      - y[0] = ΔED_INC (추가된 ED)
      - y[1] = ΔED_DEC (사라진 ED)
      - y[2] = ΔAT_INC (추가된 AT)
      - y[3] = ΔAT_DEC (사라진 AT)

    ✅ 핵심:
      - subtraction_seg를 GT로 쓰지 않음
      - time1_seg / time2_seg에서 ED/AT 마스크를 뽑아 INC/DEC 생성

    추가 반환(평가용):
      - t1_mask (2채널: ED/AT)
      - t2_mask (2채널: ED/AT)

    train: patch crop + flip aug
    val/test: 전체 슬라이스(단, 모델 입력 크기 divisible 패딩)
    """

    REQUIRED = [
        "time1_t1", "time1_t1ce", "time1_t2", "time1_flair",
        "time1_seg", "time2_seg",
    ]

    def __init__(
        self,
        root,
        slice_axis=2,
        patient_range=(100001, 100302),
        patient_ids_keep=None,
        use_minmax=True,
        cache_patients=2,
        strict_missing=True,

        # train crop/aug
        train_patch_size=192,
        train_pos_center_prob=0.95,
        train_at_center_prob=0.70,

        # label semantics
        ed_label=2,
        at_label=4,
    ):
        self.root = root
        self.slice_axis = int(slice_axis)
        self.use_minmax = bool(use_minmax)
        self.cache_patients = max(int(cache_patients), 0)
        self.strict_missing = bool(strict_missing)

        self.train_patch_size = int(train_patch_size)
        self.train_pos_center_prob = float(train_pos_center_prob)
        self.train_at_center_prob = float(train_at_center_prob)

        self.ed_label = int(ed_label)
        self.at_label = int(at_label)

        self.is_train = False

        # Δ(inc/dec) 통계 누적 (채널별)
        # ch0 ED_INC, ch1 ED_DEC, ch2 AT_INC, ch3 AT_DEC
        self._stat_total_slices = 0
        self._stat_pos_slices = np.zeros(4, dtype=np.int64)

        self._stat_total_pixels = 0
        self._stat_pos_pixels = np.zeros(4, dtype=np.int64)

        # 환자 폴더 수집
        all_patients = []
        for name in os.listdir(root):
            if re.fullmatch(r"\d{6}", name):
                pid = int(name)
                if patient_range[0] <= pid <= patient_range[1]:
                    all_patients.append(pid)
        all_patients.sort()

        if patient_ids_keep is not None:
            keep = set(patient_ids_keep)
            all_patients = [p for p in all_patients if p in keep]

        if not all_patients:
            raise RuntimeError("No patient folders found. Check root/range/keep list.")

        # index: (pid, d, has_ed_change, has_at_change)  # change는 inc/dec 둘 중 하나라도 있으면 True
        self.index = []
        self.case_map = {}  # pid -> (paths, vol_shape)

        for pid in all_patients:
            pdir = os.path.join(root, f"{pid}")

            mapping = {
                "time1_t1":    f"{pid}_time1_t1",
                "time1_t1ce":  f"{pid}_time1_t1ce",
                "time1_t2":    f"{pid}_time1_t2",
                "time1_flair": f"{pid}_time1_flair",
                "time1_seg":   f"{pid}_time1_seg",
                "time2_seg":   f"{pid}_time2_seg",
            }

            paths = {}
            missing = []
            for k, stem in mapping.items():
                fp = find_nii(pdir, stem)
                if fp is None:
                    missing.append(k)
                else:
                    paths[k] = fp

            if missing:
                if self.strict_missing:
                    raise FileNotFoundError(f"[{pid}] missing: {missing} in {pdir}")
                else:
                    continue

            vol_shape = nib.load(paths["time1_flair"]).shape  # (H,W,D) assumed
            D = vol_shape[self.slice_axis]
            self.case_map[pid] = (paths, vol_shape)

            # 통계/플래그를 위해 seg 1회 로드
            seg1 = load_nii(paths["time1_seg"]).astype(np.int16)
            seg2 = load_nii(paths["time2_seg"]).astype(np.int16)

            t1_ed = (seg1 == self.ed_label)
            t2_ed = (seg2 == self.ed_label)
            t1_at = (seg1 == self.at_label)
            t2_at = (seg2 == self.at_label)

            ed_inc = (t2_ed & (~t1_ed)).astype(np.uint8)
            ed_dec = (t1_ed & (~t2_ed)).astype(np.uint8)
            at_inc = (t2_at & (~t1_at)).astype(np.uint8)
            at_dec = (t1_at & (~t2_at)).astype(np.uint8)

            # 픽셀 통계(채널별)
            total_pixels = int(np.prod(ed_inc.shape))
            self._stat_total_pixels += total_pixels
            self._stat_pos_pixels[0] += int(ed_inc.sum())
            self._stat_pos_pixels[1] += int(ed_dec.sum())
            self._stat_pos_pixels[2] += int(at_inc.sum())
            self._stat_pos_pixels[3] += int(at_dec.sum())

            # 슬라이스별 플래그 & 슬라이스 통계
            for d in range(D):
                ed_inc_sl = self._get_slice(ed_inc, d)
                ed_dec_sl = self._get_slice(ed_dec, d)
                at_inc_sl = self._get_slice(at_inc, d)
                at_dec_sl = self._get_slice(at_dec, d)

                has_ed_change = bool((ed_inc_sl.sum() + ed_dec_sl.sum()) > 0)
                has_at_change = bool((at_inc_sl.sum() + at_dec_sl.sum()) > 0)

                self._stat_total_slices += 1
                self._stat_pos_slices[0] += int(ed_inc_sl.sum() > 0)
                self._stat_pos_slices[1] += int(ed_dec_sl.sum() > 0)
                self._stat_pos_slices[2] += int(at_inc_sl.sum() > 0)
                self._stat_pos_slices[3] += int(at_dec_sl.sum() > 0)

                self.index.append((pid, d, has_ed_change, has_at_change))

        if not self.index:
            raise RuntimeError("No slices indexed.")

        self._cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _get_slice(self, vol, d):
        if self.slice_axis == 0: return vol[d, :, :]
        if self.slice_axis == 1: return vol[:, d, :]
        return vol[:, :, d]

    def _load_patient_vols(self, pid):
        paths, _ = self.case_map[pid]
        return (
            load_nii(paths["time1_t1"]),
            load_nii(paths["time1_t1ce"]),
            load_nii(paths["time1_t2"]),
            load_nii(paths["time1_flair"]),
            load_nii(paths["time1_seg"]).astype(np.int16),
            load_nii(paths["time2_seg"]).astype(np.int16),
        )

    def _get_patient_vols_cached(self, pid):
        if self.cache_patients <= 0:
            return self._load_patient_vols(pid)
        vols = self._cache.get(pid)
        if vols is not None:
            self._cache.move_to_end(pid, last=True)
            return vols
        vols = self._load_patient_vols(pid)
        self._cache[pid] = vols
        while len(self._cache) > self.cache_patients:
            self._cache.popitem(last=False)
        return vols

    def _crop_patch(self, x, y, t1m, t2m, patch=192, prefer_mask=None):
        """
        x: (C,H,W)
        y: (4,H,W)
        t1m/t2m: (2,H,W)
        prefer_mask: (H,W) bool/0-1 array
        """
        C, H, W = x.shape
        ph = pw = int(patch)

        if H < ph or W < pw:
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            x = np.pad(x, ((0,0),(0,pad_h),(0,pad_w)), mode="edge")
            y = np.pad(y, ((0,0),(0,pad_h),(0,pad_w)), mode="constant")
            t1m = np.pad(t1m, ((0,0),(0,pad_h),(0,pad_w)), mode="constant")
            t2m = np.pad(t2m, ((0,0),(0,pad_h),(0,pad_w)), mode="constant")
            C, H, W = x.shape

        # 1) prefer_mask 중심 crop
        if prefer_mask is not None:
            ys, xs = np.where(prefer_mask)
            if len(xs) > 0:
                k = np.random.randint(len(xs))
                cx, cy = int(xs[k]), int(ys[k])
                jitter = patch // 8
                cx = int(np.clip(cx + np.random.randint(-jitter, jitter+1), 0, W-1))
                cy = int(np.clip(cy + np.random.randint(-jitter, jitter+1), 0, H-1))
                x0 = int(np.clip(cx - pw//2, 0, W - pw))
                y0 = int(np.clip(cy - ph//2, 0, H - ph))
                return (
                    x[:, y0:y0+ph, x0:x0+pw],
                    y[:, y0:y0+ph, x0:x0+pw],
                    t1m[:, y0:y0+ph, x0:x0+pw],
                    t2m[:, y0:y0+ph, x0:x0+pw],
                )

        # 2) fallback: union(ΔINC/ΔDEC 전체) 중심 crop
        union = (y.sum(axis=0) > 0)
        ys, xs = np.where(union)
        if len(xs) > 0:
            k = np.random.randint(len(xs))
            cx, cy = int(xs[k]), int(ys[k])
            jitter = patch // 8
            cx = int(np.clip(cx + np.random.randint(-jitter, jitter+1), 0, W-1))
            cy = int(np.clip(cy + np.random.randint(-jitter, jitter+1), 0, H-1))
            x0 = int(np.clip(cx - pw//2, 0, W - pw))
            y0 = int(np.clip(cy - ph//2, 0, H - ph))
            return (
                x[:, y0:y0+ph, x0:x0+pw],
                y[:, y0:y0+ph, x0:x0+pw],
                t1m[:, y0:y0+ph, x0:x0+pw],
                t2m[:, y0:y0+ph, x0:x0+pw],
            )

        # 3) fallback: random crop
        x0 = np.random.randint(0, W - pw + 1)
        y0 = np.random.randint(0, H - ph + 1)
        return (
            x[:, y0:y0+ph, x0:x0+pw],
            y[:, y0:y0+ph, x0:x0+pw],
            t1m[:, y0:y0+ph, x0:x0+pw],
            t2m[:, y0:y0+ph, x0:x0+pw],
        )

    def get_imbalance_stats(self):
        """
        Δ(inc/dec) 기준 통계 (채널별)
        """
        eps = 1e-9
        total_s = max(1, self._stat_total_slices)
        total_p = max(1, self._stat_total_pixels)

        ch_names = ["ED_INC", "ED_DEC", "AT_INC", "AT_DEC"]
        out = {
            "total_slices": int(total_s),
            "total_pixels": int(total_p),
            "pos_slices": {ch_names[i]: int(self._stat_pos_slices[i]) for i in range(4)},
            "pos_slice_ratio": {ch_names[i]: float(self._stat_pos_slices[i] / total_s) for i in range(4)},
            "pos_pixels": {ch_names[i]: int(self._stat_pos_pixels[i]) for i in range(4)},
            "pos_pixel_ratio": {ch_names[i]: float(self._stat_pos_pixels[i] / (total_p + eps)) for i in range(4)},
        }
        return out

    def __getitem__(self, i):
        pid, d, has_ed_change, has_at_change = self.index[i]
        t1, t1ce, t2, flair, seg1, seg2 = self._get_patient_vols_cached(pid)

        xs = [
            self._get_slice(t1, d),
            self._get_slice(t1ce, d),
            self._get_slice(t2, d),
            self._get_slice(flair, d),
        ]
        if self.use_minmax:
            xs = [minmax_norm(x) for x in xs]
        x = np.stack(xs, 0).astype(np.float32)  # (4,H,W)

        seg1_sl = self._get_slice(seg1, d)
        seg2_sl = self._get_slice(seg2, d)

        # t1/t2 masks for B eval (2 channels: ED/AT)
        t1_ed = (seg1_sl == self.ed_label).astype(np.float32)
        t2_ed = (seg2_sl == self.ed_label).astype(np.float32)
        t1_at = (seg1_sl == self.at_label).astype(np.float32)
        t2_at = (seg2_sl == self.at_label).astype(np.float32)

        t1_mask = np.stack([t1_ed, t1_at], 0).astype(np.float32)  # (2,H,W)
        t2_mask = np.stack([t2_ed, t2_at], 0).astype(np.float32)  # (2,H,W)

        # ΔINC/ΔDEC GT (4 channels)
        ed_inc = ((t2_ed > 0) & ~(t1_ed > 0)).astype(np.float32)
        ed_dec = ((t1_ed > 0) & ~(t2_ed > 0)).astype(np.float32)
        at_inc = ((t2_at > 0) & ~(t1_at > 0)).astype(np.float32)
        at_dec = ((t1_at > 0) & ~(t2_at > 0)).astype(np.float32)
        y = np.stack([ed_inc, ed_dec, at_inc, at_dec], 0).astype(np.float32)  # (4,H,W)

        if getattr(self, "is_train", False):
            do_pos_center = (has_ed_change or has_at_change) and (np.random.rand() < self.train_pos_center_prob)

            prefer_mask = None
            if do_pos_center:
                # AT change 있으면 일정 확률로 AT 중심(inc/dec union)
                if has_at_change and (np.random.rand() < self.train_at_center_prob):
                    prefer_mask = ((y[2] > 0) | (y[3] > 0))
                elif has_ed_change:
                    prefer_mask = ((y[0] > 0) | (y[1] > 0))

            x, y, t1_mask, t2_mask = self._crop_patch(
                x, y, t1_mask, t2_mask,
                patch=self.train_patch_size,
                prefer_mask=prefer_mask
            )

            # flip aug
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=2).copy()
                t1_mask = np.flip(t1_mask, axis=2).copy()
                t2_mask = np.flip(t2_mask, axis=2).copy()
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
                t1_mask = np.flip(t1_mask, axis=1).copy()
                t2_mask = np.flip(t2_mask, axis=1).copy()

        x = ensure_divisible_hw(x, 16)
        y = ensure_divisible_hw(y, 16)
        t1_mask = ensure_divisible_hw(t1_mask, 16)
        t2_mask = ensure_divisible_hw(t2_mask, 16)

        return (
            torch.from_numpy(x),         # (4,H,W)
            torch.from_numpy(y),         # (4,H,W)
            torch.from_numpy(t1_mask),   # (2,H,W)
            torch.from_numpy(t2_mask),   # (2,H,W)
        )
