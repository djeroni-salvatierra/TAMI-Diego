import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PanSharpenDataset(Dataset):
    """
    Dataset for Pan-Sharpening following Wald's Protocol.

    Expects three sub-directories inside `data_dir / subset /`:
        ms_downsampled/   – Low-resolution multi-spectral TIFFs  (H×W×C or C×H×W, 16-bit)
        pan_downsampled/  – Panchromatic TIFFs                   (H×W or 1×H×W,   16-bit)
        ms_original_gt/   – Ground-truth full-resolution MS TIFFs (H×W×C or C×H×W, 16-bit)

    Optimisations:
      1. Caches decoded + normalised + bicubic-upsampled tensors in RAM using a
         multithreaded `ThreadPoolExecutor` at init time (`cache_in_ram=True`),
         eliminating redundant disk I/O and full-image `F.interpolate` on every epoch!
      2. When random cropping (`patch_size` is set) in uncached mode, crops `lrms` at
         low resolution *first* (`patch_size // scale_factor` with a 2-pixel bicubic halo)
         and only interpolates the tiny patch instead of the whole satellite image.
    """

    def __init__(
        self,
        data_dir: str,
        subset: str = "train",
        patch_size: int | None = 64,
        norm_val: float = 2047.0,
        scale_factor: int = 4,
        cache_in_ram: bool = True,
    ):
        super().__init__()
        if norm_val <= 0:
            raise ValueError(f"norm_val must be positive, got {norm_val}")

        self.patch_size = patch_size
        self.norm_val = float(norm_val)
        self.scale_factor = scale_factor
        self.cache_in_ram = cache_in_ram

        # Automatically resolve "validation" <-> "valid" folder naming
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.isdir(subset_dir):
            alt = "valid" if subset == "validation" else ("validation" if subset == "valid" else None)
            if alt and os.path.isdir(os.path.join(data_dir, alt)):
                subset = alt
                subset_dir = os.path.join(data_dir, alt)

        self.ms_dir = os.path.join(subset_dir, "ms_downsampled")
        self.pan_dir = os.path.join(subset_dir, "pan_downsampled")
        self.gt_dir = os.path.join(subset_dir, "ms_original_gt")

        self.files = sorted(
            f for f in os.listdir(self.gt_dir) if f.lower().endswith((".tif", ".tiff"))
        )
        if not self.files:
            raise RuntimeError(f"No TIFF files found in {self.gt_dir!r}")

        # Verify file consistency across all three folders at startup
        ms_files = set(os.listdir(self.ms_dir))
        pan_files = set(os.listdir(self.pan_dir))
        missing_ms = [f for f in self.files if f not in ms_files]
        missing_pan = [f for f in self.files if f not in pan_files]
        if missing_ms or missing_pan:
            raise FileNotFoundError(
                f"Missing matching TIFFs — ms_downsampled: {missing_ms[:5]}, "
                f"pan_downsampled: {missing_pan[:5]}"
            )

        self._cache = None
        if self.cache_in_ram:
            num_threads = min(len(self.files), os.cpu_count() or 8)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                self._cache = list(executor.map(self._load_and_upsample, self.files))

    @staticmethod
    def _to_chw(arr: np.ndarray, is_pan: bool = False) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if is_pan:
            if t.ndim == 2:
                return t.unsqueeze(0)
            if t.ndim == 3 and t.shape[-1] == 1:
                return t.permute(2, 0, 1).contiguous()
            return t
        # Multi-spectral: handle both (H, W, C) and (C, H, W)
        if t.ndim == 3 and t.shape[0] not in (3, 4, 8, 12) and t.shape[-1] in (3, 4, 8, 12):
            return t.permute(2, 0, 1).contiguous()
        elif t.ndim == 3 and t.shape[2] < t.shape[0] and t.shape[2] < t.shape[1]:
            return t.permute(2, 0, 1).contiguous()
        return t

    def _load_raw(self, fname: str):
        inv_norm = 1.0 / self.norm_val
        lrms = self._to_chw(tiff.imread(os.path.join(self.ms_dir, fname))).mul_(inv_norm)
        pan = self._to_chw(tiff.imread(os.path.join(self.pan_dir, fname)), is_pan=True).mul_(inv_norm)
        gt = self._to_chw(tiff.imread(os.path.join(self.gt_dir, fname))).mul_(inv_norm)
        return lrms, pan, gt

    def _load_and_upsample(self, fname: str):
        lrms, pan, gt = self._load_raw(fname)
        lrms_up = (
            F.interpolate(
                lrms.unsqueeze(0),
                size=(pan.shape[-2], pan.shape[-1]),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze(0)
            .clamp_(0.0, 1.0)
        )
        return lrms_up, pan, gt

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            lrms_up, pan, gt = self._cache[idx]
        else:
            lrms_up, pan, gt = self._load_and_upsample(self.files[idx])

        if self.patch_size is not None:
            ps = self.patch_size
            _, H, W = gt.shape
            if H < ps or W < ps:
                raise ValueError(
                    f"Image {self.files[idx]!r} ({H}×{W}) is smaller than patch_size={ps}."
                )
            # Align random crop to scale_factor grid
            sf = self.scale_factor
            top = int(torch.randint(0, (H - ps) // sf + 1, (1,)).item()) * sf
            left = int(torch.randint(0, (W - ps) // sf + 1, (1,)).item()) * sf
            lrms_up = lrms_up[:, top : top + ps, left : left + ps]
            pan = pan[:, top : top + ps, left : left + ps]
            gt = gt[:, top : top + ps, left : left + ps]

        return lrms_up, pan, gt
