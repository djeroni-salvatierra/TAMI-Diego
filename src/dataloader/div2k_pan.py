import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class DIV2KPanDataset(Dataset):
    """Synthesises Pan-Sharpening pairs (lrms_up, pan, gt) on the fly from DIV2K RGB PNGs.

    Given a high-resolution RGB image `gt` of shape [3, H, W]:
      1. `pan` ([1, H, W]) is synthesized at full spatial resolution via ITU-R BT.601
         luminance projection (0.299 R + 0.587 G + 0.114 B) or channel averaging.
      2. `lrms` ([3, H//s, W//s]) is synthesized by box-filter downsampling (`F.avg_pool2d`)
         and upsampled back to `[3, H, W]` (`lrms_up`) via bicubic interpolation.
    """

    # ITU-R BT.601 luminance weights for RGB -> 1-channel Panchromatic projection
    # Best averaging for human-centric display
    _LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)

    def __init__(
        self,
        path: str,
        subset: str = "train",
        sampling: int = 4,
        patch_size: int | None = 64,
        pan_mode: str = "luma",
    ):
        """Initializes DIV2KPanDataset and pre-computes (lrms_up, pan, gt) tensors in RAM.

        Args:
            path: Root directory of the DIV2K dataset (containing train/validation/test).
            subset: Sub-folder name ("train", "validation"/"valid", or "test").
            sampling: Spatial downscaling factor for the multi-spectral image (2, 4, or 8).
            patch_size: High-resolution square patch size (None for full-image evaluation).
            pan_mode: How to synthesize the PAN band from RGB ("luma" (non-average weights) or "mean").
        """
        super().__init__()

        if patch_size is None and subset != "test":
            raise ValueError(
                'Dataset images have different sizes, so a patch_size is required '
                'to crop them into uniform squares (except for the "test" subset).'
            )

        if sampling not in (2, 4, 8):
            raise ValueError("Sampling factor should be 2, 4 or 8")

        # Automatically resolve "validation" <-> "valid" folder naming
        subset_dir = os.path.join(path, subset)
        if not os.path.isdir(subset_dir):
            alt = (
                "valid"
                if subset == "validation"
                else ("validation" if subset == "valid" else None)
            )
            if alt and os.path.isdir(os.path.join(path, alt)):
                subset = alt
                subset_dir = os.path.join(path, alt)

        img_path_list = sorted(
            os.path.join(subset_dir, img)
            for img in os.listdir(subset_dir)
            if img.endswith(".png")
        )
        if not img_path_list:
            raise RuntimeError(f"No PNG files found in {subset_dir!r}")

        self.sampling = sampling
        self.patch_size = patch_size
        self.subset = subset
        self.pan_mode = pan_mode

        self.lrms_up, self.pan, self.gt = self.generate_data(img_path_list)

    def __len__(self) -> int:
        return len(self.gt)

    def __getitem__(self, index: int):
        return self.lrms_up[index], self.pan[index], self.gt[index]

    def __getitems__(self, indices: list[int]):
        """Vectorized C++ batch slicing when patches are stacked as 4-D tensors."""
        if self.patch_size is not None:
            return self.lrms_up[indices], self.pan[indices], self.gt[indices]
        return [(self.lrms_up[i], self.pan[i], self.gt[i]) for i in indices]

    def _make_pan(self, gt: torch.Tensor) -> torch.Tensor:
        # [N, 3, H, W] -> [N, 1, H, W]
        if self.pan_mode == "luma" and gt.size(1) == 3:
            return (gt * self._LUMA_WEIGHTS).sum(dim=1, keepdim=True)
        return gt.mean(dim=1, keepdim=True)

    def _process_single_image(self, img_path: str):
        # Decode PNG to float32 [3, H, W] in [0, 1]
        gt = read_image(img_path, mode=ImageReadMode.RGB).to(torch.float32).div_(255.0)

        if self.patch_size is not None:
            P = self.patch_size
            C, H, W = gt.shape
            nh, nw = H // P, W // P
            # [3, H, W] -> [nh * nw, 3, P, P]
            gt = (
                gt[:, : nh * P, : nw * P]
                .view(C, nh, P, nw, P)
                .permute(1, 3, 0, 2, 4)
                .contiguous()
                .view(nh * nw, C, P, P)
            )
        else:
            # Ensure full-image H and W are multiples of `sampling` for exact upsampling
            C, H, W = gt.shape
            s = self.sampling
            gt = gt[:, : (H // s) * s, : (W // s) * s].unsqueeze(0)  # [1, 3, H', W']

        # [N, 3, H, W] -> [N, 1, H, W] high-resolution Panchromatic band
        pan = self._make_pan(gt)

        # [N, 3, H, W] -> [N, 3, H//s, W//s] -> [N, 3, H, W] upsampled low-res MS
        lrms = F.avg_pool2d(gt, kernel_size=self.sampling, stride=self.sampling)
        lrms_up = F.interpolate(
            lrms,
            size=(gt.shape[-2], gt.shape[-1]),
            mode="bicubic",
            align_corners=False,
        ).clamp_(0.0, 1.0)

        return lrms_up, pan, gt

    def generate_data(self, img_path_list: list[str]):
        num_threads = min(len(img_path_list), os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self._process_single_image, img_path_list))

        lrms_list = [r[0] for r in results]
        pan_list = [r[1] for r in results]
        gt_list = [r[2] for r in results]

        if self.patch_size is None:
            return (
                [x.squeeze(0) for x in lrms_list],
                [x.squeeze(0) for x in pan_list],
                [x.squeeze(0) for x in gt_list],
            )

        return (
            torch.cat(lrms_list, dim=0),
            torch.cat(pan_list, dim=0),
            torch.cat(gt_list, dim=0),
        )
