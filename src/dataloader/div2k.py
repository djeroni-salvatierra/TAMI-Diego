import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class DIV2K(Dataset):
    """
    PyTorch Dataset for the DIV2K super-resolution benchmark.

    Extreme hardware optimisations for Intel i7-12700H (20 threads) + RTX 3060:
      1. C++ `torchvision.io.read_image` + `ThreadPoolExecutor` decodes all PNGs
         in parallel across CPU cores (~5x faster than sequential PIL).
      2. Zero-copy `view().permute().contiguous()` extracts non-overlapping square
         patches directly without the huge intermediate allocation of `F.unfold` + `einops`.
      3. Implements `__getitems__` so `DataLoader` slices an entire batch in one C++
         tensor operation instead of looping over individual items in Python.
    """

    def __init__(self, path: str, subset: str, sampling: int, patch_size: int | None = None):
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
            alt = "valid" if subset == "validation" else ("validation" if subset == "valid" else None)
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

        self.gt, self.low = self.generate_data(img_path_list)

    def __len__(self) -> int:
        return len(self.gt)

    def __getitem__(self, index: int):
        return self.gt[index], self.low[index]

    def __getitems__(self, indices: list[int]):
        """
        PyTorch 2.x batch-fetching hook: slices the entire batch at once in C++
        when `self.gt` and `self.low` are stacked 4-D tensors (patch mode).
        """
        if self.patch_size is not None:
            return self.gt[indices], self.low[indices]
        return [(self.gt[i], self.low[i]) for i in indices]

    def _process_single_image(self, img_path: str):
        # Load as float32 [C, H, W] in [0, 1] using fast C++ libpng decoder
        gt = self.read_image(img_path)

        if self.patch_size is not None:
            P = self.patch_size
            C, H, W = gt.shape
            nh, nw = H // P, W // P
            # Bit-for-bit identical to F.unfold + einops rearrange, without intermediate copy
            gt = (
                gt[:, : nh * P, : nw * P]
                .view(C, nh, P, nw, P)
                .permute(1, 3, 0, 2, 4)
                .contiguous()
                .view(nh * nw, C, P, P)
            )
        else:
            gt = gt.unsqueeze(0)

        low = self.degradate(gt)
        return gt, low

    def generate_data(self, img_path_list: list[str]):
        num_threads = min(len(img_path_list), os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self._process_single_image, img_path_list))

        gt_list = [r[0] for r in results]
        low_list = [r[1] for r in results]

        if self.patch_size is None:
            gt_list = [g.squeeze(0) for g in gt_list]
            low_list = [l.squeeze(0) for l in low_list]
            return gt_list, low_list

        return torch.cat(gt_list, dim=0), torch.cat(low_list, dim=0)

    def degradate(self, gt: torch.Tensor) -> torch.Tensor:
        """
        Box-filter downsampling by `self.sampling` via fused `F.avg_pool2d`.
        """
        return F.avg_pool2d(gt, kernel_size=self.sampling, stride=self.sampling)

    @staticmethod
    def read_image(img_path: str) -> torch.Tensor:
        """
        Decodes PNG via C++ libpng directly into a [3, H, W] float32 tensor in [0.0, 1.0].
        """
        return read_image(img_path, mode=ImageReadMode.RGB).to(torch.float32).div_(255.0)
