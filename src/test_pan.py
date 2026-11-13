import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import rootutils
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataloader.div2k_pan import DIV2KPanDataset
from src.dataloader.pansharpen_dataset import PanSharpenDataset
from src.model.pan_net import PanNet, PanNetNonLocal

PAN_MODELS = {"local": PanNet, "non_local": PanNetNonLocal}


def _has_tiff_subfolders(dataset_path: str) -> bool:
    """Returns True if dataset_path follows the 3-folder TIFF structure (ms_original_gt)."""
    for sub in ("test", "train", "validation", "valid"):
        if os.path.isdir(os.path.join(dataset_path, sub, "ms_original_gt")):
            return True
    return False


# ==========================================
# REMOTE SENSING METRICS (SAM & ERGAS)
# ==========================================
def compute_sam(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Spectral Angle Mapper (SAM) in degrees (lower is better).
    Uses `F.cosine_similarity` along the spectral channel dimension (dim=1)
    to fuse dot-product and L2 norms without intermediate tensor allocations.
    """
    cos_sim = F.cosine_similarity(pred, gt, dim=1, eps=1e-8).clamp_(-1.0, 1.0)
    sam_degrees = torch.acos(cos_sim) * (180.0 / torch.pi)
    return sam_degrees.mean()


def compute_ergas(
    pred: torch.Tensor, gt: torch.Tensor, sampling_factor: int
) -> torch.Tensor:
    """
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) (lower is better).
    Note: (RMSE_c / mu_c)^2 == MSE_c / mu_c^2, which avoids an unnecessary per-channel
    sqrt() + square() round-trip.
    """
    resolution_ratio = 1.0 / sampling_factor
    mse_per_channel = torch.mean((pred - gt) ** 2, dim=(2, 3))
    mean_gt_sq = torch.mean(gt, dim=(2, 3)).clamp_(min=1e-8).square_()
    sum_term = torch.mean(mse_per_channel / mean_gt_sq, dim=1)
    ergas = 100.0 * resolution_ratio * torch.sqrt(sum_term)
    return ergas.mean()


# ==========================================


def predict_tiled(
    model,
    lrms_up: torch.Tensor,
    pan: torch.Tensor,
    tile_size: int = 128,
    tile_pad: int = 16,
) -> torch.Tensor:
    """
    Sliding-window tiled inference for Pan-Sharpening.
    Because of Early Fusion, `lrms_up` and `pan` are already at the target high resolution.
    """
    b, c, h, w = lrms_up.shape
    output = torch.empty((b, c, h, w), dtype=lrms_up.dtype, device=lrms_up.device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)

            y1_pad = max(y1 - tile_pad, 0)
            y2_pad = min(y2 + tile_pad, h)
            x1_pad = max(x1 - tile_pad, 0)
            x2_pad = min(x2 + tile_pad, w)

            tile_lrms = lrms_up[:, :, y1_pad:y2_pad, x1_pad:x2_pad]
            tile_pan = pan[:, :, y1_pad:y2_pad, x1_pad:x2_pad]

            tile_out = model(tile_lrms, tile_pan)

            sy1 = y1 - y1_pad
            sy2 = sy1 + (y2 - y1)
            sx1 = x1 - x1_pad
            sx2 = sx1 + (x2 - x1)

            output[:, :, y1:y2, x1:x2] = tile_out[:, :, sy1:sy2, sx1:sx2]

    return output


def test(args):
    """Evaluates a trained PanNet checkpoint on the test set using PSNR, SSIM, SAM, and ERGAS.

    Args:
        args: Parsed CLI arguments specifying model, checkpoint path, dataset, and output directory.
    """
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    mem_fmt = torch.channels_last if is_cuda else torch.contiguous_format

    use_tiff_pan = args.pan_dataset == "tiff" or (
        args.pan_dataset == "auto" and _has_tiff_subfolders(args.dataset_path)
    )
    ms_channels = (
        args.ms_channels if args.ms_channels is not None else (4 if use_tiff_pan else 3)
    )

    # 1. Build the selected PanNet architecture (local or non_local)
    ModelCls = PAN_MODELS[args.model]
    model = ModelCls(
        ms_channels=ms_channels,
        features=args.features,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
    ).to(device, memory_format=mem_fmt)

    # 2. Load weights, stripping the '_orig_mod.' prefix
    weights = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in weights["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Full-size test images (either 16-bit TIFFs or DIV2K PNGs)
    if use_tiff_pan:
        test_dataset = PanSharpenDataset(
            data_dir=args.dataset_path, subset="test", patch_size=None
        )
    else:
        test_dataset = DIV2KPanDataset(
            path=args.dataset_path,
            subset="test",
            sampling=args.sampling,
            patch_size=None,
        )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=is_cuda
    )

    # 4. Initialize Metrics on GPU
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_sam_gpu = torch.zeros(1, device=device)
    total_ergas_gpu = torch.zeros(1, device=device)

    os.makedirs(args.output_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as io_pool:
        futures = []
        with torch.inference_mode():
            for i, (lrms_up, pan, gt) in enumerate(tqdm(test_loader)):
                lrms_up = lrms_up.to(device, non_blocking=True, memory_format=mem_fmt)
                pan = pan.to(device, non_blocking=True, memory_format=mem_fmt)
                gt = gt.to(device, non_blocking=True, memory_format=mem_fmt)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=is_cuda
                ):
                    if args.model == "non_local" or args.tiled:
                        high = predict_tiled(
                            model,
                            lrms_up,
                            pan,
                            tile_size=args.tile_size,
                            tile_pad=args.tile_pad,
                        )
                    else:
                        high = model(lrms_up, pan)

                high = high.float()
                gt = gt[:, :, : high.shape[2], : high.shape[3]]

                # Update Metrics on GPU without per-step CPU sync
                psnr_metric.update(high, gt)
                ssim_metric.update(high, gt)
                total_sam_gpu += compute_sam(high, gt)
                total_ergas_gpu += compute_ergas(high, gt, args.sampling)

                # Offload PNG saving to background threads
                high_rgb = high[:, :3, :, :].cpu()
                lrms_rgb = lrms_up[:, :3, :, :].cpu()
                pan_cpu = pan.cpu()
                futures.append(
                    io_pool.submit(
                        save_image, high_rgb, os.path.join(args.output_path, f"{i}_out.png")
                    )
                )
                futures.append(
                    io_pool.submit(
                        save_image,
                        lrms_rgb,
                        os.path.join(args.output_path, f"{i}_lrms_up.png"),
                    )
                )
                futures.append(
                    io_pool.submit(
                        save_image, pan_cpu, os.path.join(args.output_path, f"{i}_pan.png")
                    )
                )

        for fut in futures:
            fut.result()

    n = len(test_loader)
    test_psnr_mean = psnr_metric.compute().item()
    test_ssim_mean = ssim_metric.compute().item()
    test_sam_mean = (total_sam_gpu / n).item()
    test_ergas_mean = (total_ergas_gpu / n).item()

    print("\n==========================================")
    print(f"Model: {args.model.upper()} PAN-SHARPENING")
    print("------------------------------------------")
    print(f"PSNR  (Higher is better): {test_psnr_mean:.4f} dB")
    print(f"SSIM  (Higher is better): {test_ssim_mean:.4f}")
    print(f"SAM   (Lower is better) : {test_sam_mean:.4f} degrees")
    print(f"ERGAS (Lower is better) : {test_ergas_mean:.4f}")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for Pan-Sharpening")
    parser.add_argument(
        "--model",
        type=str,
        choices=["local", "non_local"],
        required=True,
        help="Which network to test",
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=4,
        help="Sampling factor (usually 4 for remote sensing)",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path of the weights checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path where save the output images"
    )

    parser.add_argument(
        "--pan_dataset",
        type=str,
        choices=["auto", "div2k", "tiff"],
        default="auto",
        help="Pan-Sharpening dataset source: 'div2k' (PNGs), 'tiff' (PanSharpenDataset), or 'auto'",
    )
    parser.add_argument(
        "--ms_channels",
        type=int,
        default=None,
        help="Number of Multi-Spectral bands (defaults to 3 for DIV2K PNGs or 4 for TIFFs)",
    )
    parser.add_argument(
        "--kernel_size", type=int, default=3, help="Kernel size for 2d convolution"
    )
    parser.add_argument(
        "--features", type=int, default=128, help="Number of features for residual blocks"
    )
    parser.add_argument(
        "--blocks", type=int, default=3, help="Number of residual blocks"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to test the model"
    )

    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Force tiled inference even for local model on huge scenes",
    )
    parser.add_argument(
        "--tile_size", type=int, default=128, help="Tile size for tiled inference"
    )
    parser.add_argument(
        "--tile_pad", type=int, default=16, help="Tile padding for tiled inference"
    )

    args = parser.parse_args()
    test(args)
