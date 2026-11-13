import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import rootutils
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.utils import save_image
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataloader.div2k import DIV2K
from src.model.sr_net import SRNet as LocalSRNet
from src.model.sr_net_non_local import SRNet as NonLocalSRNet

MODELS = {"local": LocalSRNet, "non_local": NonLocalSRNet}


def predict_tiled(
    model, low: torch.Tensor, sampling: int, tile_size: int = 32, tile_pad: int = 8
) -> torch.Tensor:
    """
    Sliding-window tiled inference to avoid CUDA OOM caused by spatial Non-Local
    attention on full-resolution 2K images. Tiles overlap by `tile_pad` pixels on
    each side and the padded borders are discarded when stitching.
    """
    b, c, h, w = low.shape
    out_h = h * sampling
    out_w = w * sampling
    output = torch.empty((b, c, out_h, out_w), dtype=low.dtype, device=low.device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)

            y1_pad = max(y1 - tile_pad, 0)
            y2_pad = min(y2 + tile_pad, h)
            x1_pad = max(x1 - tile_pad, 0)
            x2_pad = min(x2 + tile_pad, w)

            tile_in = low[:, :, y1_pad:y2_pad, x1_pad:x2_pad]
            tile_out = model(tile_in)

            sy1 = (y1 - y1_pad) * sampling
            sy2 = sy1 + (y2 - y1) * sampling
            sx1 = (x1 - x1_pad) * sampling
            sx2 = sx1 + (x2 - x1) * sampling

            output[
                :, :, y1 * sampling : y2 * sampling, x1 * sampling : x2 * sampling
            ] = tile_out[:, :, sy1:sy2, sx1:sx2]

    return output


def test(args):
    """Evaluates a trained SRNet checkpoint on the DIV2K test set and saves output images.

    Args:
        args: Parsed CLI arguments specifying model, checkpoint path, dataset, and output directory.
    """
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    mem_fmt = torch.channels_last if is_cuda else torch.contiguous_format

    # 1. Build the selected architecture (matches trained checkpoints)
    SRNet = MODELS[args.model]
    model = SRNet(
        sampling=args.sampling,
        kernel_size=args.kernel_size,
        features=args.features,
        blocks=args.blocks,
    ).to(device, memory_format=mem_fmt)

    # 2. Load weights, stripping '_orig_mod.' if saved from a compiled model
    weights = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in weights["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Full-size test images (already preloaded in RAM by DIV2K)
    test_dataset = DIV2K(
        sampling=args.sampling, subset="test", path=args.dataset_path, patch_size=None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=is_cuda
    )

    # 4. Metric on GPU and background PNG writer pool
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    os.makedirs(args.output_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as io_pool:
        futures = []
        with torch.inference_mode():
            for i, (gt, low) in enumerate(tqdm(test_loader)):
                gt = gt.to(device, non_blocking=True, memory_format=mem_fmt)
                low = low.to(device, non_blocking=True, memory_format=mem_fmt)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=is_cuda
                ):
                    if args.model == "non_local":
                        high = predict_tiled(
                            model,
                            low,
                            sampling=args.sampling,
                            tile_size=args.tile_size,
                            tile_pad=args.tile_pad,
                        )
                    else:
                        high = model(low)

                high = high.float()
                bic = F.interpolate(
                    low, scale_factor=args.sampling, mode="bicubic", align_corners=False
                )

                # Match ground truth to output size in case of odd image sizes
                gt = gt[:, :, : high.shape[2], : high.shape[3]]
                psnr.update(high, gt)

                # Offload PNG encoding to background CPU threads so the GPU never waits
                high_cpu = high.cpu()
                low_cpu = low.cpu()
                bic_cpu = bic.cpu()
                futures.append(
                    io_pool.submit(
                        save_image, high_cpu, os.path.join(args.output_path, f"{i}_out.png")
                    )
                )
                futures.append(
                    io_pool.submit(
                        save_image, low_cpu, os.path.join(args.output_path, f"{i}_low.png")
                    )
                )
                futures.append(
                    io_pool.submit(
                        save_image, bic_cpu, os.path.join(args.output_path, f"{i}_bic.png")
                    )
                )

        for fut in futures:
            fut.result()

    test_psnr_mean = psnr.compute().item()
    print("\n==========================================")
    print(f"Model: {args.model}")
    print(f"The mean PSNR over the full test set is: {test_psnr_mean:.4f} dB")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for Local and Non-Local SRNet")
    parser.add_argument(
        "--model",
        type=str,
        choices=["local", "non_local"],
        required=True,
        help="Which network to test",
    )
    parser.add_argument("--sampling", type=int, required=True, help="Sampling factor")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path of the weights checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path where save the output images"
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
        "--tile_size", type=int, default=32, help="Tile size (low-res) for tiled inference"
    )
    parser.add_argument(
        "--tile_pad", type=int, default=8, help="Tile padding (low-res) for tiled inference"
    )

    args = parser.parse_args()
    test(args)
