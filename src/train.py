import argparse
import os

import rootutils
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataloader.div2k import DIV2K
from src.dataloader.div2k_pan import DIV2KPanDataset
from src.dataloader.pansharpen_dataset import PanSharpenDataset
from src.model.pan_net import PanNet, PanNetNonLocal
from src.model.sr_net import SRNet as LocalSRNet
from src.model.sr_net_non_local import SRNet as NonLocalSRNet
from src.utils.logger import Logger

# Registry mapping --model CLI choices to task-specific network architectures
SR_MODELS = {"local": LocalSRNet, "non_local": NonLocalSRNet}
PAN_MODELS = {"local": PanNet, "non_local": PanNetNonLocal}


def _has_tiff_subfolders(dataset_path: str) -> bool:
    """Returns True if dataset_path follows the 3-folder TIFF structure (ms_original_gt)."""
    for sub in ("train", "validation", "valid", "test"):
        if os.path.isdir(os.path.join(dataset_path, sub, "ms_original_gt")):
            return True
    return False


def _fast_collate(batch):
    """Identity collate when Dataset.__getitems__ already returns stacked batch tensors."""
    if isinstance(batch, tuple) and len(batch) in (2, 3) and isinstance(batch[0], torch.Tensor):
        return batch
    return torch.utils.data.default_collate(batch)


def train(args):
    """Runs the training and validation loop for Super-Resolution or Pan-Sharpening.

    Args:
        args: Parsed CLI arguments containing task, model, dataset, and training hyperparameters.
    """
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    is_pan = args.task == "pansharpen"

    # ── 1. Model ───────────────────────────────────────────────────────────
    # Route to PanSharpenDataset (multi-band TIFFs) or DIV2KPanDataset (RGB PNGs)
    use_tiff_pan = is_pan and (
        args.pan_dataset == "tiff"
        or (args.pan_dataset == "auto" and _has_tiff_subfolders(args.dataset_path))
    )
    if is_pan:
        # DIV2K RGB PNGs have 3 bands; default ms_channels to 3 unless using multi-band TIFFs
        ms_channels = args.ms_channels if args.ms_channels is not None else (4 if use_tiff_pan else 3)
        ModelCls = PAN_MODELS[args.model]
        model = ModelCls(
            ms_channels=ms_channels,
            features=args.features,
            blocks=args.blocks,
            kernel_size=args.kernel_size,
        ).to(device)
        dataset_label, model_label = ("PanSharpen", "PanNet") if use_tiff_pan else ("DIV2K_Pan", "PanNet")
    else:
        ModelCls = SR_MODELS[args.model]
        model = ModelCls(
            sampling=args.sampling,
            kernel_size=args.kernel_size,
            features=args.features,
            blocks=args.blocks,
        ).to(device)
        dataset_label, model_label = "DIV2K", "SRNet"

    # Channels-last (NHWC) memory format unlocks maximum Tensor Core throughput on RTX 3060
    if is_cuda:
        model = model.to(memory_format=torch.channels_last)

    model = torch.compile(model)

    # ── 2. DataLoaders ─────────────────────────────────────────────────────
    if is_pan and use_tiff_pan:
        train_dataset = PanSharpenDataset(
            data_dir=args.dataset_path, subset="train", patch_size=args.patch_size
        )
        val_dataset = PanSharpenDataset(
            data_dir=args.dataset_path, subset="validation", patch_size=args.patch_size
        )
        # PanSharpenDataset caches images in RAM and performs random cropping in __getitem__
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=min(4, os.cpu_count() or 2),
            pin_memory=is_cuda,
            persistent_workers=True,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    elif is_pan:
        # Synthesize (lrms_up, pan, gt) directly from the DIV2K PNGs in dataset/
        train_dataset = DIV2KPanDataset(
            path=args.dataset_path,
            subset="train",
            sampling=args.sampling,
            patch_size=args.patch_size,
        )
        val_dataset = DIV2KPanDataset(
            path=args.dataset_path,
            subset="validation",
            sampling=args.sampling,
            patch_size=args.patch_size,
        )
        # DIV2KPanDataset caches pre-extracted 4D patches in RAM like DIV2K;
        # num_workers=0 + _fast_collate avoids multiprocessing IPC overhead
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=is_cuda,
            collate_fn=_fast_collate,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    else:
        train_dataset = DIV2K(
            sampling=args.sampling,
            subset="train",
            path=args.dataset_path,
            patch_size=args.patch_size,
        )
        val_dataset = DIV2K(
            sampling=args.sampling,
            subset="validation",
            path=args.dataset_path,
            patch_size=args.patch_size,
        )
        # DIV2K holds pre-extracted 4D tensors in RAM and implements __getitems__:
        # num_workers=0 with _fast_collate slices batches directly in C++ (~4x faster than IPC workers)
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=is_cuda,
            collate_fn=_fast_collate,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # ── 3. Optimiser, AMP & Scheduler ──────────────────────────────────────
    # L1 loss preserves sharper edges/high-frequency details than MSE in image restoration
    loss_fn = nn.L1Loss()
    # Fused Adam runs all parameter updates in a single multi-tensor CUDA kernel, performance optimization
    optimizer = Adam(model.parameters(), lr=1e-3, fused=is_cuda)
    # Cosine decay from initial lr (1e-3) down to eta_min (1e-6) over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # RTX 3060 (Ampere) natively supports bfloat16 (no underflow/GradScaler overhead) as well as float16
    amp_dtype = torch.bfloat16 if (is_cuda and args.precision == "bf16") else torch.float16
    use_scaler = is_cuda and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=is_cuda)

    # ── 4. Logger & fixed visualisation batch ──────────────────────────────
    logger = Logger(dataset_label, model_label, args.nickname)
    logger.log_params(sum(p.numel() for p in model.parameters()))

    # Fix one validation batch up front so visualisations compare the same images each epoch
    plot_batch = next(iter(val_loader))
    best_val_loss = float("inf")
    mem_fmt = torch.channels_last if is_cuda else torch.contiguous_format

    # Guarantee TensorBoard/logger flush and cleanup even on early interruption
    try:
        for epoch in range(args.epochs):
            # ── Training ───────────────────────────────────────────────────
            model.train()
            train_loss_gpu = torch.zeros(1, device=device)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
            pbar.set_postfix({"val_loss": f"{best_val_loss:.5f}"})

            for batch in pbar:
                # Set .grad = None instead of zero-filling tensors to skip extra memory writes
                optimizer.zero_grad(set_to_none=True)

                if is_pan:
                    lrms_up, pan, gt = batch
                    lrms_up = lrms_up.to(device, non_blocking=True, memory_format=mem_fmt)
                    pan = pan.to(device, non_blocking=True, memory_format=mem_fmt)
                    gt = gt.to(device, non_blocking=True, memory_format=mem_fmt)
                    with autocast_ctx():
                        high = model(lrms_up, pan)
                        loss = loss_fn(high, gt)
                else:
                    gt, low = batch
                    gt = gt.to(device, non_blocking=True, memory_format=mem_fmt)
                    low = low.to(device, non_blocking=True, memory_format=mem_fmt)
                    with autocast_ctx():
                        high = model(low)
                        loss = loss_fn(high, gt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Accumulate on GPU without per-step CPU-GPU synchronisation!
                train_loss_gpu += loss.detach() * gt.size(0)

            total_train_loss = (train_loss_gpu / len(train_loader.dataset)).item()

            # ── Validation ─────────────────────────────────────────────────
            model.eval()
            val_loss_gpu = torch.zeros(1, device=device)
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]  ")
            pbar_val.set_postfix({"val_loss": f"{best_val_loss:.5f}"})

            with torch.inference_mode(), autocast_ctx():
                for batch in pbar_val:
                    if is_pan:
                        lrms_up, pan, gt = batch
                        lrms_up = lrms_up.to(device, non_blocking=True, memory_format=mem_fmt)
                        pan = pan.to(device, non_blocking=True, memory_format=mem_fmt)
                        gt = gt.to(device, non_blocking=True, memory_format=mem_fmt)
                        high = model(lrms_up, pan)
                    else:
                        gt, low = batch
                        gt = gt.to(device, non_blocking=True, memory_format=mem_fmt)
                        low = low.to(device, non_blocking=True, memory_format=mem_fmt)
                        high = model(low)

                    loss = loss_fn(high, gt)
                    val_loss_gpu += loss.detach() * gt.size(0)

            total_val_loss = (val_loss_gpu / len(val_loader.dataset)).item()
            best_val_loss = min(best_val_loss, total_val_loss)

            # ── Logging & Checkpoints ──────────────────────────────────────
            logger.log_loss(epoch, total_train_loss, total_val_loss)
            logger.save_checkpoints(model, epoch, total_val_loss, optimizer, scheduler)
            scheduler.step()

            # Visualise on the fixed validation batch
            with torch.inference_mode(), autocast_ctx():
                if is_pan:
                    lrms_plot, pan_plot, gt_plot = plot_batch
                    high_plot = model(
                        lrms_plot.to(device, non_blocking=True, memory_format=mem_fmt),
                        pan_plot.to(device, non_blocking=True, memory_format=mem_fmt),
                    )
                    logger.plot_results(epoch, gt_plot, lrms_plot, high_plot.float().cpu())
                else:
                    gt_plot, low_plot = plot_batch
                    high_plot = model(
                        low_plot.to(device, non_blocking=True, memory_format=mem_fmt)
                    )
                    logger.plot_results(
                        epoch,
                        gt_plot,
                        low_plot,
                        high_plot.float().cpu(),
                        sampling=args.sampling,
                    )
    finally:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super-Resolution or Pan-Sharpening network")
    parser.add_argument(
        "--task",
        type=str,
        choices=["sr", "pansharpen"],
        default="sr",
        help="Task to train: 'sr' (DIV2K Super-Resolution) or 'pansharpen' (Pan-Sharpening)",
    )
    parser.add_argument("--sampling", type=int, default=4, help="Sampling factor (2, 4 or 8)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument(
        "--model",
        type=str,
        choices=["local", "non_local"],
        required=True,
        help="Which network architecture to train",
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
        help="Number of MS bands (defaults to 3 for DIV2K PNGs or 4 for satellite TIFFs)",
    )
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for 2d convolution")
    parser.add_argument("--features", type=int, default=128, help="Number of features for residual blocks")
    parser.add_argument("--blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch dimension to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train the model")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size to train the model")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="AMP precision (RTX 3060 supports both fp16 and bf16)",
    )
    parser.add_argument("--nickname", type=str, default=None, help="Nickname for save in tensorboard")
    args = parser.parse_args()

    print("Estam amb L1 loss!")
    train(args)
