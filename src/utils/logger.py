import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    def __init__(self, dataset: str, model_name: str, nickname: str | None):
        day = datetime.now().strftime("%Y-%m-%d")
        self.model_name = model_name
        self.nickname = nickname

        project_root = os.environ.get("PROJECT_ROOT", ".")
        parts = [project_root, "logs", dataset, model_name, day]
        if nickname is not None:
            parts.append(nickname)
        self.dir_path = os.path.join(*parts)

        self.writer = SummaryWriter(self.dir_path)
        os.makedirs(os.path.join(self.dir_path, "checkpoints"), exist_ok=True)
        self.best_loss = float("inf")

    def log_loss(self, epoch: int, train_loss: float, validation_loss: float) -> None:
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/validation", validation_loss, epoch)
        self.writer.add_scalars(
            "Loss/comparison",
            {"train": train_loss, "validation": validation_loss},
            epoch,
        )

    def log_params(self, num_params: int) -> None:
        self.writer.add_text("Parameters", f"{self.model_name} = {num_params:,}")

    def save_checkpoints(
        self,
        model: torch.nn.Module,
        epoch: int,
        validation_loss: float,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
    ) -> None:
        """
        Saves training checkpoints (`last.ckpt` and `best.ckpt`, plus legacy `.cpkt`
        symlinks/copies so existing scripts or habits continue to work seamlessly).
        """
        # Save clean unwrapped state_dict if model was wrapped by torch.compile
        raw_model = getattr(model, "_orig_mod", model)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "best_loss": min(self.best_loss, validation_loss),
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

        ckpt_dir = os.path.join(self.dir_path, "checkpoints")
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        torch.save(ckpt, last_ckpt)

        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
            torch.save(ckpt, best_ckpt)

    def plot_results(
        self,
        epoch: int,
        gt: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        sampling: int | None = None,
    ) -> None:
        """
        Writes a side-by-side grid [GT | Low/Bicubic | Prediction] to TensorBoard.
        Works for both 3-channel RGB (DIV2K) and multi-band MS (Pan-Sharpening, slicing first 3 bands).
        """
        gt_vis = gt[:, :3, :, :].float()
        low_vis = low[:, :3, :, :].float()
        high_vis = high[:, :3, :, :].float()

        if low_vis.shape[-2:] != high_vis.shape[-2:]:
            if sampling is None:
                sampling = high_vis.size(2) // low_vis.size(2)
            inter = F.interpolate(
                low_vis, scale_factor=sampling, mode="bicubic", align_corners=False
            )
        else:
            # For Pan-Sharpening, `low` (lrms_up) is already upsampled to target resolution
            inter = low_vis

        images = torch.cat([gt_vis, inter, high_vis], dim=3).clamp_(0.0, 1.0)
        grid = make_grid(images, nrow=1)
        self.writer.add_image("Image comparison: Ref ~ Bic ~ Pred", grid, epoch)

    def close(self) -> None:
        self.writer.close()
