import argparse
import os
import rootutils
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torcheval.metrics import PeakSignalNoiseRatio

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.sr_net_non_local import SRNet
from src.dataloader.div2k import DIV2K


def predict_tiled(model, low, sampling, tile_size=32, tile_pad=8):
    """
    Performs sliding-window tiled inference on large images to avoid CUDA OOM 
    caused by the quadratic complexity of spatial Non-Local attention.
    Stitches tiles seamlessly by discarding padded borders.
    """
    b, c, h, w = low.shape
    out_h = h * sampling
    out_w = w * sampling
    output = torch.zeros((b, c, out_h, out_w), device=low.device)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Target patch boundaries in low-res
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)

            # Padded boundaries to avoid seam artifacts at patch edges
            y1_pad = max(y1 - tile_pad, 0)
            y2_pad = min(y2 + tile_pad, h)
            x1_pad = max(x1 - tile_pad, 0)
            x2_pad = min(x2 + tile_pad, w)

            # Extract tile and run inference
            tile_in = low[:, :, y1_pad:y2_pad, x1_pad:x2_pad]
            tile_out = model(tile_in)

            # Calculate the corresponding valid high-res slice (crop out pad)
            sy1 = (y1 - y1_pad) * sampling
            sy2 = sy1 + (y2 - y1) * sampling
            sx1 = (x1 - x1_pad) * sampling
            sx2 = sx1 + (x2 - x1) * sampling

            # Place into the output image canvas
            output[:, :, y1*sampling : y2*sampling, x1*sampling : x2*sampling] = tile_out[:, :, sy1:sy2, sx1:sx2]

    return output


def test(args):
    device = args.device
    
    # 1. Instantiate the model with the exact trained architecture
    model = SRNet(
        sampling=args.sampling,
        kernel_size=args.kernel_size,
        features=args.features,
        blocks=args.blocks
    ).to(device)
    
    # 2. Load model weights
    weights = torch.load(args.ckpt_path, map_location=args.device, weights_only=False)
    state_dict = weights['model_state_dict']
    
    # NEW: Strip the '_orig_mod.' prefix added by torch.compile() during training
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            # Remove the first 10 characters ('_orig_mod.')
            clean_key = key[10:] 
            clean_state_dict[clean_key] = value
        else:
            clean_state_dict[key] = value
            
    # Load the cleaned weights into the model
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 3. Load full-size test images (patch_size=None)
    test_dataset = DIV2K(sampling=args.sampling, subset="test", path=args.dataset_path, patch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 4. Metric and output folder setup
    PSNR = PeakSignalNoiseRatio(data_range=1.)
    os.makedirs(args.output_path, exist_ok=True)

    for i, batch in enumerate(tqdm(test_loader)):
        gt, low = batch
        gt = gt.to(device)
        low = low.to(device)

        with torch.no_grad():
            # Use tiled inference to reconstruct the entire full image without OOM
            high = predict_tiled(model, low, sampling=args.sampling, tile_size=32, tile_pad=8)
            bic = nn.functional.interpolate(low, scale_factor=args.sampling, mode='bicubic', align_corners=False)

        # Ensure ground truth matches high dimensions in case of odd image sizes
        gt = gt[:, :, :high.shape[2], :high.shape[3]]

        # Save single, full reconstructed images
        save_image(high, f'{args.output_path}/{i}_out.png')
        save_image(low, f'{args.output_path}/{i}_low.png')
        save_image(bic, f'{args.output_path}/{i}_bic.png')

        PSNR.update(high, gt)

    test_psn_mean = PSNR.compute()
    print(f"\n==========================================")
    print(f"The mean PSNR over the full test set is: {test_psn_mean:.4f} dB")
    print(f"==========================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test script with Tiled Inference")
    parser.add_argument('--sampling', type=int, required=True, help='Sampling factor')
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path of the weights checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path where save the output images")

    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for 2d convolution")
    parser.add_argument("--features", type=int, default=64, help="Number of features for residual blocks")
    parser.add_argument("--blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to test the model")

    args = parser.parse_args()
    test(args)