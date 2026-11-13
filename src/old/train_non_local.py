import argparse
import os
import rootutils
import torch

torch.set_float32_matmul_precision('high')

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logger import Logger
from src.model.sr_net_non_local import SRNet
from src.dataloader.div2k import DIV2K

torch.backends.cudnn.benchmark = True


def train(args):
    device = args.device
    # Definim el model
    model = SRNet(sampling=args.sampling,kernel_size=args.kernel_size, features=args.features, blocks=args.blocks).to(device)
    
    # NEW: Compile the model for faster execution
    model = torch.compile(model)
    
    # Carragam els dataloader
    train_dataset = DIV2K(sampling=args.sampling,subset="train",path=args.dataset_path, patch_size=args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    validation_dataset = DIV2K(sampling=args.sampling, subset="validation",path=args.dataset_path, patch_size=args.patch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    # Definim la funció de perdua i l'optimitzador (Mean Squared Error)
    loss_function = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # NEW: Initialize the AMP Scaler
    scaler = torch.amp.GradScaler('cuda')

    # NEW: Initialize the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Instanciam la classe que emplearem per monitoritzar els resultas i guardar els pesos
    logger = Logger("DIV2K", "SRNet", args.nickname)

    # Començam l'entrenament
    max_epochs = args.epochs
    total_validataion_loss = float("inf")
    num_param = sum(p.numel() for p in model.parameters())
    logger.log_params(num_param)
    for epoch in range(max_epochs):
        # Fem una passada en tot el conjunt de train dividint les imatges en batches per a que hi capiga en memòria:
        model.train()
        total_train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Training]")
        train_loader_tqdm.set_postfix({"Validation loss": total_validataion_loss})
        for idx, batch in enumerate(train_loader_tqdm):
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            gt, low = batch
            gt = gt.to(device)
            low = low.to(device)
            batch_len = gt.size(0)

            # NEW: Cast forward pass to FP16
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                high = model(low)
                loss = loss_function(high, gt)
            
            # NEW: Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += batch_len*loss.item() / len(train_loader.dataset)
        # Fem una passada en el conjunt de validació per veure com es comporta la loss en aquest conjunt
        model.eval()
        validation_loader_tqdm = tqdm(validation_loader,desc=f"Epoch {epoch+1}/{max_epochs} [Validation]")
        validation_loader_tqdm.set_postfix({"Validation loss": total_validataion_loss})
        total_validataion_loss = 0
        for idx, batch in enumerate(validation_loader_tqdm):
            gt, low = batch
            gt = gt.to(device)
            low = low.to(device)
            batch_len = gt.size(0)
            # Utilitzam torch.no_grad() per no utilitzar tanta memoria i poder fer els calculs més ràpid.
            with torch.no_grad():
                high = model(low)
            # Calculam la loss
            loss = loss_function(high,gt)
            total_validataion_loss += batch_len * loss.item() / len(validation_loader.dataset)

        # Al finalitzar cada epoch monitoritzam la loss i les imatges i també guardam els pesos
        logger.log_loss(epoch, total_train_loss, total_validataion_loss)
        logger.save_checkpoints(model, epoch, total_validataion_loss)

        # NEW: Step the scheduler to lower the learning rate for the next epoch
        scheduler.step()

        batch_plot = next(iter(validation_loader))
        gt, low = batch_plot
        gt = gt.to(device)
        low = low.to(device)
        with torch.no_grad():
            high = model(low)
        logger.plot_results(epoch, gt.cpu(), low.cpu(), high.cpu())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument('--sampling', type=int, required=True, help='Sampling factor')
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")

    parser.add_argument("--kernel_size", type=int,default=3,help="Kernel size for 2d convoltion")
    parser.add_argument("--features", type=int, default=128, help="Number of features for residual blocks")
    parser.add_argument("--blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch dimension to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train the model")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size to train the model")
    parser.add_argument("--nickname", type=str, default=None, help="Nickname for save in tensorboard")
    args = parser.parse_args()

    train(args)



