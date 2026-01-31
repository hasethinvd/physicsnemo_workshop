# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Training script for Transolver on Stokes Flow
# Lab 5: Physics-Attention and Transolver

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW, lr_scheduler

from physicsnemo.models.transolver import Transolver
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger
from physicsnemo.utils.logging.mlflow import initialize_mlflow

from utils import download_stokes_dataset, load_stokes_sample, get_num_samples


class StokesDataset:
    """Simple dataset class for Stokes flow data."""
    
    def __init__(self, num_samples=None):
        download_stokes_dataset()
        total_samples = get_num_samples()
        self.num_samples = num_samples if num_samples else total_samples
        self.num_samples = min(self.num_samples, total_samples)
        
        # Load all samples
        self.data = []
        for i in range(self.num_samples):
            coords, u, v, p = load_stokes_sample(sample_idx=i)
            self.data.append({
                'coords': torch.tensor(coords, dtype=torch.float32),
                'targets': torch.tensor(np.stack([u, v, p], axis=1), dtype=torch.float32)
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        indices = np.random.permutation(len(self.data))
        for idx in indices:
            yield self.data[idx]


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def stokes_trainer(cfg: DictConfig) -> None:
    """Training for the Stokes flow problem using Transolver."""
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Initialize logging
    log = PythonLogger(name="stokes_transolver")
    log.file_logging()
    initialize_mlflow(
        experiment_name="Stokes_Transolver",
        experiment_desc="Training Transolver for Stokes flow around obstacles",
        run_name="Stokes Transolver training",
        run_desc="Training Transolver for Stokes",
        user_name="PhysicsNeMo Workshop",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)
    
    # Define model from config
    model = Transolver(
        functional_dim=cfg.model.functional_dim,
        out_dim=cfg.model.out_dim,
        embedding_dim=cfg.model.embedding_dim,
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
        n_head=cfg.model.n_head,
        act=cfg.model.act,
        mlp_ratio=cfg.model.mlp_ratio,
        slice_num=cfg.model.slice_num,
        unified_pos=cfg.model.unified_pos,
        structured_shape=cfg.model.structured_shape,
        use_te=cfg.model.use_te,
        time_input=cfg.model.time_input,
    ).to(dist.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model created with {n_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.scheduler.initial_lr,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler.decay_epochs,
        gamma=cfg.scheduler.decay_rate
    )
    
    # Load dataset
    dataset = StokesDataset()
    log.info(f"Loaded {len(dataset)} training samples")
    
    # Checkpoint setup
    ckpt_args = {
        "path": "./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist.device, **ckpt_args)
    
    log_args = {
        "name_space": "train",
        "num_mini_batch": len(dataset),
        "epoch_alert_freq": 1,
    }
    
    if loaded_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from epoch {loaded_epoch + 1}.")
    
    # Training loop
    for epoch in range(max(1, loaded_epoch + 1), cfg.training.epochs + 1):
        model.train()
        
        with LaunchLogger(**log_args, epoch=epoch) as logger:
            for batch in dataset:
                coords = batch['coords'].unsqueeze(0).to(dist.device)
                targets = batch['targets'].unsqueeze(0).to(dist.device)
                
                # Forward pass
                optimizer.zero_grad()
                B, N, _ = coords.shape
                fx = torch.zeros(B, N, 0, device=dist.device)
                pred = model(fx, embedding=coords)
                loss = criterion(pred, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()
                
                logger.log_minibatch({"loss": loss.detach()})
            
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)
        
        # Validation
        if epoch % cfg.validation.validation_epochs == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch in dataset:
                    coords = batch['coords'].unsqueeze(0).to(dist.device)
                    targets = batch['targets'].unsqueeze(0).to(dist.device)
                    B, N, _ = coords.shape
                    fx = torch.zeros(B, N, 0, device=dist.device)
                    pred = model(fx, embedding=coords)
                    val_loss += criterion(pred, targets).item()
                val_loss /= len(dataset)
            
            with LaunchLogger("valid", epoch=epoch) as logger:
                logger.log_epoch({"Validation MSE": val_loss})
    
    save_checkpoint(**ckpt_args, epoch=cfg.training.epochs)
    log.success("Training completed!")


if __name__ == "__main__":
    stokes_trainer()
