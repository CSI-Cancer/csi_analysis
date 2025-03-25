import os
from pathlib import Path
import sys
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast

# Data loader related imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
)

# Model imports
from modeling.model import get_pretrain_model
from modeling.loss import TripletLoss, InfoNCELoss
from modeling.loader import TripletDataset

# Utils imports
from utils import set_random_seeds

# Training related imports
import wandb

from config import (
    tune_config, sweep_config
)


class Pretrainer(object):
    def __init__(self, config=None):
        self.config = config
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.early_stopping_patience = 25
        self.epochs_since_improvement = 0
        self.scaler = GradScaler()  # For mixed precision training
        
    def run_pretraining(self):  # Add debug parameter
        with wandb.init(mode='disabled' if self.config.get("debug", False) else 'online'):
            if self.config["seed"] is not None:
                set_random_seeds(self.config["seed"])
            
            self.best_pred = np.inf
            self.best_accuracy = 0
            self.best_epoch = 0

            self.build_dataset(sweep=wandb.config)
            self.build_model(sweep=wandb.config)
            self.build_optimizer(sweep=wandb.config)
            self.build_scheduler(sweep=wandb.config)
            self.build_loss(sweep=wandb.config)
            self.train(sweep=wandb.config)
    
    def build_dataset(self, sweep):
        dataset = TripletDataset(INTERIM_DATA_DIR)
        self.train_loader = DataLoader(
            dataset,
            batch_size=sweep.batch_size,
            shuffle=True,
            num_workers=8,  # Increased for faster data loading
            pin_memory=True,
            drop_last=True,
            persistent_workers=True  # Keep workers alive between epochs
        )
    
    def build_model(self, sweep):
        self.model = get_pretrain_model(
            dropout=sweep.dropout,
            embedding_dim=sweep.embedding_dim,
            model_name=sweep.model
        )
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def build_optimizer(self, sweep):
        if sweep.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=sweep.learning_rate,
                weight_decay=sweep.weight_decay
            )
        elif sweep.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=sweep.learning_rate,
                weight_decay=sweep.weight_decay
            )

    def build_scheduler(self, sweep):
        if sweep.scheduler == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                patience=sweep.scheduler_params['patience'],
                factor=sweep.scheduler_params['factor'],
                min_lr=sweep.scheduler_params['min_lr']
            )
        elif sweep.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=sweep.epochs,
                eta_min=sweep.scheduler_params['min_lr']
            )

    def build_loss(self, sweep):
        self.triplet_loss = TripletLoss(margin=sweep.margin).to(sweep.device)
        self.infonce_loss = InfoNCELoss(temperature=sweep.get('temperature', 0.07)).to(sweep.device)

    def train(self, sweep, debug=True):
        torch.backends.cudnn.benchmark = True
        
        pretrain_dir = MODELS_DIR / "pretrained"
        pretrain_dir.mkdir(exist_ok=True, parents=True)
        
        for epoch in range(sweep.epochs):
            self.model.train()
            total_loss = 0
            total_triplet_loss = 0
            total_infonce_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{sweep.epochs}")
            
            for batch_idx, (anchor, positive, negative, classes) in enumerate(progress_bar):
                # Move to device
                anchor = anchor.to(self.device, non_blocking=True)
                positive = positive.to(self.device, non_blocking=True)
                negative = negative.to(self.device, non_blocking=True)
                classes = classes.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Use mixed precision training
                with autocast(device_type='cuda'):
                    # Get embeddings
                    anchor_out = self.model(anchor, return_embeddings=True)
                    positive_out = self.model(positive, return_embeddings=True)
                    negative_out = self.model(negative, return_embeddings=True)
                    
                    # Calculate accuracy using anchor outputs
                    pred = torch.argmax(anchor_out, dim=1)
                    correct += (pred == classes).sum().item()
                    total += classes.size(0)
                    
                    # Calculate losses
                    triplet_loss = self.triplet_loss(anchor_out, positive_out, negative_out)
                    infonce_loss = self.infonce_loss(anchor_out, positive_out)
                    
                    # Combine losses
                    combined_loss = sweep.lambda_triplet * triplet_loss + sweep.lambda_infonce * infonce_loss
                
                # Use gradient scaling
                self.scaler.scale(combined_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                total_loss += combined_loss.item()
                total_triplet_loss += triplet_loss.item()
                total_infonce_loss += infonce_loss.item()
                
                # Calculate current accuracy
                current_accuracy = 100. * correct / total
                
                # Update progress bar
                if batch_idx % 5 == 0:
                    progress_bar.set_postfix({
                        'loss': f'{combined_loss.item():.3f}',
                        'triplet': f'{triplet_loss.item():.3f}',
                        'infonce': f'{infonce_loss.item():.3f}',
                        'acc': f'{current_accuracy:.1f}%'
                    })

    def save_checkpoint(self, epoch, path, sweep):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_loss,
            'config': sweep,
        }
        torch.save(state, path)
        print(f"Checkpoint saved at epoch {epoch+1} with loss: {self.best_loss:.4f}")

def run_sweep(debug=False):
    # Get the directory containing the current script
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "pretraining_config.yml"
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)
    
    pretrainer = Pretrainer(config=tune_config)
    # Run normal sweep
    wandb.login(key=tune_config["wandb_key"])
    sweep_id = wandb.sweep(sweep_config, project="channel-classifier-pretraining")

    if tune_config["tune"]:
        wandb.login(key=tune_config["wandb_key"])
        sweep_id = wandb.sweep(sweep_config, project="channel_classifier")
        wandb.agent(sweep_id, pretrainer.run_pretraining, count=tune_config['count'])
    else:
        pretrainer.run_pretraining()
    


if __name__ == "__main__":
    # Set debug=True when debugging
    run_sweep(debug=True)