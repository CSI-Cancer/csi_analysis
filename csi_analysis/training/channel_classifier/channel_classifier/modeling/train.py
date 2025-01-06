import os
from pathlib import Path
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math


#Data loader related imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, MODELS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
)
#Model imports
from modeling.model import ResNet4

#Data config imports
from config import (
    tune_config, sweep_config
)

from modeling.loader import get_data_loaders


#Utils imports
from utils import set_random_seeds

#Training related imports
import wandb


class Trainer(object):
    def __init__(self, config=None):
        self.config = config
        
    def run_training(self):
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
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(sweep)
    
    def build_model(self, sweep):
        self.model = ResNet4(dropout_rate=sweep["dropout_rate"], num_classes=sweep["num_classes"])
        self.model = self.model.to(sweep["device"])

    def build_optimizer(self, sweep):
        if sweep.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=sweep.lr,
                weight_decay=sweep.weight_decay,
                momentum=sweep.momentum
            )

        elif sweep.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=sweep.lr,
                weight_decay=sweep.weight_decay
            )

    def build_scheduler(self, sweep):
                # Create a combined scheduler with linear warmup and exponential decay
        
        if sweep.scheduler == 'LambdaLR':
            # Calculate the multiplier to scale from base_lr to max_lr
            lr_multiplier = sweep.l_e / sweep.l_b

            self.scheduler = LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: ((lr_multiplier - 1) * epoch / sweep.l_e + 1)
                if epoch < sweep.l_e else
                lr_multiplier * (sweep.l_b ** (epoch - sweep.l_e))
            )

    def build_loss(self, sweep):
        # Calculate class weights
        class_counts = []
        class_dirs = os.listdir(INTERIM_DATA_DIR)
        for class_dir in class_dirs:
            class_path = os.path.join(INTERIM_DATA_DIR, class_dir)
            class_counts.append(len(os.listdir(class_path)))

        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.config["device"])

        # Define the loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train(self, sweep):
        best_val_loss = np.inf
        best_epoch = 0

        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_loader = self.train_loader
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}"):
                inputs, targets = inputs.to(self.config["device"]), targets.to(self.config["device"])

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            train_loss /= len(train_loader)
            train_accuracy = 100. * train_correct / train_total

            # Validate the model
            val_loss, val_accuracy = self.validate(sweep)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), MODELS_DIR / f"best_model.pth")

            print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    

    def validate(self, sweep):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_loader = self.val_loader
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.config["device"]), targets.to(self.config["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        return val_loss, val_accuracy
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, val_loss, path):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }
        torch.save(state, path)
        print(f"Checkpoint saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    trainer = Trainer(tune_config)
    if tune_config["tune"]:
        wandb.login(key=tune_config["wandb_key"])
        sweep_id = wandb.sweep(sweep_config, project="channel_classifier")
        wandb.agent(sweep_id, trainer.run_training, count=tune_config['count'])
    else:
        trainer.run_training()