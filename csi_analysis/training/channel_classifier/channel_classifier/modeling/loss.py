import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    INTERIM_DATA_DIR
)

import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
def weighted_loss():
    class_counts = []
    class_dirs = os.listdir(INTERIM_DATA_DIR)
    class_dirs.remove("train.csv")
    class_dirs.remove("val.csv")
    class_dirs.remove("test.csv")
    for class_dir in class_dirs:
        class_path = os.path.join(INTERIM_DATA_DIR, class_dir)
        class_counts.append(len(os.listdir(class_path)))

    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda:0")

    # Define the loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion

def get_class_balanced_loss():
    pass
        

def get_loss_fn(sweep, alpha=None, gamma=None):
    if sweep["loss"] == "weighted":
        return weighted_loss()
    elif sweep["loss"] == "focal":
        return FocalLoss(alpha=sweep["alpha"],
                          gamma=sweep["gamma"],
                          reduction=sweep["reduction"])
