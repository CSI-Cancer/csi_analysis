import os
from pathlib import Path
import sys

import torch
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math


#Data loader related imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from channel_classifier.config import (
    DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
)
#Model imports
from channel_classifier.modeling.model import ResNet4

#Data config imports

#Model config imports
from channel_classifier.config import (
    MODELS_DIR, 
)

#Utils imports
from channel_classifier.utils import set_random_seeds

#Training related imports
import wandb


class Trainer(object):
    def __init__(self, config=None, aug_params=None):
        self.config = config
        
    def run_training(self):
        with wandb.init(mode='disabled' if self.main_config.get("debug", False) else 'online'):
            if self.config["seed"] is not None:
                set_random_seeds(self.config["seed"])
            
            self.best_pred = np.inf
            self.best_accuracy = 0
            self.best_epoch = 0

            self.build_model()
            self.build_optimizer()
            self.build_scheduler()
            self.train()
    
    def build_model(self):
        self.model = ResNet4(dropout_rate=self.config["dropout_rate"], num_classes=self.config["num_classes"])
        self.model = self.model.to(self.config["device"])