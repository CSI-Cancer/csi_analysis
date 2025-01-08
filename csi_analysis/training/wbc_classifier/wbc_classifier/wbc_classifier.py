import torch
import torch.nn as nn
import torch.optim as optim

# For image reading and processing
import numpy as np
import os
from wbc_dataloader import CustomImageDataset

# Define a complex CNN architecture for binary classification
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional block
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Second convolutional block
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Third convolutional block
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Fourth convolutional block
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Adjust input features based on image size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2)  # Output layer for binary classification

    def forward(self, x):
        out = self.layer1(x)  # Shape changes according to pooling
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dropout(nn.functional.relu(self.fc1(out)))
        out = self.fc2(out)
        return out
