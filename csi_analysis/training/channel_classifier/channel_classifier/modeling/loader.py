from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from pathlib import Path
import os
import sys
import glob
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    INTERIM_DATA_DIR, class_map
)

from PIL import Image

CLASS_MAP = {
    'D':0,
    'CK':1,
    'CD':2,
    'V':3,
    'CK|CD|V':4,
    'CK|CD':5,
    'D|CK|CD|V':6,
    'CK|V':7,
    'D|CK|CD':8,
    'D|CK|V':9,
    'D|V':10,
    'D|CD|V':11,
    'D|CD':12,
    'D|CK':13,
    'CD|V':14,
}


class CustomDataset(Dataset):
    def __init__(self, df, transform=None, split="train"):
        self.df = df
        self.transform = transform
        self.split = split
        self.labels = df['classification'].map(CLASS_MAP).tolist()
        self.prefix = df['prefix_encoded'].tolist()
        self.images_path = df['path'].tolist()
        assert len(self.images_path) == len(self.labels), "Mismatch between images and labels length"
        

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images_path[idx]))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        prefix = torch.tensor(self.prefix[idx], dtype=torch.float)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label, prefix
    
def get_transforms(augment):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  


def get_data_loaders(sweep):
    if sweep.split == 'train/val/test':
        train_df = pd.read_csv(INTERIM_DATA_DIR / 'train.csv')
        val_df = pd.read_csv(INTERIM_DATA_DIR / 'val.csv')
        test_df = pd.read_csv(INTERIM_DATA_DIR / 'test.csv')

        print(f"Number of Sample size: {train_df.shape[0]}")
        print(f"Number of Validation: {val_df.shape[0]}")
        print(f"Number of Test Images: {test_df.shape[0]}")

        # Define the transforms
        train_transform = get_transforms(augment=True)
        val_transform = get_transforms(augment=False)
        test_transform = get_transforms(augment=False)

        train_dataset = CustomDataset(train_df,
                                         transform=train_transform,
                                         split="train")
        val_dataset = CustomDataset(val_df,
                                       transform=val_transform,
                                       split="val")
        test_dataset = CustomDataset(test_df,
                                        transform=test_transform,
                                        split="test")
        
        # Calculate class weights for stratified sampling
        class_counts = train_df['classification'].value_counts().to_dict()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = train_df['classification'].map(class_weights).values

        # Create a WeightedRandomSampler
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset,
                                   batch_size=sweep.batch_size,
                                     sampler=train_sampler,
                                       num_workers=28)
        val_loader = DataLoader(val_dataset,
                                 batch_size=728,
                                   shuffle=False,
                                     num_workers=28,
                                       drop_last=True)
        test_loader = DataLoader(test_dataset,
                                  batch_size=728,
                                    shuffle=False,
                                      num_workers=28,
                                        drop_last=True)

        return train_loader, val_loader, test_loader
    elif sweep.split == "test":
        test_df = pd.read_csv(INTERIM_DATA_DIR / "test.csv")
        print(f"Number of Test Images: {test_df.shape[0]}")
        test_transform = get_transforms(augment=False)
        test_dataset = CustomDataset(test_df,
                                        transform=test_transform,
                                        split="test")
        test_loader = DataLoader(test_dataset,
                                  batch_size=728,
                                    shuffle=False,
                                      num_workers=28,
                                        drop_last=True)
        return test_loader
    else:
        raise ValueError("Invalid split value. Must be 'train/val/test' or 'test'")



if __name__ == "__main__":
    get_data_loaders(batch_size=32,
                      augment=True,
                        split='train/val/test')