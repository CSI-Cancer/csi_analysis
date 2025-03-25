from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random

from pathlib import Path
import os
import sys
import glob
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    INTERIM_DATA_DIR, class_map, negative_class_mapping
)

from PIL import Image

CLASS_MAP = {
    'D':0,
    'CK':8,
    'CD':8,
    'V':8,
    'CK|CD|V':8,
    'CK|CD':8,
    'D|CK|CD|V':1,
    'CK|V':8,
    'D|CK|CD':2,
    'D|CK|V':3,
    'D|V':4,
    'D|CD|V':5,
    'D|CD':6,
    'D|CK':7,
    'CD|V':8,
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
        image = np.load(self.images_path[idx][0:-3]+"npy")
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
            transforms.Normalize(mean=[0.9019, 1.4593, 1.3794, 0.8784],
                                  std=[2.331, 3.0221, 2.9218, 2.1913])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.9019, 1.4593, 1.3794, 0.8784],
                                 std=[2.331, 3.0221, 2.9218, 2.1913])
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
    

class TripletDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing class folders with .npy files
        """
        self.data_dir = data_dir
        self.negative_class_mapping = negative_class_mapping

        self.classes = list(CLASS_MAP.keys())

        # Load all data into memory
        self.class_to_samples = {}

        first_file = None

        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                samples = []
                for file in os.listdir(class_path):
                    if file.endswith(".npy"):
                        data = np.load(os.path.join(class_path, file))
                        # Permute NHWC to NCHW here during loading
                        data = np.transpose(data, (2, 0, 1))  # [H, W, C] -> [C, H, W]
                        if first_file is None:
                            first_file = data
                        samples.append(data)
                if len(samples) > 1:
                    self.class_to_samples[class_name] = samples

        # Create blank negative samples (already in NCHW format)
        self.blank_negative = np.zeros_like(first_file)

        # Create list of anchor samples
        self.samples = []
        for class_name, samples in self.class_to_samples.items():
            self.samples.extend([(sample, class_name) for sample in samples])

    def __getitem__(self, idx):
        anchor_sample, anchor_class = self.samples[idx]
        
        # Get positive sample (same class, different sample)
        positive_samples = self.class_to_samples[anchor_class]
        positive_sample = random.choice([s for s in positive_samples if not np.array_equal(s, anchor_sample)])
        
        # Get negative sample (different class)
        negative_classes = self.negative_class_mapping.get(anchor_class, [c for c in self.classes if c != anchor_class])
        if not negative_classes:
            negative_sample = self.blank_negative
        else:
            negative_class = random.choice(negative_classes)
            negative_sample = random.choice(self.class_to_samples[negative_class])

        # Convert to torch tensors (already in NCHW format)
        anchor = torch.from_numpy(anchor_sample).float()
        positive = torch.from_numpy(positive_sample).float()
        negative = torch.from_numpy(negative_sample).float()
        
        # Get numerical class label from CLASS_MAP
        class_label = CLASS_MAP[anchor_class]
        
        return anchor, positive, negative, class_label

    def __len__(self):
        return len(self.samples)
        

if __name__ == "__main__":
    get_data_loaders(batch_size=32,
                      augment=True,
                        split='train/val/test')