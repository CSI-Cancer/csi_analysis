from torch.utils.data import DataLoader, Dataset
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


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, metadata=None):
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels), "Mismatch between images and labels length"
        self.transform = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.metadata is not None:
            prefix = torch.tensor(self.metadata[idx], dtype=torch.long)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    
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
def get_prefix(files, metadata):
    cohort = []
    try:
        for file in files:
            slide_id = file.split('/')[-1].split('_')[0]
            prefix = metadata['prefix_encoded'][metadata['slide_id'] == slide_id]
            if prefix.empty:
                slide_id = file.split('/')[-1].split('_')[0] + '_' + file.split('/')[-1].split('_')[1]
                prefix = metadata['prefix_encoded'][metadata['slide_id'] == slide_id].values[0]
                cohort.append(prefix)
            else:
                cohort.append(prefix.values[0])
                
    except Exception as e:
        print(e, slide_id)
    return cohort


def get_data_loaders(sweep):
    if sweep.split == 'train/val/test':
        #metadata = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'hitlist_data_metadata.csv'))
        root = INTERIM_DATA_DIR
        classes = os.listdir(root)
        classes.remove('hitlist_data_metadata.csv')
        train_data = []
        train_labels = []
        #train_metadata = []
        val_data = []
        val_labels = []
        #val_metadata = []
        test_data = []
        test_labels = []
        #test_metadata = []
        for i, class_name in enumerate(classes):
            data = glob.glob(os.path.join(root, class_name, '*.png'))
            np.random.shuffle(data)
            #prefix = get_prefix(data, metadata)
            train_data.extend(data[:int(0.7*len(data))])
            train_labels.extend([class_map[class_name]]*len(data[:int(0.7*len(data))]))
            #train_metadata.extend(prefix[:int(0.7*len(data))])
            val_data.extend(data[int(0.7*len(data)):int(0.85*len(data))])
            val_labels.extend([class_map[class_name]]*len(data[int(0.7*len(data)):int(0.85*len(data))]))
            #val_metadata.extend(prefix[int(0.7*len(data)):int(0.85*len(data))])
            test_data.extend(data[int(0.85*len(data)):])
            test_labels.extend([class_map[class_name]]*len(data[int(0.85*len(data)):]))
            #test_metadata.extend(prefix[int(0.85*len(data)):])
        train_image = np.array([np.array(Image.open(data)) for data in train_data])
        val_image = np.array([np.array(Image.open(data)) for data in val_data])
        test_image = np.array([np.array(Image.open(data)) for data in test_data])

        print(f"Number of Training Images : {train_image.shape[0]}")
        print(f"Number of Validation: {val_image.shape[0]}")
        print(f"Number of Test Images: {test_image.shape[0]}")

        # Define the transforms
        train_transform = get_transforms(augment=True)
        val_transform = get_transforms(augment=False)
        test_transform = get_transforms(augment=False)

        train_dataset = CustomDataset(train_image,
                                       train_labels,
                                         transform=train_transform,
                                         metadata=None)#train_metadata)
        val_dataset = CustomDataset(val_image,
                                     val_labels,
                                       transform=val_transform,
                                       metadata=None)#val_metadata)
        test_dataset = CustomDataset(test_image,
                                      test_labels,
                                        transform=test_transform,
                                            metadata=None)#test_metadata)

        train_loader = DataLoader(train_dataset,
                                   batch_size=sweep.batch_size,
                                     shuffle=True,
                                     num_workers=0)
        val_loader = DataLoader(val_dataset,
                                 batch_size=sweep.batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   drop_last=True)
        test_loader = DataLoader(test_dataset,
                                  batch_size=sweep.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    drop_last=True)

        return train_loader, val_loader, test_loader



if __name__ == "__main__":
    get_data_loaders(batch_size=32,
                      augment=True,
                        split='train/val/test')