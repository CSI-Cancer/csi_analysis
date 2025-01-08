#data_loader

from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
import glob

import torch
import random
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import numbers
import torch.nn as nn
import pandas as pd
    
class CustomImageDataset(Dataset):
    def __init__(self, images, masks, labels, tran=False):
        """
        Custom dataset for loading 4-channel, 75x75, 16-bit TIFF images.
        :param images: Numpy array of images.
        :param masks: Numpy array of binary masks. 
        :param labels: Numpy array of labels.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.tran=tran

        self.t = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Extract a single image and its label
        #image = np.log1p(self.images[idx].astype(np.float32)) / np.log(65535.0)
        image = self.images[idx].astype(np.float32) / 65535.0
        label = self.labels[idx]
        mask = self.masks[idx].astype(np.int16)
        
        image = self.t(image)
        
        mask = self.t(mask)
        hard_masked_image = image * mask
        hard_masked_image = torch.cat((hard_masked_image, mask), dim=0)

        return hard_masked_image, torch.tensor(label, dtype=torch.long)



def get_data_loaders(data_path, batch_size=64, target_size_per_class=2000):
    types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    train_images_list, val_images_list = [], []
    train_masks_list, val_masks_list = [], []
    train_labels_list, val_labels_list = [], []
    all_slides = []
    print(types)
    for label, t in enumerate(types):
        print(t)
        if(t=="WBCs"):
            label=1
        elif(t=="Small_DAPI_Only"):
            label=1
        elif(t=="Large_DAPI_Only"):
            label=1
        else:
            label=0
        current_type_path = os.path.join(data_path, t)
        current_type_files = glob.glob(os.path.join(current_type_path, "*.hdf5"))
        class_images = []
        class_masks = []


        for file_path in current_type_files:
            with h5py.File(file_path, 'r') as f:
                imgs = np.array(f['images'][:], dtype=np.float32)  # Ensure dtype matches the image data type
                msks = np.array(f['masks'][:])

                class_images.append(imgs)
                class_masks.append(msks)
            class_features = pd.read_hdf(file_path, key='features')
            #if slide_id is a column, then print all unique slide_ids
            if 'slide_id' in class_features.columns:
                #add all unique slide_ids to all_slides
                all_slides.extend(class_features['slide_id'].unique())


                
        class_images = np.concatenate(class_images, axis=0)
        class_masks = np.concatenate(class_masks,axis=0)
        
        # Downsample if necessary
        if len(class_images) > target_size_per_class/0.8:
            indices = np.random.choice(range(len(class_images)), int(target_size_per_class/0.8), replace=False)
            class_images = class_images[indices]
            class_masks = class_masks[indices]
        
        # Split into train and validation sets
        num_train = int(len(class_images) * 0.8)
        train_imgs, val_imgs = class_images[:num_train], class_images[num_train:]
        train_masks, val_masks = class_masks[:num_train], class_masks[num_train:]
       
        train_images_list.append(train_imgs)
        train_masks_list.append(train_masks)
        train_labels_list.append(np.full(len(train_imgs), label, dtype=np.int64))

        print(len(train_imgs))
        val_images_list.append(val_imgs)
        val_masks_list.append(val_masks)
        val_labels_list.append(np.full(len(val_imgs), label, dtype=np.int64))

    # Concatenate lists to form arrays
    
    train_images = np.concatenate(train_images_list, axis=0)
    val_images = np.concatenate(val_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)

    train_masks = np.concatenate(train_masks_list, axis=0)
    val_masks = np.concatenate(val_masks_list, axis=0)

    print (len(train_images), len(val_images))
    # Create PyTorch datasets
    train_dataset = CustomImageDataset(train_images, train_masks, train_labels, tran=True)
    val_dataset = CustomImageDataset(val_images, val_masks, val_labels, tran=False)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_images), shuffle=False)

    #convert all_slides to a string
    all_slides = [str(s) for s in all_slides]

    #remove duplicates
    all_slides = list(set(all_slides))

    #remove nan
    all_slides = [s for s in all_slides if s!='nan']

    #if a slide_id doesn't begin with a 0, add a 0 to the beginning of the slide_id
    all_slides = [s if s[0]=='0' else '0'+s for s in all_slides]

    
    #print all slides, separated by a comma, with no quotes around each slide
    print(','.join(all_slides))
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(data_path="/home/tessone/Documents/prism/data/training_11_20_24",target_size_per_class=3000)
    #count number of images in each class (val_loader second element)

    class_1=0
    class_0=0
    for i, (images, labels) in enumerate(train_loader):
        class_1+=torch.sum(labels).item()
        class_0+=len(labels)-torch.sum(labels).item()
    print(class_1)
    print(class_0)
