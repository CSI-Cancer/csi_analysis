import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For image reading and processing
import numpy as np
import os
from wbc_classifier import CNNModel
from wbc_dataloader import get_data_loaders

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders['val']
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Backward pass and optimization in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # Deep copy the model if it has better accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Main function to set up data loaders, model, and start training
def main():
    # Hyperparameters
    num_epochs = 25
    batch_size = 8
    learning_rate = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_path="/home/tessone/Documents/prism/data/training_11_20_24",target_size_per_class=3000)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize model, criterion, and optimizer
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), '/mnt/deepstore/PRISM/pipeline/model/wbc_classifier_new.pth')

def main2():
    train_loader, val_loader = get_data_loaders(data_path="/home/tessone/Documents/prism/data/training_11_20_24",target_size_per_class=3000)


if __name__ == '__main__':
    main2()
