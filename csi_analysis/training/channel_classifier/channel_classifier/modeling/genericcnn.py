import torch
import torch.nn as nn
import torch.nn.functional as F

class GenericCNN(nn.Module):
    def __init__(self, dropout=0.2, num_classes=15):
        super(GenericCNN, self).__init__()
        # Convolutional layers adjustments
        self.conv1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # Adjusted output channels
        self.bn4 = nn.BatchNorm2d(512)  # Adjusted for 512 output channels
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # Adjusted output channels
        self.bn5 = nn.BatchNorm2d(512)  # Adjusted for 512 output channels
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)



    def forward(self, x, return_embeddings=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = x.view(-1, 512)  # Flatten the tensor
        
        # Get embeddings from the first fully connected layer
        x = self.dropout(F.relu(self.fc1(x)))
        embeddings = self.fc2(x)  # Use fc2 output as embeddings
        
        if return_embeddings:
            return embeddings
        
        # For classification, apply softmax
        return F.log_softmax(embeddings, dim=1)