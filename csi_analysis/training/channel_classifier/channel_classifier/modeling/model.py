import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.resnet import ResNet, BasicBlock
from modeling.densenet import DenseNet
from modeling.genericcnn import GenericCNN
# Classes are returned as an incrementing values from 0
CLASSES = [
    "D",
    "CK",
    "CD",
    "V",
    "CK|CD|V",
    "CK|CD",
    "D|CK|CD|V",
    "CK|V",
    "D|CK|CD",
    "D|CK|V",
    "D|V",
    "D|CD|V",
    "D|CD",
    "D|CK",
    "CD|V",
]


def ResNet4(dropout=0.5, num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, dropout=dropout)


def DenseNet121(dropout, num_classes):
    return DenseNet(num_classes=num_classes,
                    growth_rate=32,
                    num_layers_per_block=6,
                    dropout_rate=dropout)

def get_model(dropout=0.5, num_classes=15, model_name="generic"):
    if model_name == "resnet":
        return ResNet4(dropout=dropout, num_classes=num_classes)
    elif model_name == "densenet":
        return DenseNet121(dropout=dropout, num_classes=num_classes)
    elif model_name == "generic":
        return GenericCNN(dropout=dropout, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")

def get_pretrain_model(dropout=0.5, embedding_dim=128, model_name="generic"):
    if model_name == "resnet":
        raise ValueError("ResNet is not supported for pretraining")
    elif model_name == "densenet":
        raise ValueError("DenseNet is not supported for pretraining")
    elif model_name == "generic":
        return GenericCNN(dropout=dropout, num_classes=embedding_dim)