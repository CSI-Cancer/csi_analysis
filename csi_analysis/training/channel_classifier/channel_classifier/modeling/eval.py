import torch
from torch.utils.data import DataLoader

import os
from pathlib import Path
import sys
from tqdm import tqdm

import pandas as pd
import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1]))
# COnfig related imports
from config import (
    MODELS_DIR, INTERIM_DATA_DIR, test_sweep, pred_encoder, classes_to_merge
)
#Model imports
from modeling.model import get_model

#Data loader related imports
from modeling.loader import CustomDataset, get_transforms, get_data_loaders

#Metrics imports
from modeling.metrics import accuracy, top_k_accuracy
import matplotlib.pyplot as plt
from collections import defaultdict


class Evaluator:
    def __init__(self, sweep):
        self.sweep = sweep
        # Load the test dataset
        self.test_loader = get_data_loaders(test_sweep)
        self.class_map = pred_encoder
        self.merged_classes = classes_to_merge
        

        # Load the model
        self.load_model()

    def load_model(self):
        self.model = get_model()
        checkpoint = torch.load(self.sweep.model_path, map_location=self.sweep.device)
        # Remove 'module.' prefix if present
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.sweep.device)
        self.model.eval()
    
    def evaluate(self):
        total_accuracy = 0
        total_top_k_accuracy = 0
        total_samples = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for inputs, targets, prefix in self.test_loader:
                inputs, targets, prefix = inputs.to(self.sweep.device), targets.to(self.sweep.device), prefix.to(self.sweep.device)
                outputs = self.model(inputs, prefix)

                total_accuracy += accuracy(outputs, targets) * targets.size(0)
                top_k_acc, _ = top_k_accuracy(outputs, targets, k=self.sweep.top_k)
                total_top_k_accuracy += top_k_acc * targets.size(0)
                total_samples += targets.size(0)

                _, preds = torch.max(outputs, 1)
                for target, pred in zip(targets, preds):
                    if target == pred:
                        class_correct[target.item()] += 1
                    class_total[target.item()] += 1
        merged_corr = 0
        merged_total = 0
        merge_indices = []
        for i in range(self.sweep.num_classes):
            class_name = self.class_map[i]
            if class_name in self.merged_classes:
                merged_corr += class_correct[i]
                merged_total += class_total[i]
                merge_indices.append(i)

        # Remove merged classes from dictionaries
        for i in merge_indices:
            del class_correct[i]
            del class_total[i]

        # Calculate average metrics
        avg_accuracy = total_accuracy / total_samples
        avg_top_k_accuracy = total_top_k_accuracy / total_samples

        print(f"Accuracy: {avg_accuracy:.4f}")
        print(f"Top-{self.sweep.top_k} Accuracy: {avg_top_k_accuracy:.4f}")

        # Calculate accuracy for each remaining class (which contain "D")
        class_accuracies = {}
        for i in range(self.sweep.num_classes):
            # Skip non-D classes since they've been merged
            if i in class_correct:
                acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                class_accuracies[self.class_map[i]] = acc

        # Calculate accuracy for the merged class "D-"
        merged_acc = merged_corr / merged_total if merged_total > 0 else 0
        class_accuracies["D-"] = merged_acc
        print(f"Merged class 'D-' Accuracy: {merged_acc:.4f}")

        # Sort class accuracies first by descending accuracy then by class name
        sorted_class_accuracies = dict(sorted(class_accuracies.items(), key=lambda item: (-item[1], item[0])))

        # Plot the accuracy for each class
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_class_accuracies.keys(), sorted_class_accuracies.values())
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Each Class')
        plt.xticks(rotation=45)
        plt.yticks([i/10 for i in range(11)])
        plt.tight_layout()

        # Annotate each bar with the y-value (accuracy)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

        # Save the plot
        plt.savefig('class_accuracies.png')

if __name__ == "__main__":
    evaluator = Evaluator(test_sweep)
    evaluator.evaluate()

