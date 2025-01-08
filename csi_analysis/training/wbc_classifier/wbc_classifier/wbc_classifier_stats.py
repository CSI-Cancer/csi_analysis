import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from wbc_classifier import CNNModel
from wbc_dataloader import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt

# model_path
model_path = '/mnt/deepstore/PRISM/pipeline/model/wbc_classifier_new.pth'
train_loader, val_loader = get_data_loaders(data_path="/home/tessone/Documents/prism/data/training_11_20_24",target_size_per_class=3000)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel()
model.load_state_dict(torch.load(model_path))
model.to(device)

model.eval()
all_targets = []
all_probs = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs) # shape (batch_size, 2)
        # Convert logits to probabilities
        # probs[:, 1] will give the probability of class 1 (positive class)
        probs = F.softmax(logits, dim=1)[:, 1]

        # Move to CPU for sklearn
        probs = probs.cpu().numpy()
        targets = targets.cpu().numpy()

        # Collect results
        all_targets.extend(targets)
        all_probs.extend(probs)

all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

#Compute accuracy
preds = (all_probs > 0.5).astype(int)
accuracy = np.mean(preds == all_targets)
print(f"Accuracy: {accuracy:.4f}")

# Compute ROC AUC
fpr, tpr, _ = roc_curve(all_targets, all_probs)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall AUC
precision, recall, _ = precision_recall_curve(all_targets, all_probs)
pr_auc = average_precision_score(all_targets, all_probs)

print(f"AUROC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('roc_curve.png')

# Plot Precision-Recall Curve
plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('pr_curve.png')