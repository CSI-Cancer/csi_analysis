import torch

def top_k_accuracy(outputs, targets, k=5):
    _, preds = outputs.topk(k, 1, True, True)
    correct = preds.eq(targets.view(-1, 1).expand_as(preds))
    correct_k = correct[:, :k].sum().item()
    top_k_classes = preds.tolist()  # Convert tensor to list for easier handling
    return correct_k / targets.size(0), top_k_classes

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)