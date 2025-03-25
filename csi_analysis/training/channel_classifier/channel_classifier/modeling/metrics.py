import torch

def top_k_accuracy(outputs, targets, k=2):
    # Dummy top-k implementation for illustration purposes
    topk = torch.topk(outputs, k, dim=1)[1]
    correct = topk.eq(targets.view(-1, 1).expand_as(topk))
    topk_acc = correct.float().sum().item() / targets.size(0)
    return topk_acc, None

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()