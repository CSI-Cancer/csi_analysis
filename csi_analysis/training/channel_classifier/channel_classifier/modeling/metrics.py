import torch

def calculate_top_5_accuracy(self, output, target):
        with torch.no_grad():
            maxk = min(5, output.size(1))
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            top_5_correct = correct[:5].reshape(-1).float().sum(0, keepdim=True)
            return top_5_correct.mul_(100.0 / target.size(0))