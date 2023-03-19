import torch
from torch import Tensor


def calculate_accuracy(outputs: Tensor, targets: Tensor) -> float:
    _, predictions = torch.max(outputs, dim=1)
    return torch.sum(predictions == targets).item() / len(predictions)
