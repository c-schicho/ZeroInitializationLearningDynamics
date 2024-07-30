from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class ExperimentConfig:
    model_name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    epochs: int
    lr: float = 0.001
    runs: int = 5
    train_summary: bool = True
    test_after_epoch: bool = True
