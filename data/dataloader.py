from typing import Union, Tuple

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda


def get_mnist_loader(
        train: bool,
        batch_size: int,
        flatten: bool = True,
        data_path: str = "./data",
        shuffle: bool = True,
        num_workers: int = 1
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    transforms = [
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ]

    if flatten:
        transforms.append(
            Lambda(lambda x: torch.flatten(x))
        )

    transform = Compose(transforms)

    if train:
        dataset = MNIST(data_path, train=True, transform=transform, download=True)
        train_dataset, val_dataset = random_split(dataset, [50_000, 10_000])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, val_loader

    else:
        test_dataset = MNIST(data_path, train=False, transform=transform, download=True)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
