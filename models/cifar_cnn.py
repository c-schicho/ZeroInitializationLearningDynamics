from typing import Union

import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ELU, Dropout

from utils import initialize_model


class CIFARCNN(Module):

    def __init__(self, activation_fun=None):
        super(CIFARCNN, self).__init__()
        self.cnn1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.cnn2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnn4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.dropout = Dropout(0.25)
        self.fnn1 = Linear(128 * 2 * 2, 256)
        self.fnn2 = Linear(256, 64)
        self.fnn3 = Linear(64, 10)
        self.act = activation_fun if activation_fun is not None else ELU(inplace=True)

        self.model = Sequential(
            self.cnn1,
            self.act,
            self.pool,
            self.cnn2,
            self.act,
            self.pool,
            self.cnn3,
            self.act,
            self.pool,
            self.cnn4,
            self.act,
            self.pool,
            Flatten(),
            self.fnn1,
            self.act,
            self.dropout,
            self.fnn2,
            self.act,
            self.dropout,
            self.fnn3
        )

    def initialize(self, mode: str, scale_factor: float = 1., softmax_init: bool = False, a: Union[float, None] = None,
                   b: Union[float, None] = None):
        initialize_model(self.model, mode, scale_factor, softmax_init, a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
