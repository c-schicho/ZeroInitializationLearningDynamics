import math
from typing import Union

import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ELU, init, Parameter, Dropout


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
        if mode not in ["default", "uniform", "normal", "normal_in_features"]:
            raise ValueError(f"mode {mode} is not supported")

        if mode == "uniform" and (a is None or b is None):
            raise ValueError(f"'a' and 'b' must be provided when mode is uniform")

        if mode == "default":
            return

        last_idx = len(self.model) - 1
        for idx, layer in enumerate(self.model):
            if isinstance(layer, Linear) or isinstance(layer, Conv2d):
                init.zeros_(layer.weight)

                if softmax_init and idx == last_idx:
                    layer.bias = Parameter(torch.full_like(layer.bias, math.log(0.1), requires_grad=True))

                if mode == "normal":
                    init.normal_(layer.bias, std=scale_factor)
                elif mode == "uniform":
                    init.uniform_(layer.bias, a=a, b=b)
                else:
                    if isinstance(layer, Linear):
                        std = math.sqrt(1 / layer.in_features) * scale_factor
                        init.normal_(layer.bias, std=std)
                    else:
                        in_features = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                        std = math.sqrt(1 / in_features) * scale_factor
                        init.normal_(layer.bias, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
