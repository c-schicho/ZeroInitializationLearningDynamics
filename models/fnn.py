from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module, ELU, Linear, Sequential, init


class FNN(Module):

    def __init__(self, layer_features: List, act_fun: Callable = ELU(inplace=True)):
        super(FNN, self).__init__()

        if len(layer_features) < 2:
            raise ValueError("layer_features must at least provide input features and output features")

        fnn = []
        for i in range(len(layer_features) - 1):
            fnn.append(Linear(layer_features[i], layer_features[i + 1]))
            if i < len(layer_features) - 2:
                fnn.append(act_fun)

        self.model = Sequential(*fnn)

    def initialize(self, mode: str, mean: float = 0.0, std: float = 1.0, a: float = 0.0, b: float = 1.0):
        if mode not in ["default", "uniform", "normal"]:
            raise ValueError(f"mode {mode} is not supported")

        if mode == "default":
            return

        for layer in self.model:
            if isinstance(layer, Linear):
                init.zeros_(layer.weight)
                if mode == "uniform":
                    init.uniform_(layer.bias, a=a, b=b)
                elif mode == "normal":
                    init.normal_(layer.bias, mean=mean, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
