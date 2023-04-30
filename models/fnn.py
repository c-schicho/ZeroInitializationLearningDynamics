import math
from typing import List, Dict, Callable

import torch
from torch import Tensor
from torch.nn import Module, ELU, Linear, Sequential, init, Parameter


class FNN(Module):

    def __init__(self, linear_layer_features: List[Dict], act_fun: Callable = ELU(inplace=True)):
        super(FNN, self).__init__()

        if len(linear_layer_features) < 1:
            raise ValueError("you must at least provide one layer")

        fnn = []
        for layer_features in linear_layer_features:
            fnn.append(Linear(**layer_features))
            fnn.append(act_fun)

        fnn.pop()
        self.model = Sequential(*fnn)

    def initialize(self, mode: str, scale_factor: float = 1., softmax_init: bool = False):
        if mode not in ["default", "uniform", "normal", "normal_in_features"]:
            raise ValueError(f"mode {mode} is not supported")

        if mode == "default":
            return

        last_idx = len(self.model) - 1
        for idx, layer in enumerate(self.model):
            if isinstance(layer, Linear):
                init.zeros_(layer.weight)

                if softmax_init and idx == last_idx:
                    layer.bias = Parameter(torch.full_like(layer.bias, math.log(0.1), requires_grad=True))

                elif mode == "normal":
                    init.normal_(layer.bias, std=scale_factor)
                elif mode == "uniform":
                    raise NotImplementedError
                else:
                    std = math.sqrt(1 / layer.in_features) * scale_factor
                    init.normal_(layer.bias, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
