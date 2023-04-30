import math
from typing import Callable, List, Dict

import torch
from torch import Tensor
from torch.nn import Module, ELU, Linear, Conv2d, Sequential, init, Flatten, Parameter


class CNN(Module):

    def __init__(self, conv_layer_features: List[Dict], linear_layer_features: List[Dict],
                 act_fun: Callable = ELU(inplace=True)):
        super(CNN, self).__init__()

        if len(conv_layer_features) < 1:
            raise ValueError("you must at least provide one conv layer")

        if len(linear_layer_features) < 1:
            raise ValueError("you must at least provide one linear layer")

        cnn = []
        for layer_features in conv_layer_features:
            cnn.append(Conv2d(**layer_features))
            cnn.append(act_fun)

        cnn.append(Flatten())

        for layer_features in linear_layer_features:
            cnn.append(Linear(**layer_features))
            cnn.append(act_fun)

        cnn.pop()
        self.model = Sequential(*cnn)

    def initialize(self, mode: str, scale_factor: float = 1., softmax_init: bool = False):
        if mode not in ["default", "uniform", "normal", "normal_in_features"]:
            raise ValueError(f"mode {mode} is not supported")

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
                    raise NotImplementedError
                else:
                    if isinstance(layer, Linear):
                        std = math.sqrt(1 / layer.in_features) * scale_factor
                        init.normal_(layer.bias, std=std)
                    else:
                        in_features = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                        std = math.sqrt(1 / in_features) * scale_factor
                        init.normal_(layer.bias, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
