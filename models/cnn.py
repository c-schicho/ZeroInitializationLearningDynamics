from typing import Callable, List, Dict, Union

import torch
from torch import Tensor
from torch.nn import Module, ELU, Linear, Conv2d, Sequential, Flatten

from utils import initialize_model


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

    def initialize(self, mode: str, scale_factor: float = 1., softmax_init: bool = False, a: Union[float, None] = None,
                   b: Union[float, None] = None):
        initialize_model(self.model, mode, scale_factor, softmax_init, a, b)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
