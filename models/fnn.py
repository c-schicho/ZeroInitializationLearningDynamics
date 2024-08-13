from typing import List, Dict, Callable, Union

import torch
from torch import Tensor
from torch.nn import Module, ELU, Linear, Sequential

from utils import initialize_model


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

    def initialize(self, mode: str, scale_factor: float = 1., mean: float = 0., softmax_init: bool = False,
                   a: Union[float, None] = None, b: Union[float, None] = None):
        initialize_model(self.model, mode, scale_factor, mean, softmax_init, a, b)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
