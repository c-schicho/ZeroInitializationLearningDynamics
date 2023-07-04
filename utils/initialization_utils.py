import math
from typing import Union

import numpy as np
import torch
from torch.nn import Parameter, init, Linear, Conv2d


def initialize_model(model, mode: str, scale_factor: float = 1., softmax_init: bool = False,
                     a: Union[float, None] = None, b: Union[float, None] = None):
    if mode not in ["default", "uniform", "normal", "normal_in_features", "deterministic"]:
        raise ValueError(f"mode {mode} is not supported")

    if mode == "uniform" and (a is None or b is None):
        raise ValueError(f"'a' and 'b' must be provided when mode is uniform")

    if mode == "deterministic" and (a is None or b is None):
        raise ValueError(f"'a' and 'b' must be provided when mode is deterministic")

    if mode == "default":
        return

    last_idx = len(model) - 1
    for idx, layer in enumerate(model):
        if isinstance(layer, Linear) or isinstance(layer, Conv2d):
            init.zeros_(layer.weight)

            if softmax_init and idx == last_idx:
                layer.bias = Parameter(torch.full_like(layer.bias, math.log(0.1), requires_grad=True))
            elif mode == "normal":
                init.normal_(layer.bias, std=scale_factor)
            elif mode == "uniform":
                init.uniform_(layer.bias, a=a, b=b)
            elif mode == "deterministic":
                layer.bias = Parameter(
                    torch.tensor(
                        np.linspace(a, b, layer.bias.size(dim=0)),
                        dtype=layer.bias.dtype,
                        requires_grad=True
                    )
                )
            else:
                if isinstance(layer, Linear):
                    std = math.sqrt(1 / layer.in_features) * scale_factor
                    init.normal_(layer.bias, std=std)
                else:
                    in_features = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    std = math.sqrt(1 / in_features) * scale_factor
                    init.normal_(layer.bias, std=std)
