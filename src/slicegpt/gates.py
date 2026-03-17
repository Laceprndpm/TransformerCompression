# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class virtual_basic_operation(nn.Module):
    def __init__(self, dim, ex_dict=None):
        super().__init__()
        self.dim = dim
        self.pruning_vector = torch.ones(dim)
        self.ex_dict = ex_dict or {}

    def forward(self, input):
        if len(input.size()) == 4:
            p_v = self.pruning_vector[None, None, None, :]
        elif len(input.size()) == 3:
            p_v = self.pruning_vector[None, None, :]
        elif len(input.size()) == 2:
            p_v = self.pruning_vector[None, :]
        else:
            return input
        p_v = p_v.to(input.device)
        return p_v.expand_as(input) * input

    def set_vector_value(self, value):
        assert value.squeeze().size() == self.pruning_vector.squeeze().size()
        self.pruning_vector = value.squeeze() if value is not None else value

    def get_parameters(self):
        return 0


class virtual_block_basic_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict=None):
        super().__init__(dim=dim, ex_dict=ex_dict)


class virtual_att_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict=None):
        super().__init__(dim=dim, ex_dict=ex_dict)
        self.head_dim = ex_dict["head_dim"] if ex_dict else None

    def get_parameters(self):
        return self.ex_dict["dim_1"] * self.ex_dict["dim_2"] * self.ex_dict["num_weight"]

    def forward(self, input):
        if len(input.size()) == 4:
            p_v = self.pruning_vector[None, None, :, None]
            p_v = p_v.to(input.device)
            return p_v.expand_as(input) * input
        return input


class virtual_block_attn_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict=None):
        super().__init__(dim=dim, ex_dict=ex_dict)
        self.head_dim = ex_dict["head_dim"] if ex_dict else None

    def get_parameters(self):
        return self.ex_dict["dim_1"] * self.ex_dict["dim_2"] * self.ex_dict["num_weight"]


class virtual_mlp_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict=None):
        super().__init__(dim=dim, ex_dict=ex_dict)

    def get_parameters(self):
        return self.ex_dict["dim_1"] * self.ex_dict["dim_2"] * self.ex_dict["num_weight"]
