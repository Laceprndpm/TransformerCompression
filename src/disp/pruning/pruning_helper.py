import torch
import torch.nn as nn


def _is_llama_gate_layer(module: nn.Module) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "virtual_attn_gate_1",
            "virtual_attn_gate_2",
            "virtual_block_gate_1",
            "virtual_block_gate_2",
        )
    ) and hasattr(getattr(module, "mlp", None), "gate_proj") and hasattr(
        getattr(module, "self_attn", None), "o_proj"
    )


def _iter_llama_gate_layers(model: nn.Module):
    for module in model.modules():
        if _is_llama_gate_layer(module):
            yield module


class collect_info_reg_llama(nn.Module):
    def __init__(self, model, p=None, lam=4.0):
        super(collect_info_reg_llama, self).__init__()
        self.sum_ori_params = 0
        self.p = p
        self.lam = lam
        self.structures = []
        self.gate_type = []
        self.gate_names = []
        self.layer_specs = []

        for layer_idx, layer in enumerate(_iter_llama_gate_layers(model)):
            self.structures.extend(
                [
                    layer.virtual_attn_gate_1.dim,
                    layer.virtual_attn_gate_2.dim,
                    layer.virtual_block_gate_1.dim,
                    layer.virtual_block_gate_2.dim,
                ]
            )
            self.gate_type.extend(
                [
                    "attn_block",
                    "basic_gate",
                    "mlp_block",
                    "basic_gate",
                ]
            )
            self.gate_names.extend(
                [
                    f"model.layers.{layer_idx}.virtual_attn_gate_1",
                    f"model.layers.{layer_idx}.virtual_attn_gate_2",
                    f"model.layers.{layer_idx}.virtual_block_gate_1",
                    f"model.layers.{layer_idx}.virtual_block_gate_2",
                ]
            )
            self.sum_ori_params += layer.virtual_attn_gate_1.get_parameters()
            self.sum_ori_params += (
                layer.mlp.gate_proj.in_features * layer.mlp.gate_proj.out_features * 2
                + layer.mlp.down_proj.in_features * layer.mlp.down_proj.out_features
            )
            self.layer_specs.append(
                {
                    "attn_out_dim": layer.virtual_attn_gate_1.ex_dict["dim_2"],
                    "mlp_intermediate_dim": layer.mlp.gate_proj.out_features,
                }
            )

        print("Number of original parameters: %.3f" % (self.sum_ori_params / 10 ** 6))

    def count_pruned_parameters(self, vectors):
        sum_params = 0
        i = 0
        for layer_spec in self.layer_specs:
            attn_in_dim = vectors[i].sum()
            attn_out_dim = vectors[i + 1].sum()
            sum_params += attn_in_dim * 3 * layer_spec["attn_out_dim"] + attn_out_dim * layer_spec["attn_out_dim"]
            i += 2

            mlp_in_dim = vectors[i].sum()
            mlp_out_dim = vectors[i + 1].sum()
            mlp_mid_dim = layer_spec["mlp_intermediate_dim"]
            sum_params += mlp_in_dim * mlp_mid_dim * 2 + mlp_mid_dim * mlp_out_dim
            i += 2

        return sum_params

    def forward(self, vectors):
        sum_params = self.count_pruned_parameters(vectors)
        param_ratio = sum_params / self.sum_ori_params
        if param_ratio > self.p:
            clamped_p_ratio = torch.clamp(param_ratio, min=self.p)
            loss = torch.log(clamped_p_ratio / self.p)
        else:
            clamped_p_ratio = torch.clamp(param_ratio, max=self.p)
            loss = torch.log(self.p / clamped_p_ratio)

        return self.lam * loss


class collect_info_reg_phi2(nn.Module):
    def __init__(self, model, p=None, lam=4.0):
        super(collect_info_reg_phi2, self).__init__()
        self.sum_ori_params = 0
        self.p = p
        self.lam = lam
        self.in_dim_list = []
        self.out_dim_list = []
        self.num_w_list = []
        self.structures = []
        self.gate_type = []
        self.gate_names = []

        for name, m in model.named_modules():
            if type(m).__name__ == "virtual_block_basic_operation":
                self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append("mlp_block")
                self.gate_names.append(name)
            if type(m).__name__ == "virtual_mlp_operation":
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict["dim_1"])
                self.out_dim_list.append(m.ex_dict["dim_2"])
                self.num_w_list.append(m.ex_dict["num_weight"])
                self.structures.append(m.dim)
                self.gate_type.append("mlp")
                self.gate_names.append(name)
            if type(m).__name__ == "virtual_block_attn_operation":
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict["dim_1"])
                self.out_dim_list.append(m.ex_dict["dim_2"])
                self.num_w_list.append(m.ex_dict["num_weight"])
                self.structures.append(m.dim)
                self.head_dim = m.head_dim
                self.num_heads = m.dim
                self.gate_type.append("attn_block")
                self.gate_names.append(name)
            if type(m).__name__ == "virtual_basic_operation":
                self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append("basic_gate")
                self.gate_names.append(name)

        print("Number of original parameters: %.3f" % (self.sum_ori_params / 10 ** 6))

    def count_pruned_parameters(self, vectors):
        sum_params = 0
        i = 0
        while i < len(self.structures):
            if self.gate_type[i] == "attn_block":
                attn_in_dim = vectors[i].sum()
                attn_out_dim = vectors[i + 1].sum()
                current_params = attn_in_dim * 3 * self.out_dim_list[i] + attn_out_dim * self.out_dim_list[i]
                i += 2
                sum_params += current_params

            if self.gate_type[i] == "mlp_block":
                block_mlp_in_dim = vectors[i].sum()
                block_mlp_middle_dim = vectors[i + 1].sum()
                block_mlp_out_dim = vectors[i + 2].sum()
                current_params = block_mlp_in_dim * block_mlp_middle_dim + block_mlp_middle_dim * block_mlp_out_dim
                i += 3
                sum_params += current_params

        return sum_params

    def forward(self, vectors):
        sum_params = self.count_pruned_parameters(vectors)
        param_ratio = sum_params / self.sum_ori_params
        if param_ratio > self.p:
            clamped_p_ratio = torch.clamp(param_ratio, min=self.p)
            loss = torch.log(clamped_p_ratio / self.p)
        else:
            clamped_p_ratio = torch.clamp(param_ratio, max=self.p)
            loss = torch.log(self.p / clamped_p_ratio)

        return self.lam * loss


class help_functions_hn(nn.Module):
    def __init__(self, structures, constrained=None):
        self.structures = structures
        self.constrained = constrained

    def print_info(self, vectors):
        print(self.structures)
        config = []
        for i in range(len(vectors)):
            config.append(vectors[i].sum().item())
        print(config)

    def set_gate_vectors(self, model, vectors):
        llama_layers = list(_iter_llama_gate_layers(model))
        if llama_layers:
            ind = 0
            for layer in llama_layers:
                layer.virtual_attn_gate_1.set_vector_value(vectors[ind])
                ind += 1
                layer.virtual_attn_gate_2.set_vector_value(vectors[ind])
                ind += 1
                layer.virtual_block_gate_1.set_vector_value(vectors[ind])
                ind += 1
                layer.virtual_block_gate_2.set_vector_value(vectors[ind])
                ind += 1
            return

        if self.constrained == "structural":
            modules = list(model.modules())
            ind = 0
            model_dim = vectors[0]
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == "virtual_basic_operation":
                    m.set_vector_value(model_dim)
                if type(m).__name__ == "virtual_att_operation":
                    m.set_vector_value(vectors[ind + 1])
                    ind += 1
                if type(m).__name__ == "virtual_mlp_operation":
                    m.set_vector_value(vectors[ind + 1])
                    ind += 1
        elif self.constrained == "same":
            modules = list(model.modules())
            ind = 0
            model_dim = vectors[0]
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == "virtual_basic_operation":
                    m.set_vector_value(model_dim)
                if type(m).__name__ == "virtual_block_basic_operation":
                    m.set_vector_value(model_dim)
                if type(m).__name__ == "virtual_mlp_operation":
                    m.set_vector_value(vectors[ind + 1])
                    ind += 1
                if type(m).__name__ == "virtual_block_attn_operation":
                    m.set_vector_value(model_dim)
        else:
            modules = list(model.modules())
            ind = 0
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == "virtual_basic_operation":
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == "virtual_block_basic_operation":
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == "virtual_mlp_operation":
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == "virtual_block_attn_operation":
                    m.set_vector_value(vectors[ind])
                    ind += 1

    def set_gate_status(self, model, use_gate=False):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if hasattr(m, "use_gate"):
                m.use_gate = use_gate
