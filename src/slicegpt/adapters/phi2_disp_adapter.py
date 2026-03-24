# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi/modeling_phi.py
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# License updated to MIT license since 7e10f3e in https://huggingface.co/microsoft/phi-2/blob/main/LICENSE

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import LayerNorm, Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.phi.modeling_phi import PhiConfig, PhiDecoderLayer, PhiForCausalLM

from slicegpt.gates import (
    virtual_block_attn_operation,
    virtual_block_basic_operation,
    virtual_basic_operation,
)
from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressedPhiDecoderGateLayer(PhiDecoderLayer):
    """
    PhiDecoderLayer with shortcut_Q and DISP-style gates.
    The residual path is rotated by attn_shortcut_Q to stay consistent with SliceGPT.
    """

    def __init__(self, config: PhiConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)

        ex_dict_attn = {
            "dim_1": config.hidden_size,
            "dim_2": config.num_key_value_heads * (config.hidden_size // config.num_attention_heads),
            "head_dim": config.hidden_size // config.num_attention_heads,
            "num_weight": 4,
        }
        ex_dict_mlp = {
            "dim_1": config.intermediate_size,
            "dim_2": config.hidden_size,
            "num_weight": 2,
        }

        self.use_gate = False
        self.virtual_attn_gate_1 = virtual_block_attn_operation(dim=config.hidden_size, ex_dict=ex_dict_attn)
        self.virtual_attn_gate_2 = virtual_basic_operation(dim=config.hidden_size)

        self.virtual_block_gate_1 = virtual_block_basic_operation(dim=config.hidden_size, ex_dict=ex_dict_mlp)
        self.virtual_block_gate_2 = virtual_basic_operation(dim=config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        past_key_value: tuple[Tensor] | None = None,
    ) -> tuple:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_inputs = hidden_states
        if self.use_gate:
            attn_inputs = self.virtual_attn_gate_1(attn_inputs)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=attn_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_outputs = self.resid_dropout(attn_outputs)
        if self.use_gate:
            attn_outputs = self.virtual_attn_gate_2(attn_outputs)

        mlp_inputs = hidden_states
        if self.use_gate:
            mlp_inputs = self.virtual_block_gate_1(mlp_inputs)
            mlp_hidden = self.mlp.fc1(mlp_inputs)
            mlp_hidden = self.mlp.activation_fn(mlp_hidden)
            mlp_outputs = self.mlp.fc2(mlp_hidden)
            mlp_outputs = self.virtual_block_gate_2(mlp_outputs)
        else:
            mlp_outputs = self.mlp(mlp_inputs)

        feed_forward_hidden_states = self.resid_dropout(mlp_outputs)

        # try modify shortcut_Q
        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = attn_outputs + feed_forward_hidden_states + rotated_residual
        else:
            hidden_states = attn_outputs + feed_forward_hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Phi2DispLayerAdapter(LayerAdapter):
    def __init__(self, layer: PhiDecoderLayer) -> None:
        super().__init__()
        self._layer: PhiDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return None

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.dense

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.fc1]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.fc2


class Phi2DispModelAdapter(ModelAdapter):
    def __init__(self, model: PhiForCausalLM) -> None:
        super().__init__()
        self._model: PhiForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return PhiConfig

    @property
    def parallel_blocks(self) -> bool:
        return True

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return True

    @property
    def original_layer_type(self) -> type:
        return PhiDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LayerNorm

    @property
    def layer_adapter_type(self) -> type:
        return Phi2DispLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedPhiDecoderGateLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.final_layernorm
        assert pre_head_layernorm is not None
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
        
    ) -> ModelAdapter | None:
        if model_name != "microsoft/phi-2":
            return None

        model = PhiForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model.config.torch_dtype = dtype

        return Phi2DispModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if model_name != "microsoft/phi-2":
            return None

        class UninitializedPhiForCausalLM(PhiForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = PhiConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedPhiForCausalLM(config)
        model = model.to(dtype=dtype)

        return Phi2DispModelAdapter(model)
