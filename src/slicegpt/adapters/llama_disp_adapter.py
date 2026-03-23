# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import logging

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
)

from slicegpt.gates import (
    apply_shortcut_gate,
    virtual_basic_operation,
    virtual_block_attn_operation,
    virtual_block_basic_operation,
    virtual_mlp_operation,
)
from slicegpt.model_adapter import LayerAdapter, ModelAdapter

from .llama_adapter import LlamaLayerAdapter, LlamaModelAdapter


class CompressedLlamaDecoderGateLayer(LlamaDecoderLayer):
    """
    LlamaDecoderLayer with SliceGPT shortcut rotations and DISP-style gates.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.config = config

        head_dim = config.hidden_size // config.num_attention_heads
        ex_dict_attn = {
            "dim_1": config.hidden_size,
            "dim_2": config.num_key_value_heads * head_dim,
            "head_dim": head_dim,
            "num_weight": 4,
        }
        ex_dict_mlp = {
            "dim_1": config.intermediate_size,
            "dim_2": config.hidden_size,
            "num_weight": 3,
        }

        self.use_gate = False
        self.use_virtual_gate = True
        self.virtual_attn_gate_1 = virtual_block_attn_operation(
            dim=config.hidden_size, ex_dict=ex_dict_attn
        )
        self.virtual_attn_gate_2 = virtual_basic_operation(dim=config.hidden_size)

        self.virtual_block_gate_1 = virtual_block_basic_operation(
            dim=config.hidden_size
        )
        self.virtual_gate = virtual_mlp_operation(
            dim=config.intermediate_size, ex_dict=ex_dict_mlp
        )
        self.virtual_block_gate_2 = virtual_basic_operation(dim=config.hidden_size)
        self._warned_pretraining_tp_gated_mlp = False

    def _apply_gated_mlp(self, hidden_states: Tensor) -> Tensor:
        mlp_inputs = self.virtual_block_gate_1(hidden_states)
        pretraining_tp = self.config.pretraining_tp

        if pretraining_tp <= 1:
            gate_hidden = self.mlp.gate_proj(mlp_inputs)
            up_hidden = self.mlp.up_proj(mlp_inputs)
            gate_hidden = self.mlp.act_fn(gate_hidden)
            if self.use_virtual_gate:
                gate_hidden = self.virtual_gate(gate_hidden)
                up_hidden = self.virtual_gate(up_hidden)
            hidden_states = self.mlp.down_proj(gate_hidden * up_hidden)
            return self.virtual_block_gate_2(hidden_states)

        if not self._warned_pretraining_tp_gated_mlp:
            logging.warning(
                "Using gated Llama MLP with pretraining_tp=%s", pretraining_tp
            )
            self._warned_pretraining_tp_gated_mlp = True

        if self.mlp.intermediate_size % pretraining_tp != 0:
            raise ValueError(
                f"intermediate_size ({self.mlp.intermediate_size}) must be divisible by pretraining_tp "
                f"({pretraining_tp}) for gated Llama MLP support."
            )

        slice_size = self.mlp.intermediate_size // pretraining_tp
        gate_proj_slices = self.mlp.gate_proj.weight.split(slice_size, dim=0)
        up_proj_slices = self.mlp.up_proj.weight.split(slice_size, dim=0)
        down_proj_slices = self.mlp.down_proj.weight.split(slice_size, dim=1)

        gate_hidden = torch.cat(
            [F.linear(mlp_inputs, weight) for weight in gate_proj_slices], dim=-1
        )
        up_hidden = torch.cat(
            [F.linear(mlp_inputs, weight) for weight in up_proj_slices], dim=-1
        )
        gate_hidden = self.mlp.act_fn(gate_hidden)

        if self.use_virtual_gate:
            gate_hidden = self.virtual_gate(gate_hidden)
            up_hidden = self.virtual_gate(up_hidden)

        intermediate_states = (gate_hidden * up_hidden).split(slice_size, dim=2)
        hidden_states = sum(
            F.linear(intermediate_states[i], down_proj_slices[i])
            for i in range(pretraining_tp)
        )
        return self.virtual_block_gate_2(hidden_states)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_value: tuple[Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn_inputs = hidden_states
        if self.use_gate:
            attn_inputs = self.virtual_attn_gate_1(attn_inputs)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=attn_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if self.use_gate:
            hidden_states = self.virtual_attn_gate_2(hidden_states)

        # TODO: Tune the residual merge for the attention branch.
        if self.attn_shortcut_Q is not None:
            shortcut_q = self.attn_shortcut_Q
            if self.use_gate:
                shortcut_q = apply_shortcut_gate(
                    shortcut_q, self.virtual_attn_gate_1, self.virtual_attn_gate_2
                )
            rotated_residual = matmul(residual, shortcut_q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_gate:
            hidden_states = self._apply_gated_mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        # TODO: Tune the residual merge for the MLP branch.
        if self.mlp_shortcut_Q is not None:
            shortcut_q = self.mlp_shortcut_Q
            if self.use_gate:
                shortcut_q = apply_shortcut_gate(
                    shortcut_q, self.virtual_block_gate_1, self.virtual_block_gate_2
                )
            rotated_residual = matmul(residual, shortcut_q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaDispLayerAdapter(LlamaLayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer) -> None:
        super().__init__(layer)

    @property
    def layer(self) -> Module:
        return self._layer


class LlamaDispModelAdapter(LlamaModelAdapter):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__(model)

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return LlamaConfig

    @property
    def original_layer_type(self) -> type:
        return LlamaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return LlamaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return LlamaDispLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedLlamaDecoderGateLayer

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(
        self, layer: Module, layer_idx: int | None
    ) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(
            self.config.torch_dtype
        )
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
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
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
        attn_implementation: str | None = None,
    ) -> ModelAdapter | None:
        if not (
            model_name.startswith("meta-llama/Llama-2")
            or model_name.startswith("meta-llama/Meta-Llama-3")
        ):
            return None

        model_kwargs = {
            "torch_dtype": dtype,
            "token": token,
            "local_files_only": local_files_only,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        model = LlamaForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.config.torch_dtype = dtype

        return cls(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
        attn_implementation: str | None = None,
    ) -> ModelAdapter | None:
        if not (
            model_name.startswith("meta-llama/Llama-2")
            or model_name.startswith("meta-llama/Meta-Llama-3")
        ):
            return None

        class UninitializedLlamaForCausalLM(LlamaForCausalLM):
            def _init_weights(self, _) -> None:
                pass

        config = LlamaConfig.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
        )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        model = UninitializedLlamaForCausalLM(config)
        model = model.to(dtype=dtype)

        return cls(model)
