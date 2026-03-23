# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from inspect import get_annotations
from typing import Any, Protocol, runtime_checkable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig, OPTForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM

from slicegpt.adapters.llama_adapter import LlamaModelAdapter
from slicegpt.adapters.llama_disp_adapter import LlamaDispModelAdapter
from slicegpt.adapters.opt_adapter import OPTModelAdapter
from slicegpt.adapters.phi2_adapter import Phi2ModelAdapter
from slicegpt.adapters.phi3_adapter import Phi3ModelAdapter
from slicegpt.gates import apply_shortcut_gate, virtual_basic_operation
from slicegpt.model_adapter import ModelAdapter


@runtime_checkable
class HasShortcutsSequential(Protocol):
    mlp_shortcut_Q: Tensor | None
    attn_shortcut_Q: Tensor | None


@runtime_checkable
class HasShortcutsParallel(Protocol):
    attn_shortcut_Q: Tensor | None


@runtime_checkable
class HasWeight(Protocol):
    weight: Parameter


def _validate_protocol_attr(instance: Any, protocol: type, err_message: str) -> None:
    errors: list[str] = [err_message]
    for a, t in get_annotations(protocol).items():
        if not hasattr(instance, a):
            errors.append(f"Missing attribute '{a}")
        elif not isinstance(getattr(instance, a), t):
            errors.append(f"Attribute '{a}' is not an instance of {t}")
    if not isinstance(instance, protocol):
        errors.append(f"Does not implement {protocol}")
    success = len(errors) == 1
    assert success, "\n".join(errors)


# Name of the abstract test class can not start with "Test", because pytest tries
# to instantiate all such classes while collecting tests.
class ModelAdapterTestBase(ABC):
    @abstractmethod
    def create_adapter(self) -> ModelAdapter:
        raise NotImplementedError

    @pytest.fixture
    def model_adapter(self) -> ModelAdapter:
        return self.create_adapter()

    def test_convert_layer_to_compressed(self, model_adapter: ModelAdapter) -> None:
        for i, layer_adapter in enumerate(model_adapter.get_layers()):
            compressed_layer = model_adapter.convert_layer_to_compressed(layer_adapter.layer, i)
            assert isinstance(compressed_layer, Module), f"Converted compressed layer {i} is not a torch module"
            compressed_layer = model_adapter.convert_layer_to_compressed_and_register_buffers(layer_adapter.layer, i)
            if model_adapter.parallel_blocks:
                _validate_protocol_attr(
                    compressed_layer, HasShortcutsParallel, f"Converted compressed layer {i} is invalid"
                )
            else:
                _validate_protocol_attr(
                    compressed_layer, HasShortcutsSequential, f"Converted compressed layer {i} is invalid"
                )
            # TODO: test actual forward pass dependency on Q

    def test_layernorms_have_weight(self, model_adapter: ModelAdapter) -> None:
        pre_head_layernorm = model_adapter.get_pre_head_layernorm()
        assert isinstance(pre_head_layernorm, Module), "Pre-head layernorm is not a torch module"
        _validate_protocol_attr(pre_head_layernorm, HasWeight, "Pre-head layernorm is invalid")
        for i, layer_adapter in enumerate(model_adapter.get_layers()):
            first_layernorm = layer_adapter.get_first_layernorm()
            assert isinstance(first_layernorm, Module), f"First layernorm of layer {i} is not a torch module"
            _validate_protocol_attr(first_layernorm, HasWeight, f"First layernorm of layer {i} is invalid")
            second_layernorm = layer_adapter.get_second_layernorm()
            if second_layernorm is not None:
                assert isinstance(second_layernorm, Module), f"Second layernorm of layer {i} is not a torch module"
                _validate_protocol_attr(second_layernorm, HasWeight, f"Second layernorm of layer {i} is invalid")

    def test_embeddings_have_weight(self, model_adapter: ModelAdapter) -> None:
        for i, emb in enumerate(model_adapter.get_embeddings()):
            assert isinstance(emb, Module), f"Embeddings element {i} is not a torch module"
            _validate_protocol_attr(emb, HasWeight, f"Embeddings element {i} is invalid")

    def test_can_set_use_cache(self, model_adapter: ModelAdapter) -> None:
        old_use_cache = model_adapter.use_cache
        model_adapter.use_cache = not old_use_cache
        assert model_adapter.use_cache != old_use_cache, "use_cache.setter does not work"


class TestOPTAdapter(ModelAdapterTestBase):
    def create_adapter(self) -> OPTModelAdapter:
        config = OPTConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            ffn_dim=32,
            max_position_embeddings=16,
            num_attention_heads=2,
        )
        model = OPTForCausalLM(config)
        return OPTModelAdapter(model)


class TestLlamaAdapter(ModelAdapterTestBase):
    def create_adapter(self) -> LlamaModelAdapter:
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
        )
        model = LlamaForCausalLM(config)
        return LlamaModelAdapter(model)


def _make_llama_disp_adapter(*, pretraining_tp: int, intermediate_size: int = 32) -> LlamaDispModelAdapter:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        pretraining_tp=pretraining_tp,
    )
    model = LlamaForCausalLM(config)
    return LlamaDispModelAdapter(model)


def _set_gate_vectors_to_ones(compressed_layer: Module) -> None:
    compressed_layer.virtual_attn_gate_1.set_vector_value(
        torch.ones_like(compressed_layer.virtual_attn_gate_1.pruning_vector)
    )
    compressed_layer.virtual_attn_gate_2.set_vector_value(
        torch.ones_like(compressed_layer.virtual_attn_gate_2.pruning_vector)
    )
    compressed_layer.virtual_block_gate_1.set_vector_value(
        torch.ones_like(compressed_layer.virtual_block_gate_1.pruning_vector)
    )
    compressed_layer.virtual_gate.set_vector_value(
        torch.ones_like(compressed_layer.virtual_gate.pruning_vector)
    )
    compressed_layer.virtual_block_gate_2.set_vector_value(
        torch.ones_like(compressed_layer.virtual_block_gate_2.pruning_vector)
    )


def _reference_tp_gated_mlp(layer: Module, hidden_states: Tensor) -> Tensor:
    mlp_inputs = layer.virtual_block_gate_1(hidden_states)
    slice_size = layer.mlp.intermediate_size // layer.config.pretraining_tp
    gate_proj_slices = layer.mlp.gate_proj.weight.split(slice_size, dim=0)
    up_proj_slices = layer.mlp.up_proj.weight.split(slice_size, dim=0)
    down_proj_slices = layer.mlp.down_proj.weight.split(slice_size, dim=1)

    gate_hidden = torch.cat([F.linear(mlp_inputs, weight) for weight in gate_proj_slices], dim=-1)
    up_hidden = torch.cat([F.linear(mlp_inputs, weight) for weight in up_proj_slices], dim=-1)
    intermediate_states = (
        layer.virtual_gate(layer.mlp.act_fn(gate_hidden)) * layer.virtual_gate(up_hidden)
    ).split(slice_size, dim=2)
    outputs = [
        F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(layer.config.pretraining_tp)
    ]
    return layer.virtual_block_gate_2(sum(outputs))


def _reference_llama_disp_layer(layer: Module, hidden_states: Tensor) -> Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    attn_inputs = hidden_states
    if layer.use_gate:
        attn_inputs = layer.virtual_attn_gate_1(attn_inputs)

    hidden_states, _, _ = layer.self_attn(
        hidden_states=attn_inputs,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )
    if layer.use_gate:
        hidden_states = layer.virtual_attn_gate_2(hidden_states)

    if layer.attn_shortcut_Q is not None:
        shortcut_q = layer.attn_shortcut_Q
        if layer.use_gate:
            shortcut_q = apply_shortcut_gate(shortcut_q, layer.virtual_attn_gate_1, layer.virtual_attn_gate_2)
        hidden_states = torch.matmul(residual, shortcut_q) + hidden_states
    else:
        hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    if layer.use_gate:
        hidden_states = layer._apply_gated_mlp(hidden_states)
    else:
        hidden_states = layer.mlp(hidden_states)

    if layer.mlp_shortcut_Q is not None:
        shortcut_q = layer.mlp_shortcut_Q
        if layer.use_gate:
            shortcut_q = apply_shortcut_gate(shortcut_q, layer.virtual_block_gate_1, layer.virtual_block_gate_2)
        hidden_states = torch.matmul(residual, shortcut_q) + hidden_states
    else:
        hidden_states = residual + hidden_states

    return hidden_states


class TestLlamaDispAdapter(ModelAdapterTestBase):
    def create_adapter(self) -> LlamaDispModelAdapter:
        return _make_llama_disp_adapter(pretraining_tp=1)

    @pytest.mark.parametrize("pretraining_tp", [1, 2])
    def test_gated_forward_runs_for_supported_tp(self, pretraining_tp: int) -> None:
        model_adapter = _make_llama_disp_adapter(pretraining_tp=pretraining_tp)
        original_layer = model_adapter.get_layers()[0].layer
        compressed_layer = model_adapter.convert_layer_to_compressed(original_layer, 0)
        compressed_layer.use_gate = True

        hidden_states = torch.randn(2, 4, model_adapter.config.hidden_size)
        outputs = compressed_layer(hidden_states, use_cache=False)

        assert outputs[0].shape == hidden_states.shape

    def test_gated_tp_path_matches_reference_implementation(self) -> None:
        model_adapter = _make_llama_disp_adapter(pretraining_tp=2)
        original_layer = model_adapter.get_layers()[0].layer
        compressed_layer = model_adapter.convert_layer_to_compressed(original_layer, 0)
        compressed_layer.use_gate = True
        _set_gate_vectors_to_ones(compressed_layer)

        hidden_states = torch.randn(2, 4, model_adapter.config.hidden_size)
        normalized_hidden_states = compressed_layer.post_attention_layernorm(hidden_states)

        actual = compressed_layer._apply_gated_mlp(normalized_hidden_states)
        expected = _reference_tp_gated_mlp(compressed_layer, normalized_hidden_states)

        torch.testing.assert_close(actual, expected)

    def test_non_gated_path_matches_original_layer(self) -> None:
        model_adapter = _make_llama_disp_adapter(pretraining_tp=2)
        original_layer = model_adapter.get_layers()[0].layer
        compressed_layer = model_adapter.convert_layer_to_compressed(original_layer, 0)
        compressed_layer.use_gate = False
        original_layer.eval()
        compressed_layer.eval()

        hidden_states = torch.randn(2, 4, model_adapter.config.hidden_size)
        actual = compressed_layer(hidden_states, use_cache=False)[0]
        expected = original_layer(hidden_states, use_cache=False)[0]

        torch.testing.assert_close(actual, expected)

    def test_gated_tp_path_raises_for_invalid_intermediate_size(self) -> None:
        model_adapter = _make_llama_disp_adapter(pretraining_tp=3, intermediate_size=10)
        original_layer = model_adapter.get_layers()[0].layer
        compressed_layer = model_adapter.convert_layer_to_compressed(original_layer, 0)
        compressed_layer.use_gate = True

        hidden_states = torch.randn(2, 4, model_adapter.config.hidden_size)
        normalized_hidden_states = compressed_layer.post_attention_layernorm(hidden_states)

        with pytest.raises(ValueError, match="intermediate_size"):
            compressed_layer._apply_gated_mlp(normalized_hidden_states)

    def test_apply_shortcut_gate_returns_original_matrix_for_all_one_vectors(self) -> None:
        input_gate = virtual_basic_operation(dim=3)
        output_gate = virtual_basic_operation(dim=4)
        shortcut = torch.randn(3, 4)

        actual = apply_shortcut_gate(shortcut, input_gate, output_gate)

        torch.testing.assert_close(actual, shortcut)

    def test_apply_shortcut_gate_zeroes_expected_rows_and_columns(self) -> None:
        input_gate = virtual_basic_operation(dim=3)
        output_gate = virtual_basic_operation(dim=4)
        input_gate.set_vector_value(torch.tensor([1.0, 0.0, 1.0]))
        output_gate.set_vector_value(torch.tensor([1.0, 0.0, 1.0, 0.0]))
        shortcut = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        actual = apply_shortcut_gate(shortcut, input_gate, output_gate)
        expected = shortcut * torch.tensor([[1.0], [0.0], [1.0]]) * torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        torch.testing.assert_close(actual, expected)

    def test_apply_shortcut_gate_raises_for_dimension_mismatch(self) -> None:
        input_gate = virtual_basic_operation(dim=2)
        output_gate = virtual_basic_operation(dim=4)
        shortcut = torch.randn(3, 4)

        with pytest.raises(ValueError, match="input gate dimension"):
            apply_shortcut_gate(shortcut, input_gate, output_gate)

    def test_gated_shortcut_path_matches_reference_for_all_one_vectors(self) -> None:
        model_adapter = _make_llama_disp_adapter(pretraining_tp=1)
        original_layer = model_adapter.get_layers()[0].layer
        compressed_layer = model_adapter.convert_layer_to_compressed(original_layer, 0)
        compressed_layer.use_gate = True
        compressed_layer.eval()
        _set_gate_vectors_to_ones(compressed_layer)

        hidden_states = torch.randn(2, 4, model_adapter.config.hidden_size)

        actual = compressed_layer(hidden_states, use_cache=False)[0]
        expected = _reference_llama_disp_layer(compressed_layer, hidden_states)

        torch.testing.assert_close(actual, expected)

    def test_gated_shortcut_path_matches_reference_for_sparse_vectors(self) -> None:
        model_adapter = _make_llama_disp_adapter(pretraining_tp=1)
        original_layer = model_adapter.get_layers()[0].layer
        compressed_layer = model_adapter.convert_layer_to_compressed(original_layer, 0)
        compressed_layer.use_gate = True
        compressed_layer.eval()

        compressed_layer.virtual_attn_gate_1.set_vector_value(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
        compressed_layer.virtual_attn_gate_2.set_vector_value(torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
        compressed_layer.virtual_block_gate_1.set_vector_value(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
        compressed_layer.virtual_block_gate_2.set_vector_value(torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]))

        hidden_states = torch.randn(2, 4, model_adapter.config.hidden_size)

        actual = compressed_layer(hidden_states, use_cache=False)[0]
        expected = _reference_llama_disp_layer(compressed_layer, hidden_states)

        torch.testing.assert_close(actual, expected)


class TestPhi2Adapter(ModelAdapterTestBase):
    def create_adapter(self) -> Phi2ModelAdapter:
        # a tiny phi, just to test adapter.
        config = PhiConfig(
            vocab_size=32, hidden_size=8, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2
        )
        model = PhiForCausalLM(config)
        return Phi2ModelAdapter(model)


class TestPhi3Adapter(ModelAdapterTestBase):
    def create_adapter(self) -> Phi3ModelAdapter:
        config = Phi3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
            # must set 0 <= pad_token_id <= vocab_size unless set to None in config
            pad_token_id=None,
        )
        model = Phi3ForCausalLM(config)
        return Phi3ModelAdapter(model)
