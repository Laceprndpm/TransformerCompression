import torch
import torch.nn as nn
from torch import Tensor, matmul
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaSdpaAttention,
)


def compiled_index_add(base: Tensor, values: Tensor, index: Tensor) -> Tensor:
    return base.index_add(-1, index, values)


def compiled_index_select(values: Tensor, index: Tensor) -> Tensor:
    return torch.index_select(values, -1, index)


def _get_layer_cfg(config: LlamaConfig, layer_idx: int) -> dict:
    layer_cfgs = getattr(config, "pruned_layer_configs", None)
    if layer_cfgs is None:
        raise ValueError("Missing pruned_layer_configs in config for PrunedLlamaForCausalLM.")
    return layer_cfgs[layer_idx]


class _PrunedAttentionMixin:
    def _register_pruned_indices(self, layer_cfg: dict) -> None:
        self.register_buffer(
            "select_index",
            torch.tensor(layer_cfg["attn_select_index"], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "copy_index",
            torch.tensor(layer_cfg["attn_copy_index"], dtype=torch.long),
            persistent=True,
        )

    def _replace_attention_linears(self, config: LlamaConfig, layer_cfg: dict) -> None:
        input_size = len(layer_cfg["attn_select_index"])
        output_size = len(layer_cfg["attn_copy_index"])
        q_output = self.num_heads * self.head_dim
        kv_output = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(input_size, q_output, bias=config.attention_bias)
        self.k_proj = nn.Linear(input_size, kv_output, bias=config.attention_bias)
        self.v_proj = nn.Linear(input_size, kv_output, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, output_size, bias=config.attention_bias)

    def _select_hidden_states(self, hidden_states: Tensor) -> Tensor:
        return compiled_index_select(hidden_states, self.select_index.to(hidden_states.device)).contiguous()


class PrunedLlamaAttention(_PrunedAttentionMixin, LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, layer_cfg: dict):
        super().__init__(config, layer_idx)
        self._register_pruned_indices(layer_cfg)
        self._replace_attention_linears(config, layer_cfg)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        hidden_states = self._select_hidden_states(hidden_states)
        return super().forward(hidden_states, *args, **kwargs)


class PrunedLlamaFlashAttention2(_PrunedAttentionMixin, LlamaFlashAttention2):
    def __init__(self, config: LlamaConfig, layer_idx: int, layer_cfg: dict):
        super().__init__(config, layer_idx)
        self._register_pruned_indices(layer_cfg)
        self._replace_attention_linears(config, layer_cfg)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        hidden_states = self._select_hidden_states(hidden_states)
        return super().forward(hidden_states, *args, **kwargs)


class PrunedLlamaSdpaAttention(_PrunedAttentionMixin, LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, layer_cfg: dict):
        super().__init__(config, layer_idx)
        self._register_pruned_indices(layer_cfg)
        self._replace_attention_linears(config, layer_cfg)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        hidden_states = self._select_hidden_states(hidden_states)
        return super().forward(hidden_states, *args, **kwargs)


PRUNED_LLAMA_ATTENTION_CLASSES = {
    "eager": PrunedLlamaAttention,
    "flash_attention_2": PrunedLlamaFlashAttention2,
    "sdpa": PrunedLlamaSdpaAttention,
}


class PrunedLlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, layer_cfg: dict):
        super().__init__(config)
        self.register_buffer(
            "select_index",
            torch.tensor(layer_cfg["mlp_select_index"], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "copy_index",
            torch.tensor(layer_cfg["mlp_copy_index"], dtype=torch.long),
            persistent=True,
        )
        input_size = len(layer_cfg["mlp_select_index"])
        intermediate_size = len(layer_cfg["mlp_intermediate_index"])
        output_size = len(layer_cfg["mlp_copy_index"])
        self.hidden_size = input_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = compiled_index_select(x, self.select_index.to(x.device)).contiguous()
        return super().forward(x)


class PrunedRMSN(nn.Module):
    def __init__(self, mean_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class PrunedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        layer_cfg = _get_layer_cfg(config, layer_idx)
        super().__init__(config, layer_idx)
        attn_impl = getattr(config, "_attn_implementation", "eager") or "eager"
        attention_cls = PRUNED_LLAMA_ATTENTION_CLASSES.get(attn_impl, PrunedLlamaAttention)
        self.self_attn = attention_cls(config=config, layer_idx=layer_idx, layer_cfg=layer_cfg)
        self.mlp = PrunedLlamaMLP(config, layer_cfg)
        self.input_layernorm = PrunedRMSN(layer_cfg["input_size"], eps=config.rms_norm_eps)
        self.post_attention_layernorm = PrunedRMSN(layer_cfg["attn_output_size"], eps=config.rms_norm_eps)
        self.attn_shortcut_Q = nn.Parameter(
            torch.empty(layer_cfg["input_size"], layer_cfg["attn_output_size"]),
            requires_grad=False,
        )
        self.mlp_shortcut_Q = nn.Parameter(
            torch.empty(layer_cfg["mlp_input_size"], layer_cfg["mlp_output_size"]),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        attn_residual = matmul(residual, self.attn_shortcut_Q)
        hidden_states = compiled_index_add(
            attn_residual,
            hidden_states.to(attn_residual.dtype),
            self.self_attn.copy_index.to(hidden_states.device),
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        mlp_residual = matmul(residual, self.mlp_shortcut_Q)
        hidden_states = compiled_index_add(
            mlp_residual,
            hidden_states.to(mlp_residual.dtype),
            self.mlp.copy_index.to(hidden_states.device),
        )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class PrunedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        layer_cfgs = getattr(config, "pruned_layer_configs", None)
        if layer_cfgs is None:
            raise ValueError("Missing pruned_layer_configs in config for PrunedLlamaForCausalLM.")

        embedding_dim = int(getattr(config, "pruned_embedding_dim"))
        final_hidden_size = int(getattr(config, "pruned_final_hidden_size"))

        self.model.embed_tokens = nn.Embedding(config.vocab_size, embedding_dim, self.model.padding_idx)
        self.model.layers = nn.ModuleList(
            [PrunedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.model.norm = PrunedRMSN(final_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(final_hidden_size, config.vocab_size, bias=False)
