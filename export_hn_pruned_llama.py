import argparse
import copy
import json
import logging
import os
import pathlib

import torch

from slicegpt import hf_utils
from slicegpt.config import config
from slicegpt.pruned.modeling_llama_pruned import PrunedLlamaForCausalLM
from src.disp.pruning.hypernetwork import hypernetwork
from src.disp.pruning.pruning_helper import collect_info_reg_llama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model identifier, e.g. meta-llama/Llama-2-7b-hf.")
    parser.add_argument("--sliced-model-path", type=str, required=True, help="Path to the step1 sliced model directory.")
    parser.add_argument("--hn-ckpt-path", type=str, required=True, help="Path to the hypernetwork checkpoint.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for the final pruned model.")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Step1 sliced model sparsity.")
    parser.add_argument("--round-interval", type=int, default=8)
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", None))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def process_args(args: argparse.Namespace) -> None:
    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError("Data type should be one of 'fp16', 'fp32'")

    if not (args.model.startswith("meta-llama/Llama-2") or args.model.startswith("meta-llama/Meta-Llama-3")):
        raise argparse.ArgumentTypeError("Only Llama models are supported by export_hn_pruned_llama.py")


def load_hn_hard_vectors(model: torch.nn.Module, ckpt_path: str) -> list[torch.Tensor]:
    reg = collect_info_reg_llama(model, p=0.0, lam=0.0)
    hn_state = torch.load(ckpt_path, map_location="cpu")
    if any(key.startswith("module.") for key in hn_state.keys()):
        hn_state = {key.replace("module.", "", 1): value for key, value in hn_state.items()}

    hn = hypernetwork(t_structures=reg.structures)
    hn.load_state_dict(hn_state)
    hn.to(config.device)
    hn.eval()
    with torch.no_grad():
        vectors = hn.hard_output()
    return [vector.detach().to(device="cpu") for vector in vectors]


def _mask_to_index(mask: torch.Tensor, expected_size: int, name: str) -> list[int]:
    mask = mask.detach().cpu().reshape(-1)
    if mask.numel() != expected_size:
        raise ValueError(f"{name} size mismatch: expected {expected_size}, got {mask.numel()}")
    index = torch.nonzero(mask > 0, as_tuple=False).flatten().tolist()
    if not index:
        raise ValueError(f"{name} produced an empty index after hard pruning")
    return index


def build_layer_configs(model: torch.nn.Module, vectors: list[torch.Tensor]) -> list[dict]:
    layer_cfgs: list[dict] = []
    vector_index = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attn_in = _mask_to_index(
            vectors[vector_index],
            layer.self_attn.q_proj.in_features,
            f"layer {layer_idx} attention input gate",
        )
        attn_out = _mask_to_index(
            vectors[vector_index + 1],
            layer.self_attn.o_proj.out_features,
            f"layer {layer_idx} attention output gate",
        )
        mlp_in = _mask_to_index(
            vectors[vector_index + 2],
            layer.mlp.gate_proj.in_features,
            f"layer {layer_idx} mlp input gate",
        )
        mlp_out = _mask_to_index(
            vectors[vector_index + 3],
            layer.mlp.down_proj.out_features,
            f"layer {layer_idx} mlp output gate",
        )
        mlp_mid = list(range(layer.mlp.gate_proj.out_features))

        layer_cfgs.append(
            {
                "input_size": int(layer.input_layernorm.mean_dim),
                "attn_output_size": int(layer.attn_shortcut_Q.shape[1]),
                "mlp_input_size": int(layer.post_attention_layernorm.mean_dim),
                "mlp_output_size": int(layer.mlp_shortcut_Q.shape[1]),
                "attn_select_index": attn_in,
                "attn_copy_index": attn_out,
                "mlp_select_index": mlp_in,
                "mlp_intermediate_index": mlp_mid,
                "mlp_copy_index": mlp_out,
            }
        )
        vector_index += 4

    return layer_cfgs


def build_pruned_config(model: torch.nn.Module, layer_cfgs: list[dict], args: argparse.Namespace):
    pruned_config = copy.deepcopy(model.config)
    pruned_config.pruned_export_kind = "llama_hn_physical"
    pruned_config.pruned_embedding_dim = int(model.model.embed_tokens.weight.shape[1])
    pruned_config.pruned_final_hidden_size = int(model.model.norm.mean_dim)
    pruned_config.pruned_layer_configs = layer_cfgs
    pruned_config.architectures = ["PrunedLlamaForCausalLM"]
    pruned_config.step1_sparsity = float(args.sparsity)
    pruned_config.hn_checkpoint_path = str(args.hn_ckpt_path)
    return pruned_config


def _copy_linear_weight(dst: torch.nn.Linear, src: torch.nn.Linear, row_index: list[int] | None, col_index: list[int] | None) -> None:
    weight = src.weight.detach().cpu()
    if row_index is not None:
        weight = weight[row_index, :]
    if col_index is not None:
        weight = weight[:, col_index]
    dst.weight.data.copy_(weight.to(dtype=dst.weight.dtype))
    if dst.bias is not None and src.bias is not None:
        bias = src.bias.detach().cpu()
        if row_index is not None:
            bias = bias[row_index]
        dst.bias.data.copy_(bias.to(dtype=dst.bias.dtype))


def copy_pruned_weights(pruned_model: PrunedLlamaForCausalLM, source_model: torch.nn.Module, layer_cfgs: list[dict]) -> None:
    pruned_model.model.embed_tokens.weight.data.copy_(
        source_model.model.embed_tokens.weight.detach().cpu().to(dtype=pruned_model.model.embed_tokens.weight.dtype)
    )
    pruned_model.lm_head.weight.data.copy_(
        source_model.lm_head.weight.detach().cpu().to(dtype=pruned_model.lm_head.weight.dtype)
    )

    for layer_idx, (pruned_layer, source_layer, layer_cfg) in enumerate(
        zip(pruned_model.model.layers, source_model.model.layers, layer_cfgs, strict=True)
    ):
        _copy_linear_weight(pruned_layer.self_attn.q_proj, source_layer.self_attn.q_proj, None, layer_cfg["attn_select_index"])
        _copy_linear_weight(pruned_layer.self_attn.k_proj, source_layer.self_attn.k_proj, None, layer_cfg["attn_select_index"])
        _copy_linear_weight(pruned_layer.self_attn.v_proj, source_layer.self_attn.v_proj, None, layer_cfg["attn_select_index"])
        _copy_linear_weight(pruned_layer.self_attn.o_proj, source_layer.self_attn.o_proj, layer_cfg["attn_copy_index"], None)

        _copy_linear_weight(
            pruned_layer.mlp.gate_proj,
            source_layer.mlp.gate_proj,
            layer_cfg["mlp_intermediate_index"],
            layer_cfg["mlp_select_index"],
        )
        _copy_linear_weight(
            pruned_layer.mlp.up_proj,
            source_layer.mlp.up_proj,
            layer_cfg["mlp_intermediate_index"],
            layer_cfg["mlp_select_index"],
        )
        _copy_linear_weight(
            pruned_layer.mlp.down_proj,
            source_layer.mlp.down_proj,
            layer_cfg["mlp_copy_index"],
            layer_cfg["mlp_intermediate_index"],
        )

        pruned_layer.attn_shortcut_Q.data.copy_(
            source_layer.attn_shortcut_Q.detach().cpu().to(dtype=pruned_layer.attn_shortcut_Q.dtype)
        )
        pruned_layer.mlp_shortcut_Q.data.copy_(
            source_layer.mlp_shortcut_Q.detach().cpu().to(dtype=pruned_layer.mlp_shortcut_Q.dtype)
        )

        if hasattr(pruned_layer.self_attn, "layer_idx"):
            pruned_layer.self_attn.layer_idx = layer_idx


def save_metadata(out_dir: pathlib.Path, args: argparse.Namespace, layer_cfgs: list[dict], param_count: int) -> None:
    metadata = {
        "base_model_id": args.model,
        "step1_sparsity": float(args.sparsity),
        "hn_checkpoint_path": args.hn_ckpt_path,
        "final_parameter_count": int(param_count),
        "layers": [
            {
                "layer": idx,
                "attn_input_kept": len(layer_cfg["attn_select_index"]),
                "attn_output_kept": len(layer_cfg["attn_copy_index"]),
                "mlp_input_kept": len(layer_cfg["mlp_select_index"]),
                "mlp_intermediate_kept": len(layer_cfg["mlp_intermediate_index"]),
                "mlp_output_kept": len(layer_cfg["mlp_copy_index"]),
            }
            for idx, layer_cfg in enumerate(layer_cfgs)
        ],
    }
    (out_dir / "pruned_export_metadata.json").write_text(json.dumps(metadata, indent=2))


def main(args: argparse.Namespace) -> None:
    logging.info("Loading step1 sliced model from %s", args.sliced_model_path)
    model_adapter, tokenizer = hf_utils.load_sliced_model(
        args.model,
        args.sliced_model_path,
        token=args.hf_token,
        sparsity=args.sparsity,
        round_interval=args.round_interval,
    )
    source_model = model_adapter.model
    source_model.eval()

    logging.info("Loading hypernetwork checkpoint from %s", args.hn_ckpt_path)
    vectors = load_hn_hard_vectors(source_model, args.hn_ckpt_path)

    logging.info("Building per-layer pruning indices")
    layer_cfgs = build_layer_configs(source_model, vectors)
    pruned_config = build_pruned_config(source_model, layer_cfgs, args)

    logging.info("Constructing pruned llama model")
    pruned_model = PrunedLlamaForCausalLM(pruned_config)
    copy_pruned_weights(pruned_model, source_model, layer_cfgs)
    pruned_model.eval()
    pruned_model.register_for_auto_class("AutoModelForCausalLM")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving pruned model to %s", out_dir)
    pruned_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    param_count = sum(parameter.numel() for parameter in pruned_model.parameters())
    save_metadata(out_dir, args, layer_cfgs, param_count)
    logging.info("Saved final pruned model with %d parameters", param_count)


if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(levelname)s:%(message)s", level=logging.INFO)
    parsed_args = parse_args()
    process_args(parsed_args)
    main(parsed_args)
