import argparse
import logging
import os
import pathlib
import shutil

import torch

from slicegpt import hf_utils
from slicegpt.config import config
from src.disp.pruning.hypernetwork import hypernetwork
from src.disp.pruning.pruning_helper import collect_info_reg_llama, collect_info_reg_phi2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model identifier, e.g. meta-llama/Llama-2-7b-hf.")
    parser.add_argument("--sliced-model-path", type=str, required=True, help="Path to the step1 sliced model directory.")
    parser.add_argument("--hn-ckpt-path", type=str, required=True, help="Path to the hypernetwork checkpoint.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for the final sliced model.")
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


def build_param_reg(model, model_name: str):
    if model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3"):
        return collect_info_reg_llama(model, p=0.0, lam=0.0)
    if model_name == "microsoft/phi-2":
        return collect_info_reg_phi2(model, p=0.0, lam=0.0)
    raise ValueError(
        f"Unsupported model {model_name}. Only meta-llama/Llama-2*, meta-llama/Meta-Llama-3*, and microsoft/phi-2 are supported."
    )


def load_hn_hard_vectors(model: torch.nn.Module, model_name: str, ckpt_path: str) -> list[torch.Tensor]:
    reg = build_param_reg(model, model_name)
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


def _mask_to_bool(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    return mask.to(dtype=torch.bool, device=device)


def _zero_linear_input_columns(linear: torch.nn.Linear, keep_mask: torch.Tensor) -> None:
    drop_mask = ~_mask_to_bool(keep_mask, linear.weight.device)
    linear.weight.data[:, drop_mask] = 0


def _zero_linear_output_rows(linear: torch.nn.Linear, keep_mask: torch.Tensor) -> None:
    drop_mask = ~_mask_to_bool(keep_mask, linear.weight.device)
    linear.weight.data[drop_mask, :] = 0
    if linear.bias is not None:
        linear.bias.data[drop_mask] = 0


def _zero_linear_input_and_output(linear: torch.nn.Linear, input_mask: torch.Tensor, output_mask: torch.Tensor) -> None:
    _zero_linear_input_columns(linear, input_mask)
    _zero_linear_output_rows(linear, output_mask)


def bake_llama_hn_into_weights(model: torch.nn.Module, vectors: list[torch.Tensor]) -> None:
    index = 0
    for layer in model.model.layers:
        attn_in = vectors[index]
        attn_out = vectors[index + 1]
        mlp_in = vectors[index + 2]
        mlp_out = vectors[index + 3]

        _zero_linear_input_columns(layer.self_attn.q_proj, attn_in)
        _zero_linear_input_columns(layer.self_attn.k_proj, attn_in)
        _zero_linear_input_columns(layer.self_attn.v_proj, attn_in)
        _zero_linear_output_rows(layer.self_attn.o_proj, attn_out)

        _zero_linear_input_columns(layer.mlp.gate_proj, mlp_in)
        _zero_linear_input_columns(layer.mlp.up_proj, mlp_in)
        _zero_linear_output_rows(layer.mlp.down_proj, mlp_out)

        layer.use_gate = False
        index += 4


def bake_phi2_hn_into_weights(model: torch.nn.Module, vectors: list[torch.Tensor]) -> None:
    index = 0
    for layer in model.model.layers:
        attn_in = vectors[index]
        attn_out = vectors[index + 1]
        mlp_in = vectors[index + 2]
        mlp_mid = vectors[index + 3]
        mlp_out = vectors[index + 4]

        _zero_linear_input_columns(layer.self_attn.q_proj, attn_in)
        _zero_linear_input_columns(layer.self_attn.k_proj, attn_in)
        _zero_linear_input_columns(layer.self_attn.v_proj, attn_in)
        _zero_linear_output_rows(layer.self_attn.dense, attn_out)

        _zero_linear_input_and_output(layer.mlp.fc1, mlp_in, mlp_mid)
        _zero_linear_input_columns(layer.mlp.fc2, mlp_mid)
        _zero_linear_output_rows(layer.mlp.fc2, mlp_out)

        layer.use_gate = False
        index += 5


def bake_hn_into_weights(model: torch.nn.Module, model_name: str, vectors: list[torch.Tensor]) -> None:
    if model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3"):
        bake_llama_hn_into_weights(model, vectors)
    elif model_name == "microsoft/phi-2":
        bake_phi2_hn_into_weights(model, vectors)
    else:
        raise ValueError(f"Unsupported model {model_name}")

    for module in model.modules():
        if hasattr(module, "use_gate"):
            module.use_gate = False


def copy_metadata_files(src_dir: pathlib.Path, dst_dir: pathlib.Path) -> None:
    for file in src_dir.glob("*.json"):
        if "safetensors" not in str(file):
            shutil.copy(str(file), dst_dir)
    for file in src_dir.glob("*token*.model"):
        shutil.copy(str(file), dst_dir)
    for file in src_dir.glob("merges.txt"):
        shutil.copy(str(file), dst_dir)


def save_final_sliced_model(model: torch.nn.Module, src_dir: pathlib.Path, out_dir: pathlib.Path, model_name: str, sparsity: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sliced_model_name = out_dir / f"{pathlib.Path(model_name).name}_{sparsity}.pt"
    torch.save(model.state_dict(), sliced_model_name)

    config_name = f"{pathlib.Path(model_name).name}_{sparsity}.json"
    src_config = src_dir / config_name
    if src_config.exists():
        shutil.copy(str(src_config), out_dir / config_name)

    copy_metadata_files(src_dir, out_dir)


def main(args: argparse.Namespace) -> None:
    logging.info("Loading sliced model from %s", args.sliced_model_path)
    model_adapter, _ = hf_utils.load_sliced_model(
        args.model,
        args.sliced_model_path,
        token=args.hf_token,
        sparsity=args.sparsity,
        round_interval=args.round_interval,
    )
    model = model_adapter.model
    model.to(config.device)
    model.eval()

    logging.info("Loading hypernetwork checkpoint from %s", args.hn_ckpt_path)
    vectors = load_hn_hard_vectors(model, args.model, args.hn_ckpt_path)

    logging.info("Baking hard gates into model weights")
    bake_hn_into_weights(model, args.model, vectors)
    model.cpu()

    out_dir = pathlib.Path(args.out_dir)
    save_final_sliced_model(
        model=model,
        src_dir=pathlib.Path(args.sliced_model_path),
        out_dir=out_dir,
        model_name=args.model,
        sparsity=args.sparsity,
    )
    logging.info("Saved final sliced model to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )
    parsed_args = parse_args()
    process_args(parsed_args)
    main(parsed_args)
