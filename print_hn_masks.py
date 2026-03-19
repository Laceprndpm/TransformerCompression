#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import pathlib

import torch

from slicegpt import hf_utils, layernorm_fusion
from src.disp.pruning.hypernetwork import hypernetwork
from src.disp.pruning.pruning_helper import collect_info_reg_llama, help_functions_hn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained hypernetwork and print its gate masks.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., microsoft/phi-2).")
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path (optional).",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        default=None,
        help="Path to sliced model directory (recommended if HN was trained on sliced weights).",
    )
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity used for sliced model (if provided).")
    parser.add_argument("--round-interval", type=int, default=1, help="Round interval for slicing config.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Model dtype for loading.",
    )
    parser.add_argument("--hn-ckpt", type=str, required=True, help="Path to hypernetwork checkpoint (.pt).")
    parser.add_argument("--hn-p", type=float, default=0.48, help="Target parameter ratio p (for reg stats).")
    parser.add_argument("--hn-lam", type=float, default=16.0, help="Regularization strength (for reg stats).")
    parser.add_argument("--print-full", action="store_true", help="Print full 0/1 masks (can be very large).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.sliced_model_path:
        model_adapter, _ = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
        )
    else:
        model_adapter, _ = hf_utils.get_model_and_tokenizer(
            args.model,
            args.model_path,
            dtype=dtype,
        )
        # Ensure DISP-style gate layers are in place.
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)

    model = model_adapter.model

    # Build structures for HN
    reg = collect_info_reg_llama(model, p=args.hn_p, lam=args.hn_lam)
    hn_helper = help_functions_hn(reg.structures)

    # Load HN checkpoint (strip DDP prefix if needed)
    ckpt_path = pathlib.Path(args.hn_ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    hn = hypernetwork(t_structures=reg.structures)
    hn.load_state_dict(state)
    hn.eval()

    with torch.no_grad():
        vectors = hn()

    # Print summary and masks
    hn_helper.print_info(vectors)
    for idx, v in enumerate(vectors):
        v_cpu = v.detach().cpu()
        ones = int(v_cpu.sum().item())
        total = v_cpu.numel()
        print(f"[{idx}] ones={ones} / {total} ({ones / max(total, 1):.4f})")
        if args.print_full:
            print(v_cpu.int().tolist())


if __name__ == "__main__":
    main()
