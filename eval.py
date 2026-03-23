import argparse
import json
import logging
import os
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slicegpt import data_utils, gpu_utils, hf_utils
from slicegpt.config import config
from src.disp.pruning.hypernetwork import hypernetwork
from src.disp.pruning.pruning_helper import collect_info_reg_llama, collect_info_reg_phi2, help_functions_hn


DEFAULT_LM_EVAL_TASKS = "hellaswag,arc_easy,arc_challenge,piqa,winogrande"


def _is_trust_remote_code_model_dir(model_path: str | None) -> bool:
    if model_path is None:
        return False
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False

    with open(config_path) as config_file:
        data = json.load(config_file)

    auto_map = data.get("auto_map", {})
    return "AutoModelForCausalLM" in auto_map or data.get("pruned_export_kind") == "llama_hn_physical"


def eval_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ppl", "lm_eval"],
        help="Evaluation task to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier used to choose the adapter, e.g. meta-llama/Llama-2-7b-hf.",
    )
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a local model directory.",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        default=None,
        help="Path to a sliced model directory.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="Sparsity of the sliced model. Only used with --sliced-model-path.",
    )
    parser.add_argument(
        "--hn-ckpt-path",
        type=str,
        default=None,
        help="Path to a hypernetwork checkpoint. Only supported with --task ppl.",
    )
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", None))
    parser.add_argument("--round-interval", type=int, default=8)

    parser.add_argument(
        "--cal-dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default=None,
        help="Dataset for PPL evaluation.",
    )
    parser.add_argument("--ppl-eval-seqlen", type=int, default=None)
    parser.add_argument("--ppl-eval-batch-size", type=int, default=None)
    parser.add_argument("--distribute-model", action="store_true")

    parser.add_argument("--lm-eval-tasks", type=str, default=None)
    parser.add_argument("--lm-eval-batch-size", type=int, default=None)
    parser.add_argument("--lm-eval-dtype", type=str, default=None)
    parser.add_argument("--lm-eval-num-processes", type=int, default=None)
    parser.add_argument("--lm-eval-main-process-port", type=int, default=None)
    return parser.parse_args()


def process_eval_args(args: argparse.Namespace) -> None:
    if args.model_path and args.sparsity != 0.0:
        raise argparse.ArgumentTypeError("--sparsity is only supported with --sliced-model-path")

    if args.task == "ppl":
        if args.lm_eval_tasks is not None:
            raise argparse.ArgumentTypeError("--lm-eval-tasks is only supported with --task lm_eval")
        if args.lm_eval_batch_size is not None:
            raise argparse.ArgumentTypeError("--lm-eval-batch-size is only supported with --task lm_eval")
        if args.lm_eval_dtype is not None:
            raise argparse.ArgumentTypeError("--lm-eval-dtype is only supported with --task lm_eval")
        if args.lm_eval_num_processes is not None:
            raise argparse.ArgumentTypeError("--lm-eval-num-processes is only supported with --task lm_eval")
        if args.lm_eval_main_process_port is not None:
            raise argparse.ArgumentTypeError("--lm-eval-main-process-port is only supported with --task lm_eval")

        if args.cal_dataset is None:
            args.cal_dataset = "wikitext2"
        if args.ppl_eval_seqlen is None:
            args.ppl_eval_seqlen = 2048
        if args.ppl_eval_batch_size is None:
            args.ppl_eval_batch_size = 1
    else:
        if args.hn_ckpt_path is not None:
            raise argparse.ArgumentTypeError("--hn-ckpt-path is not supported with --task lm_eval")
        if args.cal_dataset is not None:
            raise argparse.ArgumentTypeError("--cal-dataset is only supported with --task ppl")
        if args.ppl_eval_seqlen is not None:
            raise argparse.ArgumentTypeError("--ppl-eval-seqlen is only supported with --task ppl")
        if args.ppl_eval_batch_size is not None:
            raise argparse.ArgumentTypeError("--ppl-eval-batch-size is only supported with --task ppl")
        if args.distribute_model:
            raise argparse.ArgumentTypeError("--distribute-model is only supported with --task ppl")

        if args.lm_eval_tasks is None:
            args.lm_eval_tasks = DEFAULT_LM_EVAL_TASKS
        if args.lm_eval_batch_size is None:
            args.lm_eval_batch_size = 16
        if args.lm_eval_dtype is None:
            args.lm_eval_dtype = "bfloat16"
        if args.lm_eval_num_processes is None:
            args.lm_eval_num_processes = 1
        if args.lm_eval_main_process_port is None:
            args.lm_eval_main_process_port = 12323

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError("Data type should be one of 'fp16', 'fp32'")


def build_param_reg(model, model_name: str):
    reg_builders = {
        "llama": collect_info_reg_llama,
        "phi2": collect_info_reg_phi2,
    }

    if model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3"):
        return reg_builders["llama"](model, p=0.0, lam=0.0)
    if model_name == "microsoft/phi-2":
        return reg_builders["phi2"](model, p=0.0, lam=0.0)

    raise ValueError(
        f"--hn-ckpt-path is not supported for model {model_name}. "
        "Supported models are meta-llama/Llama-2*, meta-llama/Meta-Llama-3*, and microsoft/phi-2."
    )


def apply_hn_gates_from_ckpt(model: torch.nn.Module, model_name: str, ckpt_path: str) -> None:
    reg = build_param_reg(model, model_name)
    hn_helper = help_functions_hn(reg.structures)

    hn_state = torch.load(ckpt_path, map_location="cpu")
    if any(key.startswith("module.") for key in hn_state.keys()):
        hn_state = {key.replace("module.", "", 1): value for key, value in hn_state.items()}

    hn = hypernetwork(t_structures=reg.structures)
    hn.load_state_dict(hn_state)
    hn.to(config.device)
    hn.eval()
    with torch.no_grad():
        vectors = hn.hard_output()

    hn_helper.set_gate_vectors(model, vectors)
    hn_helper.set_gate_status(model, use_gate=True)


def load_model_for_eval(args: argparse.Namespace):
    if args.model_path and _is_trust_remote_code_model_dir(args.model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            token=args.hf_token,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            token=args.hf_token,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=config.dtype,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer, None

    if args.sliced_model_path:
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            token=args.hf_token,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
        )
        return model_adapter.model, tokenizer, model_adapter

    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
        args.model,
        model_path=args.model_path,
        token=args.hf_token,
        dtype=config.dtype,
    )
    return model_adapter.model, tokenizer, model_adapter


def get_pretrained_path(args: argparse.Namespace) -> str:
    return args.sliced_model_path if args.sliced_model_path else args.model_path


def run_ppl(args: argparse.Namespace) -> None:
    logging.info("Running perplexity evaluation.")
    logging.info("PyTorch device: %s", config.device)
    logging.info("Number of available cuda devices: %s", torch.cuda.device_count())

    model, tokenizer, model_adapter = load_model_for_eval(args)

    if args.distribute_model:
        if model_adapter is None:
            raise ValueError("--distribute-model is not supported when loading a trust_remote_code export directory")
        gpu_utils.distribute_model(model_adapter)
    else:
        model.to(config.device)

    if args.hn_ckpt_path:
        if model_adapter is None:
            raise ValueError("--hn-ckpt-path is not supported when evaluating a trust_remote_code export directory")
        apply_hn_gates_from_ckpt(model, args.model, args.hn_ckpt_path)

    dataset = data_utils.get_dataset(args.cal_dataset)
    test_loader = data_utils.prepare_test_dataloader(
        dataset["test"],
        tokenizer,
        seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
    )
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info("Perplexity: %.4f", dataset_ppl)


def run_lm_eval(args: argparse.Namespace) -> None:
    pretrained_path = get_pretrained_path(args)
    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--main_process_port",
        str(args.lm_eval_main_process_port),
        "--num_processes",
        str(args.lm_eval_num_processes),
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={pretrained_path},dtype={args.lm_eval_dtype},trust_remote_code=true",
        "--tasks",
        args.lm_eval_tasks,
        "--batch_size",
        str(args.lm_eval_batch_size),
    ]

    logging.info("Running lm_eval with command: %s", " ".join(command))
    logging.info(
        "lm_eval env overrides: NCCL_P2P_DISABLE=%s NCCL_IB_DISABLE=%s",
        env["NCCL_P2P_DISABLE"],
        env["NCCL_IB_DISABLE"],
    )
    subprocess.run(command, check=True, env=env)


def eval_main(args: argparse.Namespace) -> None:
    if args.task == "ppl":
        run_ppl(args)
    elif args.task == "lm_eval":
        run_lm_eval(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )
    arguments = eval_arg_parser()
    process_eval_args(arguments)
    eval_main(arguments)
