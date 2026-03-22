# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import ipdb
import shutil
import datetime
import sys
import torch
import wandb
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler

# CHANGED: ensure local src/ is importable when running from repo root.
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

from src.disp.utils.distributed_env import DistributedEnv
from src.disp.data import dataloader_creator, load_hf_dataset_wikitext
from src.disp.pruning.hypernetwork import hypernetwork
from src.disp.pruning.pruning_helper import collect_info_reg_llama, collect_info_reg_phi2, help_functions_hn


def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--cal-max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--final-orientation",
        type=str,
        default="random",
        choices=["random", "pca", "pca_random_index"],
        help="Final orientation of the sliced weights.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )
    # DISP-style hypernetwork training (copied from DISP train_hypernetwork.py with minimal changes)
    parser.add_argument('--train-hn', action="store_true", help="Train a DISP hypernetwork after loading model.")
    parser.add_argument('--hn-steps', type=int, default=100000, help="Total training steps for hypernetwork.")
    parser.add_argument('--hn-start-iter', type=int, default=0, help="Start iteration for hypernetwork training.")
    parser.add_argument('--hn-batch-size', type=int, default=1, help="Batch size for hypernetwork training.")
    parser.add_argument('--hn-use-fsdp', action="store_true", help="Use FSDP for model during hn training.")
    parser.add_argument('--hn-num-workers', type=int, default=2, help="Dataloader workers (kept for DISP parity).")
    parser.add_argument(
        '--hn-seed', type=int, default=None, help="Random seed for hn training (default uses start_iter)."
    )
    parser.add_argument('--hn-block-size', type=int, default=2048, help="Sequence length for hn training.")
    parser.add_argument('--hn-p', type=float, default=0.48, help="Target parameter ratio p for hn regularizer.")
    parser.add_argument('--hn-lam', type=float, default=16.0, help="Regularization strength for hn.")
    parser.add_argument('--hn-lr', type=float, default=1e-3, help="Hypernetwork learning rate.")
    parser.add_argument('--hn-min-lr', type=float, default=1e-3, help="Min LR for hn scheduler.")
    parser.add_argument('--hn-use-sch', action="store_true", help="Use cosine scheduler for hn.")
    parser.add_argument('--hn-use-bf16', action="store_true", help="Use bf16 for hn training.")
    parser.add_argument('--hn-out-dir', type=str, default=None, help="Output dir to save hn checkpoint.")
    parser.add_argument(
        '--hn-model-kind',
        type=str,
        choices=["llama", "phi2"],
        default=None,
        help="Model kind for HN regularization/structure collection. Controls which collect_info_reg_* function is used.",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    if args.train_hn and args.hn_model_kind is None:
        raise argparse.ArgumentTypeError(
            "When using --train-hn, you must also pass --hn-model-kind {llama,phi2}. Example: --hn-model-kind phi2"
        )


def build_param_reg(model, args):
    if args.hn_model_kind is None:
        raise ValueError(
            "HN model kind is required to build pruning structures. Pass --hn-model-kind {llama,phi2}. "
            "Example: --hn-model-kind phi2"
        )

    reg_builders = {
        "llama": collect_info_reg_llama,
        "phi2": collect_info_reg_phi2,
    }
    return reg_builders[args.hn_model_kind](model, p=args.hn_p, lam=args.hn_lam)


def slicing_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT experiment.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")
    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')
    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )
    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    hn_ckpt_path = None
    hn_out_dir_used = None

    def apply_hn_gates_from_ckpt(ckpt_path: str) -> None:
        if not ckpt_path or not os.path.exists(ckpt_path):
            logging.warning(f"HN checkpoint not found for gating: {ckpt_path}")
            return

        reg = build_param_reg(model, args)
        hn_helper = help_functions_hn(reg.structures)

        hn_state = torch.load(ckpt_path, map_location="cpu")
        if any(k.startswith("module.") for k in hn_state.keys()):
            hn_state = {k.replace("module.", "", 1): v for k, v in hn_state.items()}

        hn = hypernetwork(t_structures=reg.structures)
        hn.load_state_dict(hn_state)
        hn.to(config.device)
        hn.eval()
        with torch.no_grad():
            vectors = hn()

        hn_helper.set_gate_vectors(model, vectors)
        hn_helper.set_gate_status(model, use_gate=True)

    dataset = data_utils.get_dataset(args.cal_dataset)
    _, test_dataset = dataset["train"], dataset["test"]
    # train_loader = data_utils.prepare_dataloader(
    #     dataset=train_dataset,
    #     tokenizer=tokenizer,
    #     max_seqlen=args.cal_max_seqlen,
    #     batch_size=args.cal_batch_size,
    #     nsamples=args.cal_nsamples,
    #     varied_seqlen=args.varied_seqlen,
    #     seed=args.seed,
    # )
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    def train_hn() -> None:
        # NOTE: This function mirrors DISP-LLM/train_hypernetwork.py flow.
        # CHANGED: Uses SliceGPT dataloader for simplicity and local consistency.
        # NOTE: DistributedEnv expects torchrun/torch.distributed env vars to be set.
        env = DistributedEnv()
        env.print_master(env)

        dist.init_process_group(
            backend="nccl",
            rank=env.global_rank,
            world_size=env.world_size,
            timeout=datetime.timedelta(seconds=3600 * 5),
        )

        data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if args.hn_use_bf16:
            data_type = torch.bfloat16

        nonlocal hn_ckpt_path, hn_out_dir_used
        hn_out_dir = args.hn_out_dir
        if hn_out_dir is None:
            user_name = "user"
            date_time_obj = datetime.datetime.now()
            hn_out_dir = os.path.join("/output/", user_name, date_time_obj.strftime("%Y%m%d-%H%M%S"))
        hn_out_dir_used = hn_out_dir

        if args.hn_seed is None:
            args.hn_seed = args.hn_start_iter

        if env.global_rank == 0:
            os.makedirs(hn_out_dir, exist_ok=True)

        device_id = env.local_rank
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

        # NOTE: keep model in train mode like DISP.
        model.config.use_cache = False
        model.to(device_id)

        ignored_token = tokenizer.bos_token_id
        if ignored_token is None:
            ignored_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        train_dataset_hn = load_hf_dataset_wikitext("train", env.world_size * args.hn_num_workers)
        train_loader = dataloader_creator(
            dataset=train_dataset_hn,
            tokenizer=tokenizer,
            batch_size=args.hn_batch_size,
            block_size=args.hn_block_size,
            num_workers=args.hn_num_workers,
            cycling=False,
            rank=env.global_rank,
            world_size=env.world_size,
            ignored_token=ignored_token,
            shuffle_seed=args.hn_seed,
        )

        # collect pruning info and build hypernetwork
        param_reg = build_param_reg(model, args)
        hn = hypernetwork(t_structures=param_reg.structures)
        hn_helper = help_functions_hn(param_reg.structures)

        hn.to(device_id)
        hn = DDP(hn)

        # Wrap model with FSDP if requested (DISP parity).
        if args.hn_use_fsdp:
            # CHANGED: use current layer class for wrapping instead of PruneLlamaDecoderLayer.
            wrap_cls = {type(model.model.layers[0])}
            my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=wrap_cls)
            if args.hn_use_bf16:
                model.to(data_type)
                model_fsdp = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, use_orig_params=True)
            else:
                model_fsdp = FSDP(
                    model,
                    auto_wrap_policy=my_auto_wrap_policy,
                    use_orig_params=True,
                    mixed_precision=MixedPrecision(
                        param_dtype=data_type,
                        reduce_dtype=data_type,
                        buffer_dtype=data_type,
                    ),
                )
            model_to_train = model_fsdp
        else:
            if args.hn_use_bf16:
                model.to(data_type)
            model_to_train = DDP(model)

        # Explicitly enable gate usage during HN training.
        hn_helper.set_gate_status(model_to_train, use_gate=True)

        # Freeze model params; only train hn.
        for param in model_to_train.parameters():
            param.requires_grad = False
        for param in hn.parameters():
            param.requires_grad = True
        hn.train()

        optimizer = torch.optim.AdamW(hn.parameters(), lr=args.hn_lr, weight_decay=0.05)
        if args.hn_use_sch:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.hn_steps,
                eta_min=args.hn_min_lr,
                last_epoch=args.hn_start_iter - 1,
            )
        else:
            scheduler = None

        # CHANGED: use GradScaler for AMP to match DISP on single process.
        scaler = GradScaler("cuda")
        iter_num = args.hn_start_iter

        while True:
            for batch in train_loader:
                if iter_num >= args.hn_steps:
                    break

                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device_id)
                    targets = batch["labels"].to(device_id)

                with autocast(device_type="cuda", dtype=data_type):
                    vectors = hn()
                    hn_helper.set_gate_vectors(model_to_train, vectors)
                    output = model_to_train(input_ids)
                    logits = output.logits if hasattr(output, "logits") else output
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=ignored_token,
                    )

                    if hasattr(hn, "module"):
                        hard_out = hn.module.hard_output()
                    else:
                        hard_out = hn.hard_output()

                    reg_loss = param_reg(hard_out)
                    total_loss = ce_loss + reg_loss

                if torch.isnan(total_loss):
                    env.print_master("!!! nan loss detected !!!")
                    total_loss.fill_(0)

                env.print_master(f"ce loss: {ce_loss:.4f} reg_loss: {reg_loss:.4f}")

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if iter_num % 50 == 0:
                    env.print_master(
                        f"Iter {iter_num}/{args.hn_steps}, "
                        f"loss={total_loss.item():.4f}, "
                        f"reg={reg_loss.item():.4f}"
                    )

                iter_num += 1
                if iter_num >= args.hn_steps:
                    break
            if iter_num >= args.hn_steps:
                break

        # After training, apply hard gates for a stable evaluation pass.
        if hasattr(hn, "module"):
            hard_out = hn.module.hard_output()
        else:
            hard_out = hn.hard_output()
        hn_helper.set_gate_vectors(model_to_train, hard_out)
        hn_helper.set_gate_status(model_to_train, use_gate=True)

        # Save the hypernetwork checkpoint (match DISP behavior).
        if env.world_size == 1:
            if env.global_rank == 0:
                hn_ckpt_path = os.path.join(hn_out_dir, f"hn-ckpt-final-{args.hn_p:.2f}.pt")
                torch.save(hn.state_dict(), hn_ckpt_path)
        else:
            if hasattr(hn, "module"):
                state_dict_hn = hn.module.state_dict()
            else:
                state_dict_hn = hn.state_dict()
            if env.global_rank == 0:
                hn_ckpt_path = os.path.join(hn_out_dir, f"hn-ckpt-final-{args.hn_p:.2f}.pt")
                torch.save(state_dict_hn, hn_ckpt_path)

    if args.train_hn:
        train_hn()

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.sliced_model_path or args.ppl_only:
        reset_model_device()
        if hn_ckpt_path:
            apply_hn_gates_from_ckpt(hn_ckpt_path)
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Loaded model(gated) perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
    # # original ppl
    # if args.eval_baseline:
    #     reset_model_device()
    #     dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    #     logging.info(f'Original ppl: {dataset_ppl:.4f}')
    #     wandb.log({"original_ppl": dataset_ppl})
    #     model.cpu()
    #     utils.cleanup_memory()

    # # replace modules with compressible equivalents
    # layernorm_fusion.replace_layers(model_adapter)

    # # fuse layernorms and add rotations to skip connections
    # layernorm_fusion.fuse_modules(model_adapter)

    # # don't run this on large and/or distributed models
    # if args.eval_fused_model and not args.distribute_model:
    #     model.to(config.device)

    #     dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    #     logging.info(f'Post-fusion: {dataset_ppl:.4f}')
    #     wandb.log({"post_fusion_ppl": dataset_ppl})

    #     model.cpu()

    #     # run GC and cleanup GPU memory
    #     utils.cleanup_memory()

    # original_param_count = sum(int(p.nelement()) for p in model.parameters())
    # logging.info(f'Original model parameters: {original_param_count:,d}')

    # # compute new embedding dimension given the desired sparsity level
    # new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
    # # round (down) to the nearest multiple of round_interval
    # new_embedding_dimension -= new_embedding_dimension % args.round_interval
    # logging.info(
    #     f"New embedding dimension: {new_embedding_dimension} (sparsity {100*(1 - new_embedding_dimension / model_adapter.hidden_size):.4f} %)"
    # )

    # scheduler = ConstSlicingScheduler(new_embedding_dimension)
    # rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation=args.final_orientation)
    # logging.info("Model structure:\n%s", model)

    # if args.save_dir:
    #     sliced_model_dir = pathlib.Path(args.save_dir)
    #     sliced_model_dir.mkdir(parents=True, exist_ok=True)

    #     sliced_model_name = sliced_model_dir / f'{pathlib.Path(args.model).name}_{args.sparsity}.pt'

    #     # Save the sliced model
    #     torch.save(model.state_dict(), sliced_model_name)

    #     # Save the slicing config
    #     config_path = sliced_model_name.with_suffix('.json')
    #     config_path.write_text(model_adapter.slicing_conf.to_json_string())

    #     # If slicing a local model, also save HF config files in sliced model dir
    #     if args.model_path:
    #         try:
    #             # copy all config files (tokenizer, model and slicing configs)
    #             for file in pathlib.Path(args.model_path).glob("*.json"):
    #                 if 'safetensors' not in str(file):
    #                     shutil.copy(str(file), sliced_model_dir)
    #             # copy all tokenizer models
    #             for file in pathlib.Path(args.model_path).glob("*token*.model"):
    #                 shutil.copy(str(file), sliced_model_dir)
    #             # copy vocab merges if any
    #             for file in pathlib.Path(args.model_path).glob("merges.txt"):
    #                 shutil.copy(str(file), sliced_model_dir)
    #         except OSError as e:
    #             logging.info(f'Failed to copy configs and tokenizer files: {e}')

    #     logging.info(f"Saved sliced model to {args.save_dir}")

    # reset_model_device()
    # dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    # logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    # wandb.log({"sliced_ppl": dataset_ppl})

    # sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    # sliced_fraction = 1.0 - sliced_param_count / original_param_count
    # logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)
