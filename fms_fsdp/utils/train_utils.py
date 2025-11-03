import os
import re
from dataclasses import asdict
from functools import partial
from typing import Dict, List, Optional, Set

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp.policies import *


def parse_freeze_layers(freeze_layer: str) -> Dict[str, List[int]]:
    """
    Parse freeze_layer config string into a dictionary of layer indices.

    Example: "ssm:[0-8,10-17];attn:[9,18,27];mlp:[0-31]"
    Returns: {"ssm": [0,1,...,8,10,...,17], "attn": [9,18,27], "mlp": [0,1,...,31]}
    """
    freeze_layer_dict = {"ssm": [], "attn": [], "mlp": []}
    if not freeze_layer:
        return freeze_layer_dict

    blocks = freeze_layer.split(';')
    for b in blocks:
        key, value = b.split(':')
        chunks = value.strip("[]")
        layers = []
        if len(chunks) > 0:
            chunks = chunks.split(',')
            for c in chunks:
                nums = [int(num) for num in c.split('-')]
                if len(nums) == 1:
                    layers += [nums[0]]
                else:
                    layers += list(range(nums[0], nums[1] + 1))
        freeze_layer_dict[key] = layers

    return freeze_layer_dict


def build_gradient_mask(
    model,
    freeze_layer: str,
    freeze_embedding: bool = True,
    freeze_norm_f: bool = True,
    freeze_lm_head: bool = True,
) -> Set[str]:
    """
    Build a set of parameter names that should have their gradients zeroed.

    This is used with gradient masking approach: all params are trainable during
    forward/backward, but gradients are zeroed for "frozen" params before optimizer.step().

    Args:
        model: The model (can be FSDP-wrapped or not)
        freeze_layer: Config string like "ssm:[0-8];attn:[9];mlp:[0-31]"
        freeze_embedding: Whether to freeze backbone.embedding
        freeze_norm_f: Whether to freeze backbone.norm_f
        freeze_lm_head: Whether to freeze lm_head

    Returns:
        Set of parameter names (FQNs) to mask gradients for
    """
    freeze_layer_dict = parse_freeze_layers(freeze_layer)
    frozen_params: Set[str] = set()

    for name, param in model.named_parameters():
        should_freeze = False

        # Check layer-specific freezing (ssm/attn mixer, mlp)
        # Pattern: backbone.layers.{i}.mixer.* or _fsdp_wrapped_module.backbone.layers.{i}.mixer.*
        layer_match = re.search(r'backbone\.layers\.(\d+)\.', name) or \
                      re.search(r'_orig_mod\.backbone\.layers\.(\d+)\.', name)

        if layer_match:
            layer_idx = int(layer_match.group(1))

            # Check if this is a mixer param (ssm or attn)
            if '.mixer.' in name or '.norm.' in name:
                # For simplicity, we treat both ssm and attn as "mixer" params
                # The freeze_layer config specifies which layers to freeze for each type
                if layer_idx in freeze_layer_dict["ssm"] or layer_idx in freeze_layer_dict["attn"]:
                    should_freeze = True

            # Check if this is an mlp param
            if '.mlp.' in name or '.norm2.' in name:
                if layer_idx in freeze_layer_dict["mlp"]:
                    should_freeze = True

        # Check embedding
        if freeze_embedding and ('backbone.embedding.' in name or 'backbone.embedding.weight' in name):
            should_freeze = True

        # Check norm_f
        if freeze_norm_f and 'backbone.norm_f.' in name:
            should_freeze = True

        # Check lm_head
        if freeze_lm_head and ('lm_head.' in name or name.endswith('lm_head.weight')):
            should_freeze = True

        if should_freeze:
            frozen_params.add(name)

    return frozen_params


def apply_gradient_mask(model, frozen_param_names: Set[str]) -> int:
    """
    Zero out gradients for parameters in the frozen set.

    Call this after loss.backward() but before optimizer.step().

    Args:
        model: The model
        frozen_param_names: Set of parameter names to zero gradients for

    Returns:
        Number of parameters whose gradients were zeroed
    """
    count = 0
    for name, param in model.named_parameters():
        if name in frozen_param_names and param.grad is not None:
            param.grad.zero_()
            count += 1
    return count


def train(
    cfg,
    model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
    cp_degree: int = 1,
):
    if cfg.tracker:
        if cfg.tracker not in ["wandb", "aim"]:
            raise ValueError(f"tracker {cfg.tracker} not supported.")
        tracker_dir = cfg.tracker_dir
        project_name = cfg.tracker_project_name
        run_id = cfg.tracker_run_id

        if cfg.tracker == "wandb":
            try:
                import wandb  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to wandb but wandb is not installed.")
            if rank == 0:
                print("--> wandb is enabled!")
                try:
                    wandb.init(
                        project=project_name,
                        dir=tracker_dir,
                        resume="allow",
                        id=run_id,
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )
                wandb.config = asdict(cfg)

        if cfg.tracker == "aim":
            try:
                from aim import Run  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to aim but aim is not installed.")
            if rank == 0:
                print("--> aim is enabled!")
                run = Run(
                    experiment=project_name,
                    repo=tracker_dir,
                    run_hash=run_id,
                )
                run["hparams"] = asdict(cfg)

    is_exp_out = (len(model.backbone.experiments) != 0)
    model.train()
    ddp_stats = torch.zeros(3).to(local_rank)

    frozen_param_names = None
    if cfg.freeze_layer:
        frozen_param_names = build_gradient_mask(
            model,
            cfg.freeze_layer,
            freeze_embedding=True,
            freeze_norm_f=True,
            freeze_lm_head=True,
        )
    if rank == 0:
        print(f"--> Using gradient masking for {len(frozen_param_names)} parameters")
        for name in frozen_param_names:
            print(name)

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        optimizer.zero_grad()
        if not is_exp_out:
            output = model(input)
        else:
            output, exp_out_collect = model(input)

        output = output.logits if hasattr(output, "logits") else output
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())
        loss = loss + cfg.zl_coeff * torch.logsumexp(output, dim=-1).pow(2).mean()
        loss.backward()

        # Apply gradient mask to zero out gradients for "frozen" params
        if frozen_param_names:
            apply_gradient_mask(model, frozen_param_names)

        # =====================================================================
        # NEW: DEBUG GRADIENT NORMS (Run once on Rank 0)
        # =====================================================================
        if batch_idx == start_step + 1:
            print(f"\n{'='*20} DEBUG: GRADIENT CHECK (Step {batch_idx}) {'='*20}")
            print(f"{'Param Name':<60} | {'Grad Norm (Local Shard)'}")
            print("-" * 85)
            
            total_active_params = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # NOTE: In FSDP, this is the norm of the LOCAL shard, not global.
                    # But it is sufficient to prove the gradient is non-zero.
                    g_norm = param.grad.norm().item()
                    print(f"[Rank {rank}] {name:<60} | {g_norm:.6f}")
                    total_active_params += 1
                elif param.requires_grad:
                    # Warn if a trainable param has no gradient (Broken Chain)
                    print(f"[Rank {rank}] {name:<60} | {'[WARNING: None] (Broken Chain?)'}")
            
            print("-" * 85)
            print(f"Total params with gradients: {total_active_params}")
            print(f"{'='*65}\n")
        # =====================================================================

        ddp_stats[1] += model.clip_grad_norm_(cfg.grad_clip_thresh).item()
        optimizer.step()
        scheduler.step()

        ddp_stats[0] += loss.item()
        ddp_stats[2] += 1

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[0] / ddp_stats[2]
            g_norm = ddp_stats[1] / ddp_stats[2]
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            new_tokens_seen = (
                (batch_idx - start_step)
                * world_size
                * cfg.batch_size
                * cfg.seq_length
                // cp_degree
            )
            if rank == 0:
                total_tokens_seen = tokens_seen + new_tokens_seen
                current_loss = train_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = g_norm.item()
                current_step_time = (time.time() - start) / cfg.report_interval
                overall_step_time = elapsed_time / (batch_idx - start_step)
                current_throughput = int(
                    cfg.batch_size * cfg.seq_length / cp_degree / current_step_time
                )
                overall_throughput = int(
                    cfg.batch_size * cfg.seq_length / cp_degree / overall_step_time
                )
                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )

                print("step:", batch_idx)
                print("loss:", current_loss)
                print("LR:", current_lr)
                print("tokens seen:", total_tokens_seen)
                print("gradient norm:", current_gnorm)
                print("reserved memory:", reserved_mem)
                print("allocated memory:", allocated_mem)
                print("current step time:", current_step_time)
                print("overall step time:", overall_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("overall token per gpu per sec:", overall_throughput)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                print(f"Total tok/step: {world_size * cfg.batch_size * cfg.seq_length}")
                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    elif cfg.tracker == "aim":
                        tracker_fn = run.track
                    tracker_fn(vals_to_track, step=batch_idx)

            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % cfg.checkpoint_interval == 0:
            checkpointer.save(
                batch_idx,
                model,
                optimizer,
                None,
                tokens_seen=tokens_seen + new_tokens_seen,
            )

        # Figure out the current rank, make sure the experiment_out does not overwrite each other
        if is_exp_out and batch_idx % cfg.report_interval == 0:
            for layer_idx in exp_out_collect:
                # for key in exp_out_collect[layer_idx]:
                if "final_states" in exp_out_collect[layer_idx]:
                    experiment_out = exp_out_collect[layer_idx]["final_states"]
                    dist.reduce(experiment_out, dst=0, op=dist.ReduceOp.SUM, async_op=False)
                    if rank == 0:
                        num_nodes = int(os.environ["WORLD_SIZE"]) // torch.cuda.device_count()
                        exp_out_collect[layer_idx]["final_states"] = experiment_out / num_nodes
            if rank == 0:
                os.makedirs(cfg.exp_out_path, exist_ok=True)
                torch.save(exp_out_collect, os.path.join(cfg.exp_out_path, f"experiment_out_step={batch_idx}.pt"))

    return train_loss


def setup():
    dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_mixed_precision_policy(cfg, rank):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
    else:
        mixed_precision_policy = None

    return mixed_precision_policy


def get_policies(cfg, rank, block):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    mixed_precision_policy = get_mixed_precision_policy(cfg, rank)

    # wrapping policy
    wrapping_policy = get_wrapper(block)

    # sharding strategy
    if cfg.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif cfg.sharding_strategy == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    # ac handler
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    # param init function
    if cfg.low_cpu_fsdp:
        param_init_fn = param_init_function
    else:
        param_init_fn = None

    return (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy,
        apply_selective_ac,
        param_init_fn,
    )


def get_profiler(cfg, rank):
    if not cfg.use_profiler:
        return
    if cfg.profiler_rank0_only and rank != 0:
        return
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/gpfs/hshen/profile_traces"),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )
