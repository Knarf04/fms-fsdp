import os
from dataclasses import asdict
from functools import partial

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

from fms_fsdp.policies.ac_handler import apply_fsdp_checkpointing
from fms_fsdp.experiments.param_freeze_utils import *
from fms_fsdp.fsdp2.mixed_precision import (
    fpSixteen, 
    bfSixteen,
)

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
    is_compiled: bool = False,
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

    frozen_param_names = set()
    if cfg.freeze_layer:
        frozen_param_names = build_gradient_mask(
            model,
            cfg.freeze_layer,
            freeze_embedding=True,
            freeze_norm_f=True,
            freeze_lm_head=True,
        )
    else:
        frozen_param_names = build_gradient_mask(
            model,
            "",
            freeze_embedding=False,
            freeze_norm_f=False,
            freeze_lm_head=False,
        )

    if rank == 0:
        print(f"--> Using gradient masking for {len(frozen_param_names)} parameters")
    #     for name in frozen_param_names:
    #         print(name)

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
        # DEBUG GRADIENT NORMS
        # =====================================================================
        # if batch_idx == start_step + 1:
        #     print(f"\n{'='*20} DEBUG: GRADIENT CHECK (Step {batch_idx}) {'='*20}")
        #     print(f"{'Param Name':<60} | {'Grad Norm (Local Shard)'}")
        #     print("-" * 85)
            
        #     total_active_params = 0
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             # NOTE: In FSDP, this is the norm of the LOCAL shard, not global.
        #             # But it is sufficient to prove the gradient is non-zero.
        #             g_norm = param.grad.norm().item()
        #             print(f"[Rank {rank}] {name:<60} | {g_norm:.6f}")
        #             total_active_params += 1
        #         elif param.requires_grad:
        #             # Warn if a trainable param has no gradient (Broken Chain)
        #             print(f"[Rank {rank}] {name:<60} | {'[WARNING: None] (Broken Chain?)'}")
            
        #     print("-" * 85)
        #     print(f"Total params with gradients: {total_active_params}")
        #     print(f"{'='*65}\n")
        # =====================================================================
        
        ddp_stats[1] += torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_thresh).full_tensor().item()
        optimizer.step()
        scheduler.step()

        ddp_stats[0] += loss.item()
        ddp_stats[2] += 1

        if profiler:
            profiler.step()

        world_size = int(os.environ["WORLD_SIZE"])
        new_tokens_seen = (
            (batch_idx - start_step)
            * world_size
            * cfg.batch_size
            * cfg.seq_length
            // cp_degree
        )
            
        if batch_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[0] / ddp_stats[2]
            g_norm = ddp_stats[1] / ddp_stats[2]
            elapsed_time = time.time() - loop_start

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
            # checkpointer.save(
            #     batch_idx,
            #     model,
            #     optimizer,
            #     None,
            #     tokens_seen=tokens_seen + new_tokens_seen,
            # )
            checkpointer.save_single_file(
                batch_idx,
                model,
                is_compiled=is_compiled,
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
            buffer_dtype = torch.bfloat16
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            buffer_dtype = torch.float16
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
        def cast_buffers(model):
            if rank == 0:
                print(f"--> casting buffers to {buffer_dtype} for mixed precision parity with FSDP1")
            for module in model.modules():
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    setattr(module, buffer_name, buffer.to(buffer_dtype))
            return model
        mixed_precision_buffer = cast_buffers
    else:
        mixed_precision_policy = None
        mixed_precision_buffer = lambda model: model

    return mixed_precision_policy, mixed_precision_buffer


def get_policies(cfg, rank, block):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    mixed_precision_policy, mixed_precision_buffer = get_mixed_precision_policy(cfg, rank)

    # ac handler
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    return (
        mixed_precision_policy,
        mixed_precision_buffer,
        apply_selective_ac,
    )
