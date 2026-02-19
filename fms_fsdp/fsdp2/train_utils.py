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
import torch.nn.functional as F

from fms_fsdp.policies.ac_handler import apply_fsdp_checkpointing
from fms_fsdp.fsdp2.mixed_precision import (
    fpSixteen, 
    bfSixteen,
)
from fms_fsdp.fsdp2.online_loss import (
    streaming_ce_and_zloss,
    streaming_forward_kl,
    streaming_reverse_kl,
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
    ref_model=None,
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
    ddp_stats = torch.zeros(5).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1

    do_distill = ref_model is not None and getattr(cfg, 'distill_coeff', 0) > 0
    cp_rank = rank % cp_degree if cp_degree > 1 else 0

    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        cp_overlap = getattr(cfg, 'cp_overlap', 0)
        distill_ctx_len = getattr(cfg, 'distill_ctx_len', 4096)
        # Cap warmup at half the chunk — discarding more than half is wasteful
        distill_warmup = min(cp_overlap, distill_ctx_len // 2)
        distill_stride = distill_ctx_len - distill_warmup

        # Split full input into overlapping context-length chunks for distillation
        needs_copy = do_distill and cp_degree > 1 and distill_warmup > 0 and cp_rank == 0
        if do_distill:
            chunk_input = input
            if needs_copy:
                # CP rank 0 has fewer tokens (no left overlap) → 1 fewer chunk.
                # Prepend a copy of the first chunk for batch equalization, and
                # shift real chunks right so that after warmup discard the copy's
                # [:stride] stitches contiguously with the real chunks.
                shift = distill_stride - distill_warmup

                # Match chunk count with non-zero cp ranks
                other_len = chunk_input.size(1) + cp_overlap
                other_rem = (other_len - distill_ctx_len) % distill_stride
                other_padded = other_len + (distill_stride - other_rem if other_rem > 0 else 0)
                n_shifted = (other_padded - distill_ctx_len) // distill_stride  # n_other - 1

                needed = shift + max(0, n_shifted - 1) * distill_stride + distill_ctx_len if n_shifted > 0 else distill_ctx_len
                if chunk_input.size(1) < needed:
                    chunk_input = F.pad(chunk_input, (0, needed - chunk_input.size(1)))

                input_chunks = [chunk_input[:, :distill_ctx_len]]  # copy
                for k in range(n_shifted):
                    chunk_start = shift + k * distill_stride
                    input_chunks.append(chunk_input[:, chunk_start : chunk_start + distill_ctx_len])
            else:
                # Standard chunking (non-zero cp ranks, or no CP)
                remainder = (chunk_input.size(1) - distill_ctx_len) % distill_stride
                if remainder > 0:
                    chunk_input = F.pad(chunk_input, (0, distill_stride - remainder))
                input_chunks = [
                    chunk_input[:, i * distill_stride : i * distill_stride + distill_ctx_len]
                    for i in range((chunk_input.size(1) - distill_ctx_len) // distill_stride + 1)
                ]

        # Strip overlap prefix for training — cp_rank 0 has no left overlap
        if cp_overlap > 0 and cp_rank > 0:
            input = input[:, cp_overlap:]
            label = label[:, cp_overlap:]

        optimizer.zero_grad()
        if not is_exp_out:
            h_s = model.backbone(input)
        else:
            h_s, exp_out_collect = model.backbone(input)

        # ---- streaming CE + zloss ----
        W_s = model.lm_head.weight  # (V,d)
        vchunk = getattr(cfg, "vocab_chunk", 8192)
        ignore_index = -100

        loss, _, _, _ = streaming_ce_and_zloss(
            h_s=h_s,
            W_s=W_s,
            labels=label,
            ignore_index=ignore_index,
            zl_coeff=getattr(cfg, "zl_coeff", 0.0),
            vchunk=vchunk,
        )
        nce_loss = float(loss.item())  # for logging

        # ---- distillation ----
        distill_loss = None
        if do_distill:
            batch_size = input.size(0)

            ref_input = torch.cat(input_chunks, dim=0)

            with torch.no_grad():
                h_t = ref_model.backbone(ref_input)
                n_chunks = len(input_chunks)
                h_t = h_t.view(n_chunks, batch_size, distill_ctx_len, -1)

                if needs_copy:
                    h_t[0, :, distill_warmup:, :] = h_t[0, :, :distill_stride, :]

                h_t = h_t[:, :, distill_warmup:, :]  # (n_chunks, B, stride, d)

                # stitch to (B, T, d)
                h_t = h_t.permute(1, 0, 2, 3).contiguous().view(batch_size, -1, h_t.size(-1))
                h_t = h_t[:, :h_s.size(1), :]
                h_t_stitched = h_t

            # forward KL(pt||ps)
            W_t = ref_model.lm_head.weight

            # mask: only distill where labels are valid (recommended)
            mask_bt = label.ne(ignore_index)
            # ensure mask matches h_s length
            mask_bt = mask_bt[:, :h_s.size(1)]

            temperature = getattr(cfg, "distill_temperature", 1.0)
            kl_type     = getattr(cfg, "distill_kl_type", "forward")  # "forward" or "reverse"

            kl_fn = streaming_reverse_kl if kl_type == "reverse" else streaming_forward_kl
            distill_loss = kl_fn(
                h_t=h_t_stitched,
                W_t=W_t,
                h_s=h_s,
                W_s=W_s,
                mask_bt=mask_bt,
                temperature=temperature,
                vchunk=vchunk,
            )

            loss = loss + getattr(cfg, "distill_coeff", 0.0) * distill_loss

        loss.backward()

        ddp_stats[1] += torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_thresh).full_tensor().item()
        optimizer.step()
        scheduler.step()

        ddp_stats[0] += loss.detach()
        ddp_stats[2] += 1
        ddp_stats[3] += nce_loss
        if do_distill:
            ddp_stats[4] += distill_loss.detach()

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
                current_nce = (ddp_stats[3] / ddp_stats[2]).item()
                print("ce_loss:", current_nce)
                if do_distill:
                    current_distill = (ddp_stats[4] / ddp_stats[2]).item()
                    print("distill_loss:", current_distill)
                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "nce_loss": current_nce,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    if do_distill:
                        vals_to_track["distill_loss"] = current_distill
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
