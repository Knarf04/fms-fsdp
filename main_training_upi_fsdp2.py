"""UPI-only fine-tuning with FSDP2.

All base model parameters are frozen.  Only the per-layer `upi_scale_raw`
nn.Parameters (created when experiments["upi"] = {"target_multiplier": M}) are
updated.

Usage
-----
experiments["upi"] MUST be {"target_multiplier": M} (trainable mode).
Passing a file path / {"mask_path": ...} creates a non-trainable buffer and
there is nothing to train, so the script will raise an error early.

Extra cfg fields (beyond the standard train_config):
  upi_load_path: (optional) path to a previously saved upi_state.pth so that
                 a prior upi training run can be resumed from its final upi
                 state while re-loading the base weights from ckpt_load_path.
                 When ckpt_save_path already contains upi checkpoints the
                 normal checkpointer resume logic takes precedence.
"""

import math
import os
from pathlib import Path

import fire
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from torch import distributed as dist

# FSDP2
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed._composable.fsdp import fully_shard, register_fsdp_forward_method
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_profiler,
    setup,
    setup_environ_flags,
)

# FSDP2
from fms_fsdp.fsdp2.train_utils import (
    get_policies,
    train,
)
from fms_fsdp.fsdp2.checkpointing_utils import Checkpointer_FSDP2

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main(**kwargs):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    cfg = config.train_config()
    update_config(cfg, **kwargs)

    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        Path.home(), ".triton", "cache", str(local_rank)
    )

    block = Block
    (
        mixed_precision_policy,
        mixed_precision_buffer,
        apply_selective_ac,
    ) = get_policies(cfg, rank, block)

    def get_1D_world_mesh(world_size: int) -> DeviceMesh:
        return init_device_mesh("cuda", (world_size,))

    def get_2D_world_mesh(world_size: int) -> DeviceMesh:
        num_gpu_per_node = torch.cuda.device_count()
        assert world_size % num_gpu_per_node == 0
        return init_device_mesh(
            "cuda",
            (world_size // num_gpu_per_node, num_gpu_per_node),
            mesh_dim_names=("inter_node", "intra_node"),
        )

    requires_2d_mesh = (cfg.sharding_strategy == "hsdp") or (
        cfg.cp and not cfg.cp_over_world
    )
    if requires_2d_mesh:
        mesh = get_2D_world_mesh(world_size)
        cp_mesh = mesh["intra_node"] if cfg.cp else None
    else:
        mesh = get_1D_world_mesh(world_size)
        cp_mesh = mesh if cfg.cp else None
    fsdp_mesh = mesh

    if cfg.cp:
        cp_degree = world_size if cfg.cp_over_world else torch.cuda.device_count()
    else:
        cp_degree = 1

    dp_degree = world_size // cp_degree

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if cfg.upi_target_multiplier <= 1.0:
        raise ValueError(
            f"upi_target_multiplier must be > 1.0, got {cfg.upi_target_multiplier}. "
            "Pass --upi_target_multiplier=<M> (e.g. 8.0) on the command line."
        )

    config_data = get_model_config(cfg.model_variant)
    # Inject trainable upi config, overriding any existing experiments["upi"] value.
    if isinstance(config_data, dict):
        experiments = config_data.setdefault("experiments", {})
    else:
        # MambaConfig dataclass (LLaMA-style variants not relevant here, but be safe)
        experiments = getattr(config_data, "experiments", {})
    experiments["upi"] = {"target_multiplier": cfg.upi_target_multiplier}

    mamba_config = MambaConfig(**config_data)
    if rank == 0:
        print(mamba_config)

    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = MambaLMHeadModel(
                mamba_config,
                cp_mesh=cp_mesh if cfg.cp else None,
                cp_mamba_impl=cfg.cp_mamba_impl if cfg.cp else None,
                cp_attn_impl=cfg.cp_attn_impl if cfg.cp else None,
            )
    else:
        model = MambaLMHeadModel(
            mamba_config,
            cp_mesh=cp_mesh if cfg.cp else None,
            cp_mamba_impl=cfg.cp_mamba_impl if cfg.cp else None,
            cp_attn_impl=cfg.cp_attn_impl if cfg.cp else None,
        )

    # ------------------------------------------------------------------
    # Verify that the model was built with trainable upi params.
    # (experiments["upi"] must be {"target_multiplier": M})
    # ------------------------------------------------------------------
    upi_param_names = [n for n, _ in model.named_parameters() if "upi_scale_raw" in n]
    if not upi_param_names:
        raise ValueError(
            "No upi_scale_raw parameters found in the model. "
            "Set experiments['upi'] = {'target_multiplier': M} in the model "
            "config to enable trainable UPI mode."
        )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        n_upi = sum(p.numel() for n, p in model.named_parameters() if "upi_scale_raw" in n)
        print(
            f"--> model has {total_params / 1e6:.2f}M total params; "
            f"{n_upi} are trainable upi_scale_raw params ({len(upi_param_names)} tensors), "
            f"all others will be frozen."
        )

    # ------------------------------------------------------------------
    # Freeze everything; unfreeze only upi_scale_raw.
    # Do this BEFORE fully_shard so FSDP2 sees the correct requires_grad
    # flags and can skip grad reduce-scatter for frozen parameters.
    # ------------------------------------------------------------------
    for name, param in model.named_parameters():
        param.requires_grad_("upi_scale_raw" in name)

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size, dp_degree)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    if cfg.low_cpu_fsdp:
        model.to_empty(device=torch.cuda.current_device())

    # ------------------------------------------------------------------
    # Activation checkpointing
    # ------------------------------------------------------------------
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print("--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=cfg.selective_checkpointing)

    # ------------------------------------------------------------------
    # FSDP2 wrapping  (same policy as the base training script)
    # ------------------------------------------------------------------
    def lambda_fn(name, module):
        if isinstance(module, CheckpointWrapper):
            return True
        if isinstance(module, (Block, nn.Embedding)) or module is model.lm_head:
            if isinstance(module, Block) and "_checkpoint_wrapped_module" in name:
                return False
            return True
        return False

    for name, module in model.named_modules():
        if module is model:
            continue
        if lambda_fn(name, module):
            fully_shard(
                module,
                mesh=fsdp_mesh,
                mp_policy=mixed_precision_policy,
                reshard_after_forward=True,
            )
    fully_shard(
        model.backbone,
        mesh=fsdp_mesh,
        mp_policy=mixed_precision_policy,
        reshard_after_forward=True,
    )
    fully_shard(
        model,
        mesh=fsdp_mesh,
        mp_policy=mixed_precision_policy,
        reshard_after_forward=True,
    )

    if rank == 0:
        print(model)

    register_fsdp_forward_method(model, "generate")

    if cfg.use_torch_compile:
        if rank == 0:
            print("--> enabling torch compile...")
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Optimizer: only upi_scale_raw params, no weight decay.
    # ------------------------------------------------------------------
    upi_params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"--> optimizer tracking {len(upi_params)} upi parameter tensor(s)")

    optimizer = optim.AdamW(
        upi_params,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # ------------------------------------------------------------------
    # Load base model checkpoint (frozen weights).
    # strict=False is essential: the saved weights have no upi_scale_raw
    # (we filter it out in save_single_file), so strict loading would fail.
    # ------------------------------------------------------------------
    checkpointer = Checkpointer_FSDP2(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank,
        mesh=fsdp_mesh if cfg.sharding_strategy == "hsdp" else None,
    )
    model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
        model,
        optimizer,
        None,
        path=(
            os.path.join(cfg.ckpt_load_path, "checkpoints/")
            if not os.path.isfile(cfg.ckpt_load_path)
            else cfg.ckpt_load_path
        ),
        strict=False,
    )

    if not is_resuming:
        start_step = 0
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]

    # ------------------------------------------------------------------
    # Optionally restore upi state from a separate upi_state.pth.
    # This is useful when upi_load_path points to a previously saved
    # upi_state.pth (e.g. from save_single_file) and you want to warm-start
    # the upi scales rather than begin from the zero-initialised defaults.
    # Skipped when is_resuming=True since the training checkpoint already
    # contains the upi optimizer state.
    # ------------------------------------------------------------------
    upi_load_path = cfg.upi_load_path
    if upi_load_path and not is_resuming and os.path.isfile(upi_load_path):
        if rank == 0:
            print(f"--> loading upi state from {upi_load_path}")
        upi_state = torch.load(upi_load_path, map_location="cpu", weights_only=False)
        target_model = model._orig_mod if cfg.use_torch_compile else model
        set_model_state_dict(
            target_model,
            upi_state,
            options=StateDictOptions(strict=False, full_state_dict=True),
        )

    # ------------------------------------------------------------------
    # LR schedule (identical shape to the base training script)
    # ------------------------------------------------------------------
    warmup_interval = min(2000, cfg.num_steps // 20)
    warmup = lambda x: 1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2
    if cfg.training_stage == "annealing":
        schedule = lambda x: min(warmup(x), 1 - x / cfg.num_steps)
    elif cfg.training_stage == "constant":
        schedule = warmup
    else:
        # cosine decay
        schedule = lambda x: min(
            warmup(x),
            0.1
            + 0.5
            * (1 - 0.1)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )

    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    profiler = get_profiler(cfg, rank)

    if rank == 0:
        print(f"Training for {cfg.num_steps} steps (upi-only, base model frozen)")

    train(
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
        cp_degree,
        is_compiled=cfg.use_torch_compile,
        ref_model=None,
    )

    # Final save: base model (unchanged) + upi_state.pth alongside it.
    checkpointer.save_single_file(cfg.num_steps, model, is_compiled=cfg.use_torch_compile)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
