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
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_profiler,
    setup,
    setup_environ_flags,
)

from fms_fsdp.experiments.lr_group_utils import *

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
    
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        Path.home(), ".triton", "cache", str(local_rank)
    )

    # get policy.
    block = Block
    (        
        mixed_precision_policy,
        mixed_precision_buffer,
        apply_selective_ac,
    ) = get_policies(cfg, rank, block)

    # Meshes for FSDP and CP. NOTE: @goon - Getting hangs and/or OOMs if I don't explicitly specify
    # the FSDP mesh when using 4+ nodes with HSDP + in-node-CP.
    def get_1D_world_mesh(world_size: int) -> DeviceMesh:
        mesh = init_device_mesh("cuda", (world_size,))
        return mesh

    def get_2D_world_mesh(world_size: int) -> DeviceMesh:
        num_gpu_per_node = torch.cuda.device_count()
        assert world_size % num_gpu_per_node == 0
        mesh = init_device_mesh(
            "cuda",
            (world_size // num_gpu_per_node, num_gpu_per_node),
            mesh_dim_names=("inter_node", "intra_node"),
        )
        return mesh

    requires_2d_mesh = (cfg.sharding_strategy == "hsdp") or (
        cfg.cp and not cfg.cp_over_world
    )
    if requires_2d_mesh:
        mesh = get_2D_world_mesh(world_size)
        cp_mesh = mesh["intra_node"] if cfg.cp else None
    else:
        mesh = get_1D_world_mesh(world_size)
        cp_mesh = mesh if cfg.cp else None
    fsdp_mesh = mesh                # HSDP

    if cfg.cp:
        cp_degree = world_size if cfg.cp_over_world else torch.cuda.device_count()
    else:
        cp_degree = 1

    dp_degree = world_size // cp_degree

    # get model
    config_data = get_model_config(cfg.model_variant)
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

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
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

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print("--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=cfg.selective_checkpointing)      
        
    # FSDP2
    def lambda_fn(name, module):
        if isinstance(module, CheckpointWrapper):
            return True
        if isinstance(module, (Block, nn.Embedding)) or module is model.lm_head:
            if isinstance(module, Block) and "_checkpoint_wrapped_module" in name:
                return False
            return True
        return False

    # Cast buffers to match FSDP1 behavior (FSDP2 lacks buffer_dtype in MixedPrecisionPolicy)
    # model = mixed_precision_buffer(model)
    for name, module in model.named_modules():
        if module is model:
            continue

        if lambda_fn(name, module):
            fully_shard(
                module,
                mesh=fsdp_mesh,
                mp_policy=mixed_precision_policy,
                reshard_after_forward=True
            )
    # Shard backbone as its own FSDP unit so that model.backbone(input)
    # triggers FSDP's parameter gather (e.g. for backbone.norm_f).
    # Must come after its children (Block, Embedding) are sharded and
    # before the root model is sharded.
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
        reshard_after_forward=True
    )
    if rank == 0:
        print(model)

    # FSDP2: Register generate() for param sharding
    register_fsdp_forward_method(model, "generate")

    # Reference model for distillation (no CP, eval mode, FSDP for memory)
    ref_model = None
    if cfg.ref_model_variant:
        if rank == 0:
            print(f"--> loading reference model: {cfg.ref_model_variant}")
        ref_config_data = get_model_config(cfg.ref_model_variant)
        ref_mamba_config = MambaConfig(**ref_config_data)

        if cfg.low_cpu_fsdp:
            with torch.device("meta"):
                ref_model = MambaLMHeadModel(ref_mamba_config)
        else:
            ref_model = MambaLMHeadModel(ref_mamba_config)

        if cfg.low_cpu_fsdp:
            ref_model.to_empty(device=torch.cuda.current_device())

        # HSDP for ref model: DP over inter-node, FSDP over intra-node, no CP
        ref_mesh = get_2D_world_mesh(world_size) if not requires_2d_mesh else mesh
        # ref_model = mixed_precision_buffer(ref_model)
        for name, module in ref_model.named_modules():
            if module is ref_model:
                continue
            if isinstance(module, (CheckpointWrapper, Block, nn.Embedding)) or module is ref_model.lm_head:
                if isinstance(module, Block) and "_checkpoint_wrapped_module" in name:
                    continue
                fully_shard(
                    module,
                    mesh=ref_mesh,
                    mp_policy=mixed_precision_policy,
                    reshard_after_forward=True,
                )
        # Same as student model: shard backbone so ref_model.backbone(input)
        # triggers FSDP's parameter gather for backbone.norm_f etc.
        fully_shard(
            ref_model.backbone,
            mesh=ref_mesh,
            mp_policy=mixed_precision_policy,
            reshard_after_forward=True,
        )
        fully_shard(
            ref_model,
            mesh=ref_mesh,
            mp_policy=mixed_precision_policy,
            reshard_after_forward=True,
        )

        # Load ref model checkpoint
        ref_ckpt_path = cfg.ref_ckpt_load_path or cfg.ckpt_load_path
        ref_save_dir = os.path.dirname(ref_ckpt_path) if os.path.isfile(ref_ckpt_path) else ref_ckpt_path
        ref_checkpointer = Checkpointer_FSDP2(
            ref_save_dir, 1000, cfg.sharding_strategy, rank, local_rank,
            mesh=fsdp_mesh if cfg.sharding_strategy == "hsdp" else None,
        )
        ref_model, _, _, _, _, _ = ref_checkpointer.load(
            ref_model,
            None,
            None,
            path=(
                os.path.join(ref_ckpt_path, "checkpoints/")
                if not os.path.isfile(ref_ckpt_path)
                else ref_ckpt_path
            ),
            strict=False,
        )

        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        if rank == 0:
            ref_params = sum(p.numel() for p in ref_model.parameters())
            print(f"--> ref model has {ref_params / 1e6} Million params (frozen)")

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print("--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer: build parameter groups with per-pattern LR scaling.
    # cfg.lr_groups format: "pattern1:scale1;pattern2:scale2"
    # First match on param name wins; unmatched params get scale 1.0. Scale 0.0 freezes params.
    lr_groups = _parse_lr_groups(cfg.lr_groups)
    is_no_decay = lambda name: any(k in name for k in ('A_log', 'D', 'dt_bias'))

    # Bucket params by (lr_scale, weight_decay)
    param_buckets = {}
    for name, param in model.named_parameters():
        lr_scale = _get_lr_scale(name, lr_groups)
        wd = 0.0 if is_no_decay(name) else 0.1
        param_buckets.setdefault((lr_scale, wd), []).append(param)

    total_params = len(list(model.named_parameters()))
    assert sum(len(v) for v in param_buckets.values()) == total_params

    if rank == 0 and cfg.lr_groups:
        print(f"--> lr_groups: {cfg.lr_groups}")
        for (scale, wd), params in param_buckets.items():
            print(f"    lr={cfg.learning_rate * scale:.2e}  wd={wd}  n_params={len(params)}")

    param_groups = [
        {"params": params, "weight_decay": wd, "lr": cfg.learning_rate * scale, "lr_scale": scale}
        for (scale, wd), params in param_buckets.items()
    ]

    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        lr=cfg.learning_rate,
    )

    # optionally load from checkpoint (when continue pretraining)
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
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]
    else:
        # Re-enforce zero-lr groups so scale=0 params stay frozen after resume
        for g in optimizer.param_groups:
            if g.get("lr_scale", 1.0) == 0.0:
                g["initial_lr"] = 0.0
                g["lr"] = 0.0

    # LR schedule
    warmup_interval = min(2000, cfg.num_steps // 20)
    warmup = lambda x: 1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2
    # linear decay for annealing
    if cfg.training_stage == "annealing":
        schedule = lambda x: min(
            warmup(x),
            1 - x / cfg.num_steps,
        )
    elif cfg.training_stage == "constant":
        # no decay for intermediate jobs
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

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
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
        ref_model=ref_model,
    )

    checkpointer.save_single_file(cfg.num_steps, model, is_compiled=cfg.use_torch_compile)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
