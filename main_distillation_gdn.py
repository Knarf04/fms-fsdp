import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fire
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.optim.lr_scheduler import LambdaLR

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
)
from fms_fsdp.utils.gdn_block_utils import (
    split_gdn_block,
    GDNBlockMixer,
    GDNBlockMLP,
    GDNBlockSplit,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


@dataclass
class distill_config(config.train_config):
    """Extended config for distillation training."""
    # Teacher model (Mamba2)
    teacher_ckpt_path: str = ""
    teacher_model_variant: str = "7b"
    freeze_teacher: bool = True

    # Student model (GDN)
    student_hidden_size: int = 2048
    student_num_layers: int = 24
    student_num_heads: int = 6
    student_head_dim: int = 256
    student_expand_v: float = 2.0
    student_hidden_ratio: int = 4
    student_use_short_conv: bool = True
    student_conv_size: int = 4
    student_use_gate: bool = True
    student_fuse_norm: bool = True

    # Distillation settings
    distill_loss_type: str = "kl"  # "kl", "mse", "cosine"
    distill_temperature: float = 1.0
    distill_alpha: float = 0.5  # weight for distillation loss vs CE loss
    distill_layers: str = ""  # layer-wise distillation, e.g., "0:0,1:1,2:2" (teacher:student)
    hidden_distill_weight: float = 0.0  # weight for hidden state distillation


def get_student_wrapping_policy(cfg, model):
    """Get FSDP wrapping policy for the GDN student model."""
    wrap_cls = [GatedDeltaNetBlock, nn.Embedding]

    if cfg.freeze_layer and not cfg.full_block:
        wrap_cls += [GDNBlockMixer, GDNBlockMLP, GDNBlockSplit]

    def lambda_fn(module: nn.Module):
        return isinstance(module, tuple(wrap_cls)) or module is model.lm_head

    return CustomPolicy(lambda_fn)


def get_teacher_wrapping_policy(model):
    """Get FSDP wrapping policy for the Mamba2 teacher model."""
    wrap_cls = [Block, nn.Embedding]

    def lambda_fn(module: nn.Module):
        return isinstance(module, tuple(wrap_cls)) or module is model.lm_head

    return CustomPolicy(lambda_fn)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_type: str = "kl",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute distillation loss between student and teacher logits.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        loss_type: "kl", "mse", or "cosine"
        temperature: softmax temperature for KL divergence

    Returns:
        Scalar loss tensor
    """
    if loss_type == "kl":
        # KL divergence with temperature scaling
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        # Scale by T^2 as per Hinton et al.
        loss = loss * (temperature ** 2)
    elif loss_type == "mse":
        loss = F.mse_loss(student_logits, teacher_logits)
    elif loss_type == "cosine":
        # Cosine similarity loss
        student_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        loss = (1 - cos_sim).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


def compute_hidden_distillation_loss(
    student_hiddens: list[torch.Tensor],
    teacher_hiddens: list[torch.Tensor],
    layer_mapping: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Compute hidden state distillation loss between mapped layers.

    Args:
        student_hiddens: List of hidden states from student model
        teacher_hiddens: List of hidden states from teacher model
        layer_mapping: List of (teacher_layer, student_layer) tuples

    Returns:
        Scalar loss tensor
    """
    if not layer_mapping:
        return torch.tensor(0.0)

    total_loss = 0.0
    for teacher_idx, student_idx in layer_mapping:
        teacher_h = teacher_hiddens[teacher_idx]
        student_h = student_hiddens[student_idx]

        # Project if dimensions don't match
        if teacher_h.shape[-1] != student_h.shape[-1]:
            # TODO: Add projection layer if needed
            continue

        total_loss += F.mse_loss(student_h, teacher_h)

    return total_loss / len(layer_mapping)


def parse_layer_mapping(mapping_str: str) -> list[tuple[int, int]]:
    """Parse layer mapping string like '0:0,1:1,2:2' into list of tuples."""
    if not mapping_str:
        return []

    mappings = []
    for pair in mapping_str.split(","):
        teacher_idx, student_idx = pair.split(":")
        mappings.append((int(teacher_idx), int(student_idx)))
    return mappings


def distill_train(
    cfg,
    teacher_model,
    student_model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
):
    """
    Distillation training loop.
    """
    teacher_model.eval()
    student_model.train()

    layer_mapping = parse_layer_mapping(cfg.distill_layers)

    for step, batch in enumerate(train_loader, start=start_step):
        if step >= cfg.num_steps:
            break

        input_ids = batch["input_ids"].to(local_rank)
        labels = batch.get("labels", input_ids.clone())
        labels = labels.to(local_rank)

        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids)
            if isinstance(teacher_outputs, tuple):
                teacher_logits = teacher_outputs[0]
            else:
                teacher_logits = teacher_outputs

        # Student forward
        student_outputs = student_model(input_ids)
        if isinstance(student_outputs, tuple):
            student_logits = student_outputs[0]
        else:
            student_logits = student_outputs

        # Compute losses
        # 1. Cross-entropy loss (student prediction vs ground truth)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 2. Distillation loss (student vs teacher)
        distill_loss = compute_distillation_loss(
            student_logits,
            teacher_logits,
            loss_type=cfg.distill_loss_type,
            temperature=cfg.distill_temperature,
        )

        # 3. Hidden state distillation (optional)
        hidden_loss = torch.tensor(0.0, device=local_rank)
        # TODO: Extract hidden states if cfg.hidden_distill_weight > 0

        # Combined loss
        total_loss = (
            (1 - cfg.distill_alpha) * ce_loss
            + cfg.distill_alpha * distill_loss
            + cfg.hidden_distill_weight * hidden_loss
        )

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()

        if cfg.grad_clip_thresh > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), cfg.grad_clip_thresh)

        optimizer.step()
        scheduler.step()

        tokens_seen += input_ids.numel()

        # Logging
        if step % cfg.report_interval == 0 and rank == 0:
            print(
                f"Step {step}: total_loss={total_loss.item():.4f}, "
                f"ce_loss={ce_loss.item():.4f}, distill_loss={distill_loss.item():.4f}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Checkpointing
        if step % cfg.checkpoint_interval == 0 and step > 0:
            checkpointer.save(step, student_model, optimizer, None, tokens_seen)

        if profiler:
            profiler.step()

    return tokens_seen


def main(**kwargs):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Get configs
    cfg = distill_config()
    update_config(cfg, **kwargs)

    # Ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # Setup
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        Path.home(), ".triton", "cache", str(local_rank)
    )

    # Get policies
    (
        mixed_precision_policy,
        _,
        sharding_strategy_policy,
        apply_selective_ac,
        _,
    ) = get_policies(cfg, rank, (GatedDeltaNetBlock,))

    if cfg.low_cpu_fsdp:
        def param_init_fn(module):
            module.to_empty(device=torch.cuda.current_device())
    else:
        param_init_fn = None

    # Device mesh setup
    def get_1D_world_mesh(world_size: int) -> DeviceMesh:
        mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
        return mesh

    def get_2D_world_mesh(world_size: int) -> DeviceMesh:
        num_gpu_per_node = torch.cuda.device_count()
        assert world_size % num_gpu_per_node == 0
        mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (world_size // num_gpu_per_node, num_gpu_per_node),
            mesh_dim_names=("inter_node", "intra_node"),
        )
        return mesh

    requires_2d_mesh = cfg.sharding_strategy == "hsdp"
    if requires_2d_mesh:
        mesh = get_2D_world_mesh(world_size)
        fsdp_mesh = mesh
    else:
        mesh = get_1D_world_mesh(world_size)
        fsdp_mesh = mesh

    dp_degree = world_size

    # =========================================================================
    # Create Teacher Model (Mamba2)
    # =========================================================================
    if rank == 0:
        print("Creating teacher model (Mamba2)...")

    teacher_config_data = get_model_config(cfg.teacher_model_variant)
    teacher_mamba_config = MambaConfig(**teacher_config_data)

    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            teacher_model = MambaLMHeadModel(teacher_mamba_config)
    else:
        teacher_model = MambaLMHeadModel(teacher_mamba_config)

    # Freeze teacher
    if cfg.freeze_teacher:
        for param in teacher_model.parameters():
            param.requires_grad = False

    teacher_wrapping_policy = get_teacher_wrapping_policy(teacher_model)

    teacher_model = FSDP(
        teacher_model,
        device_mesh=fsdp_mesh,
        auto_wrap_policy=teacher_wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_orig_params,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )

    # Load teacher checkpoint
    if cfg.teacher_ckpt_path:
        if rank == 0:
            print(f"Loading teacher checkpoint from {cfg.teacher_ckpt_path}")
        # TODO: Load teacher checkpoint
        # checkpointer.load(teacher_model, ...)

    if rank == 0:
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"Teacher model has {teacher_params / 1e6:.2f}M params")

    # =========================================================================
    # Create Student Model (GDN)
    # =========================================================================
    if rank == 0:
        print("Creating student model (GatedDeltaNet)...")

    student_config = GatedDeltaNetConfig(
        hidden_size=cfg.student_hidden_size,
        num_hidden_layers=cfg.student_num_layers,
        num_heads=cfg.student_num_heads,
        head_dim=cfg.student_head_dim,
        expand_v=cfg.student_expand_v,
        hidden_ratio=cfg.student_hidden_ratio,
        use_short_conv=cfg.student_use_short_conv,
        conv_size=cfg.student_conv_size,
        use_gate=cfg.student_use_gate,
        fuse_norm=cfg.student_fuse_norm,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.seq_length,
    )

    if rank == 0:
        print(f"Student config: {student_config}")

    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            student_model = GatedDeltaNetForCausalLM(student_config)
    else:
        student_model = GatedDeltaNetForCausalLM(student_config)

    student_wrapping_policy = get_student_wrapping_policy(cfg, student_model)

    student_model = FSDP(
        student_model,
        device_mesh=fsdp_mesh,
        auto_wrap_policy=student_wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_orig_params,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )

    if rank == 0:
        student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"Student model has {student_params / 1e6:.2f}M trainable params")

    # FSDP activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print("--> applying FSDP activation checkpointing...")
        apply_selective_ac(student_model, p=cfg.selective_checkpointing)

    # Torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print("--> enabling torch compile...")
        torch._dynamo.config.accumulated_cache_size_limit = 128
        student_model = torch.compile(student_model)
        if not cfg.freeze_teacher:
            teacher_model = torch.compile(teacher_model)

    # =========================================================================
    # Data loader
    # =========================================================================
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size, dp_degree)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    # =========================================================================
    # Optimizer (only for student)
    # =========================================================================
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Checkpointer
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )

    # Optionally load student checkpoint
    student_model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
        student_model,
        optimizer,
        None,
        path=(
            os.path.join(cfg.ckpt_load_path, "checkpoints/")
            if cfg.ckpt_load_path and not os.path.isfile(cfg.ckpt_load_path)
            else cfg.ckpt_load_path
        ),
        strict=False,
    )

    if not is_resuming:
        start_step = 0
        tokens_seen = 0
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.learning_rate

    # LR schedule
    warmup_interval = min(2000, cfg.num_steps // 20)
    warmup = lambda x: 1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2

    if cfg.training_stage == "annealing":
        schedule = lambda x: min(warmup(x), 1 - x / cfg.num_steps)
    elif cfg.training_stage == "constant":
        schedule = warmup
    else:
        schedule = lambda x: min(
            warmup(x),
            0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )

    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # Profiler
    profiler = get_profiler(cfg, rank)

    # =========================================================================
    # Train
    # =========================================================================
    if rank == 0:
        print(f"Starting distillation training for {cfg.num_steps} steps")

    tokens_seen = distill_train(
        cfg,
        teacher_model,
        student_model,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        profiler,
        checkpointer,
        start_step,
        tokens_seen,
    )

    checkpointer.save_single_file(cfg.num_steps, student_model)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
