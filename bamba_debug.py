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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main(**kwargs):
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.INFO)
    
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

    # get policy. NOTE: @goon - overriding {wrapping_policy, param_init_fn} below
    block = Block
    (
        mixed_precision_policy,
        _,
        sharding_strategy_policy,
        apply_selective_ac,
        _,  # NOTE: @goon - We'll override param_init_fn for mamba below
    ) = get_policies(cfg, rank, block)
    if cfg.low_cpu_fsdp:
        # NOTE: @goon - the params will be junk after using this. Only intended to be used in
        # conjunction with loading proper weights from a checkpoint.
        def param_init_fn(module):
            module.to_empty(device=torch.cuda.current_device())
    else:
        param_init_fn = None

    # Meshes for FSDP and CP. NOTE: @goon - Getting hangs and/or OOMs if I don't explicitly specify
    # the FSDP mesh when using 4+ nodes with HSDP + in-node-CP.
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

    requires_2d_mesh = (cfg.sharding_strategy == "hsdp") or (
        cfg.cp and not cfg.cp_over_world
    )
    if requires_2d_mesh:
        mesh = get_2D_world_mesh(world_size)
        fsdp_mesh = mesh
        cp_mesh = mesh["intra_node"] if cfg.cp else None
    else:
        mesh = get_1D_world_mesh(world_size)
        fsdp_mesh = mesh
        cp_mesh = mesh if cfg.cp else None

    if cfg.cp:
        cp_degree = world_size if cfg.cp_over_world else torch.cuda.device_count()
    else:
        cp_degree = 1

    dp_degree = world_size // cp_degree

    # get model
    config_data = get_model_config(cfg.model_variant)
    mamba_config = MambaConfig(**config_data)

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

    def lambda_fn(module: nn.Module):
        return isinstance(module, (Block, nn.Embedding)) or module is model.lm_head

    wrapping_policy = CustomPolicy(lambda_fn)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # FSDP
    model = FSDP(
        model,
        device_mesh=fsdp_mesh,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )
    if rank == 0:
        print(model)

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print("--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=cfg.selective_checkpointing)

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print("--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    model, optimizer, _, _, _, _ = checkpointer.load(
        model,
        optimizer,
        None,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
        if not os.path.isfile(cfg.ckpt_load_path)
        else cfg.ckpt_load_path,
        strict=False,
    )


    # huggingface model
    tokenizer_hf = AutoTokenizer.from_pretrained(cfg.hf_load_path, trust_remote_code=True)
    tokenizer_hf.pad_token = tokenizer_hf.eos_token
    tokenizer_hf.padding_side = 'left'
    config_hf = AutoConfig.from_pretrained(cfg.hf_load_path)
    config_hf.experiments = {
        "upi": "/gpfs/hshen/UPI_configs/upi_mask_layer.pt",
    }    
    model_hf = AutoModelForCausalLM.from_pretrained(
        cfg.hf_load_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config_hf
    )
    if rank == 0:
        print(model_hf.model.rotery_emb.rope_type)
        print(model_hf.model.rotery_emb.max_seq_len_cached)
        print(model_hf.model.rotery_emb.original_max_seq_len)
        print(model_hf.model.rotery_emb.config)
        print(model_hf.model.rotery_emb.attention_scaling)

    x = torch.arange(1024)[None, ...].to(torch.int64) # Add batch size
    y = model(x).logits.cpu()
    y_hf = model_hf(x).logits.cpu()
    
    if rank == 0:
        print("Mamba_ssm model: ", y)
        print("Huggingface model: ", y_hf)

        print("Max error: ", (y - y_hf).abs().max())
        print(torch.allclose(y, y_hf))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
