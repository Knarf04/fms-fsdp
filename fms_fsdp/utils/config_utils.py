from fms.models.llama import LLaMAConfig

from fms_fsdp.config import train_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def get_model_config(model_variant):
    if model_variant == "llama2_70b":
        model_config = LLaMAConfig(
            emb_dim=8192,
            multiple_of=4096,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=28672 / 8192,
        )
    elif model_variant == "llama2_34b":
        model_config = LLaMAConfig(
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=48,
            hidden_grow_factor=22016 / 8192,
            max_expected_seq_len=16384,
            rope_theta=1000000.0,
        )
    elif model_variant == "llama2_13b":
        model_config = LLaMAConfig(
            emb_dim=5120,
            nheads=40,
            nlayers=40,
            hidden_grow_factor=13824 / 5120,
        )
    elif model_variant == "llama2_7b":
        model_config = LLaMAConfig(
            hidden_grow_factor=11008 / 4096,
            kvheads=32,
        )
    elif model_variant == "llama2_1.4b":
        model_config = LLaMAConfig(
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            hidden_grow_factor=3,
            kvheads=4,
        )
    elif model_variant == "llama3_8b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_8b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_1.8b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_1.8b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_3.2b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_3.2b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=4096,
        )
    elif model_variant == "llama3_70b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_70b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_194m_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1024,
            nheads=8,
            nlayers=10,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "mamba_9.8b":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 0,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_1b":
        model_config = {
            "d_model": 1280,
            "d_intermediate": 3072,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 10,
                "num_heads_kv": 5,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 0,
                "rotary_emb_base": 10_000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_1b_rope":
        model_config = {
            "d_model": 1280,
            "d_intermediate": 3072,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 10,
                "num_heads_kv": 5,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
                "rotary_emb_base": 10_000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_1b_group5":
        model_config = {
            "d_model": 1280,
            "d_intermediate": 3072,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {
                "layer": "Mamba2", 
                "ngroups": 5,
            },
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 10,
                "num_heads_kv": 5,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 0,
                "rotary_emb_base": 10_000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_1b_group5_rope":
        model_config = {
            "d_model": 1280,
            "d_intermediate": 3072,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {
                "layer": "Mamba2", 
                "ngroups": 5,
            },
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 10,
                "num_heads_kv": 5,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
                "rotary_emb_base": 10_000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "llama_1b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=False,
            prune_thresh=0,
        )
    elif model_variant == "llama_1b_triton":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=20,
            kvheads=5,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama_1b_aux20":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            sparsity=.05,
        )
    elif model_variant == "llama_1b_halfrope":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=.5,
        )
    elif model_variant == "llama_1b_rope652":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=652.0,
        )
    elif model_variant == "llama_1b_unrope6":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=6.283,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
        )
    elif model_variant == "llama_1b_unrope32":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=32,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
            prune=False,
            prune_thresh=0,
        )
    elif model_variant == "llama_1b_unrope32_triton":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=20,
            kvheads=5,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=32,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
        )
    elif model_variant == "llama_1b_nope":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=False,
            prune_thresh=0,
        )
    elif model_variant == "llama_3b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=4,
            nlayers=48,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama_3b_halfrope":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=4,
            nlayers=48,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=.5,
        )
    elif model_variant == "llama_3b_rope652":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=4,
            nlayers=48,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=652.0,
        )
    elif model_variant == "llama_3b_unrope6":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=4,
            nlayers=48,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=6.283,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
        )
    elif model_variant == "llama_3b_unrope32":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=4,
            nlayers=48,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=32,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
        )
    elif model_variant == "mamba_9.8b_32k":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
                "rotary_emb_base": 80000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_9.8b_64k":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
                "rotary_emb_base": 160000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_9.8b_128k":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
                "rotary_emb_base": 320000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_9.8b_512k":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
                "rotary_emb_base": 1280000,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "llama_1b_p=1/1000":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/1000),
        )
    elif model_variant == "llama_1b_unrope32_p=1/1000":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=32,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
            prune=True,
            prune_thresh=float(1/1000),
        )
    elif model_variant == "llama_1b_nope_p=1/1000":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/1000),
        )
    elif model_variant == "llama_1b_p=1/100":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/100),
        )
    elif model_variant == "llama_1b_unrope32_p=1/100":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=32,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
            prune=True,
            prune_thresh=float(1/100),
        )
    elif model_variant == "llama_1b_nope_p=1/100":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/100),
        )
    elif model_variant == "llama_1b_p=1/20":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/20),
        )
    elif model_variant == "llama_1b_unrope32_p=1/20":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=32,
            rope_scaling={"rope_type":"unrope"},
            rope_partial=.5,
            prune=True,
            prune_thresh=float(1/20),
        )
    elif model_variant == "llama_1b_nope_p=1/20":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/20),
        )
    elif model_variant == "llama_1b_p=1/15":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/15),
        )
    elif model_variant == "llama_1b_nope_p=1/15":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/15),
        )
    elif model_variant == "llama_1b_p=1/12":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/12),
        )
    elif model_variant == "llama_1b_nope_p=1/12":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/12),
        )
    elif model_variant == "llama_1b_p=1/11":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/11),
        )
    elif model_variant == "llama_1b_nope_p=1/11":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/11),
        )
    elif model_variant == "llama_1b_p=1/10":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/10),
        )
    elif model_variant == "llama_1b_nope_p=1/10":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/10),
        )
    elif model_variant == "llama_1b_p=1/9":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/9),
        )
    elif model_variant == "llama_1b_nope_p=1/9":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/9),
        )
    elif model_variant == "llama_1b_p=1/8":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/8),
        )
    elif model_variant == "llama_1b_nope_p=1/8":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/8),
        )
    elif model_variant == "llama_1b_p=1/7":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/7),
        )
    elif model_variant == "llama_1b_nope_p=1/7":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/7),
        )
    elif model_variant == "llama_1b_p=1/6":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/6),
        )
    elif model_variant == "llama_1b_nope_p=1/6":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/6),
        )
    elif model_variant == "llama_1b_p=1/5":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/5),
        )
    elif model_variant == "llama_1b_nope_p=1/5":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/5),
        )
    elif model_variant == "llama_1b_p=1/4":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/4),
        )
    elif model_variant == "llama_1b_nope_p=1/4":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/4),
        )
    elif model_variant == "llama_1b_p=1/3":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/3),
        )
    elif model_variant == "llama_1b_nope_p=1/3":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/3),
        )
    elif model_variant == "llama_1b_p=1/2":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_thresh=float(1/2),
        )
    elif model_variant == "llama_1b_nope_p=1/2":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_partial=0,
            prune=True,
            prune_thresh=float(1/2),
        )
    elif model_variant == "llama_1b_yarn":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_scaling={"rope_type": "yarn"},
            prune=False,
        )
    elif model_variant == "llama_1b_yarn_old":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            rope_scaling={"rope_type": "yarn"},
        )
    elif model_variant == "llama_1b_fixed":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1280,
            nheads=16,
            kvheads=4,
            nlayers=32,
            hidden_grow_factor=3.2,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
            prune=True,
            prune_topk=256,
        )
    else:
        raise ValueError(f"model variant {model_variant} not supported.")

    return model_config
