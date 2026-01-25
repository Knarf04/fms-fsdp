from typing import Optional

import torch
from torch import Tensor, nn

from fla.modules.layernorm import RMSNorm
from fla.models.gated_deltanet import GatedDeltaNetBlock


# Custom Block classes that split (attn, attn_norm) and (mlp, mlp_norm) into 2 separate nn.Modules for cleaner FSDP
# Only necessary if the block is partially frozen for training
class GDNBlockMixer(nn.Module):
    """
    Wraps the attention/GatedDeltaNet mixer and its normalization layer.
    Used for partial freezing with FSDP support.
    """
    def __init__(
        self,
        hidden_size: int,
        attn_cls,
        norm_cls=RMSNorm,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fuse_norm = fuse_norm
        self.attn_norm = norm_cls(hidden_size, eps=norm_eps)
        self.attn = attn_cls(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        if self.fuse_norm:
            hidden_states, residual = self.attn_norm(
                hidden_states, residual=residual, prenorm=True
            )
        else:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.attn_norm(hidden_states)

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        # Handle different return formats from attention
        if isinstance(attn_outputs, tuple):
            hidden_states = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            past_key_values = attn_outputs[2] if len(attn_outputs) > 2 else None
        else:
            hidden_states = attn_outputs
            attn_weights = None
            past_key_values = None

        return hidden_states, residual, attn_weights, past_key_values


class GDNBlockMLP(nn.Module):
    """
    Wraps the MLP and its normalization layer.
    Used for partial freezing with FSDP support.
    """
    def __init__(
        self,
        hidden_size: int,
        mlp_cls,
        norm_cls=RMSNorm,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fuse_norm = fuse_norm
        self.mlp_norm = norm_cls(hidden_size, eps=norm_eps)
        self.mlp = mlp_cls(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
    ):
        if self.fuse_norm:
            hidden_states, residual = self.mlp_norm(
                hidden_states, residual=residual, prenorm=True
            )
        else:
            residual = hidden_states + residual
            hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GDNBlockSplit(nn.Module):
    """
    A split version of GatedDeltaNetBlock that separates mixer and MLP into
    distinct submodules for better FSDP support when partially freezing layers.

    This allows FSDP to wrap GDNBlockMixer and GDNBlockMLP separately, enabling
    different freezing states for the attention/mixer vs MLP components.
    """
    def __init__(
        self,
        hidden_size: int,
        attn_cls,
        mlp_cls,
        norm_cls=RMSNorm,
        norm_eps: float = 1e-6,
        fuse_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fuse_norm = fuse_norm
        self.mixer_norm = GDNBlockMixer(
            hidden_size, attn_cls, norm_cls, norm_eps, fuse_norm
        )
        self.mlp_norm = GDNBlockMLP(
            hidden_size, mlp_cls, norm_cls, norm_eps, fuse_norm
        )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        # Mixer (attention) pass
        hidden_states, residual, attn_weights, past_key_values = self.mixer_norm(
            hidden_states,
            residual=residual,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        # MLP pass
        hidden_states, residual = self.mlp_norm(hidden_states, residual)

        # Match the expected return format of GatedDeltaNetBlock
        if not self.fuse_norm:
            hidden_states = hidden_states + residual

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_values,)

        return outputs if len(outputs) > 1 else hidden_states


def split_gdn_block(block: GatedDeltaNetBlock) -> GDNBlockSplit:
    """
    Converts an existing GatedDeltaNetBlock instance into a GDNBlockSplit instance.
    This is used when partial freezing requires separate FSDP wrapping of mixer vs MLP.

    Args:
        block: A GatedDeltaNetBlock instance from fla.models.gated_deltanet

    Returns:
        GDNBlockSplit: A new block with the same weights but split structure
    """
    # Placeholder class to satisfy the constructor
    class _Placeholder(nn.Module):
        def __init__(self, hidden_size, *args, **kwargs):
            super().__init__()

    # Get configuration from the existing block
    hidden_size = block.attn_norm.weight.shape[0]
    norm_cls = type(block.attn_norm)
    norm_eps = block.attn_norm.eps if hasattr(block.attn_norm, 'eps') else 1e-6
    fuse_norm = getattr(block, 'fuse_norm', True)

    # Create the split block with placeholder modules
    block_split = GDNBlockSplit(
        hidden_size=hidden_size,
        attn_cls=_Placeholder,
        mlp_cls=_Placeholder,
        norm_cls=norm_cls,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
    )

    # Transfer the actual modules from the original block
    block_split.mixer_norm.attn = block.attn
    block_split.mixer_norm.attn_norm = block.attn_norm
    block_split.mlp_norm.mlp = block.mlp
    block_split.mlp_norm.mlp_norm = block.mlp_norm

    # Copy other attributes
    block_split.fuse_norm = fuse_norm

    return block_split
