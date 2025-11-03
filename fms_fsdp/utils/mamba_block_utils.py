from typing import Optional

import torch
from torch import Tensor, nn

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.modules.block import Block


def add_fsdp_heartbeat(module: nn.Module) -> None:
    """
    Add a tiny trainable parameter to a frozen module to ensure FSDP processes it.

    When a module is completely frozen, FSDP may skip gradient computation for it,
    which can break gradient flow to upstream trainable modules. This heartbeat
    parameter ensures FSDP still processes the module during backward pass.

    Only adds the heartbeat if the module is completely frozen (all params have
    requires_grad=False) and doesn't already have one.
    """
    if not hasattr(module, '_fsdp_heartbeat'):
        return  # Module doesn't support heartbeat

    if module._fsdp_heartbeat is not None:
        return  # Already has heartbeat

    # Get non-heartbeat params
    other_params = [p for n, p in module.named_parameters() if "_fsdp_heartbeat" not in n]

    if not other_params:
        return  # No params to reference

    # Only add heartbeat if module is completely frozen
    if all(not p.requires_grad for p in other_params):
        ref_p = other_params[0]
        module._fsdp_heartbeat = nn.Parameter(
            torch.zeros(1, dtype=ref_p.dtype, device=ref_p.device),
            requires_grad=True
        )


class BlockMixer(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        
        # Heartbeat parameter for FSDP compatibility when frozen
        self._fsdp_heartbeat: Optional[nn.Parameter] = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        if len(self.mixer.experiments) == 0:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        else:
            hidden_states, experiment_out = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        # Inject heartbeat into computation graph to ensure FSDP processes this module
        if self._fsdp_heartbeat is not None:
            hidden_states = hidden_states + self._fsdp_heartbeat.mul(0)

        if len(self.mixer.experiments) == 0:
            return hidden_states, residual
        else:
            return hidden_states, residual, experiment_out
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class LMHeadWrapper(nn.Module):
    """
    Wrapper for lm_head (nn.Linear) that supports FSDP heartbeat when frozen.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self._fsdp_heartbeat: Optional[nn.Parameter] = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = self.linear(hidden_states)
        if self._fsdp_heartbeat is not None:
            output = output + self._fsdp_heartbeat.mul(0)
        return output

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias


def wrap_lm_head(model) -> 'LMHeadWrapper':
    """
    Wrap model.lm_head with LMHeadWrapper for FSDP heartbeat support.
    Returns the wrapper (also sets model.lm_head to the wrapper).
    """
    if isinstance(model.lm_head, LMHeadWrapper):
        return model.lm_head  # Already wrapped

    wrapper = LMHeadWrapper(model.lm_head)
    model.lm_head = wrapper
    return wrapper


class BlockMLP(nn.Module):
    def __init__(
        self, dim, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm2 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm2, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        # Heartbeat parameter for FSDP compatibility when frozen
        self._fsdp_heartbeat: Optional[nn.Parameter] = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None,
    ):
        if not self.fused_add_norm:
            residual = hidden_states + residual
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm2.weight,
                self.norm2.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm2.eps,
                is_rms_norm=isinstance(self.norm2, RMSNorm)
            )
        hidden_states = self.mlp(hidden_states)

        # Inject heartbeat into computation graph to ensure FSDP processes this module
        if self._fsdp_heartbeat is not None:
            hidden_states = hidden_states + self._fsdp_heartbeat.mul(0)

        return hidden_states, residual

class BlockSplit(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer_norm = BlockMixer(dim, mixer_cls, norm_cls, fused_add_norm, residual_in_fp32)
        if mlp_cls is not nn.Identity:
            self.mlp_norm = BlockMLP(dim, mlp_cls, norm_cls, fused_add_norm, residual_in_fp32)
        else:
            self.mlp_norm = None
        self.no_output = (len(self.mixer_norm.mixer.experiments) == 0)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        if self.no_output:
            hidden_states, residual = self.mixer_norm(hidden_states, residual, inference_params=inference_params, **mixer_kwargs)
        else:
            hidden_states, residual, experiment_out = self.mixer_norm(hidden_states, residual, inference_params=inference_params, **mixer_kwargs)

        if self.mlp_norm is not None:
            hidden_states, residual = self.mlp_norm(hidden_states, residual)

        if self.no_output:
            return hidden_states, residual
        else:
            return hidden_states, residual, experiment_out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer_norm.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def split_block(block: Block) -> BlockSplit:
    class _Placeholder(nn.Module):
        def __init__(self, dim, *args, **kwargs):
            super().__init__()
            self.experiments = [] 

    block_split = BlockSplit(
        dim=block.norm.weight.shape[0],
        mixer_cls=_Placeholder,
        mlp_cls=_Placeholder if block.mlp is not None else nn.Identity,
        norm_cls=type(block.norm),
        fused_add_norm=block.fused_add_norm,
        residual_in_fp32=block.residual_in_fp32
    )

    block_split.mixer_norm.mixer = block.mixer
    block_split.mixer_norm.norm = block.norm

    if block.mlp is not None and block_split.mlp_norm is not None:
        block_split.mlp_norm.mlp = block.mlp
        block_split.mlp_norm.norm2 = block.norm2
    elif block.mlp is None:
        block_split.mlp_norm = None

    block_split.residual_in_fp32 = block.residual_in_fp32
    block_split.fused_add_norm = block.fused_add_norm
    block_split.no_output = (len(block_split.mixer_norm.mixer.experiments) == 0)

    return block_split