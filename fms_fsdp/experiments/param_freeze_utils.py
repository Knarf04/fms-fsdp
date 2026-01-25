import re
from typing import Dict, List, Set


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

