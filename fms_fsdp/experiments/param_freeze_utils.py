import re
from typing import Optional, Set, Tuple

_VALID_COMPONENTS = {"attn", "mamba", "mlp", "mamba_post_attn", "norm", "mamba_norm"}

# Matches the per-Block pre-mixer norm (".norm."), pre-mlp norm (".norm2."),
# and the model-level final norm ("backbone.norm_f."), with optional
# torch.compile prefix. Does NOT match the mixer-internal RMSNorm inside Mamba2.
_NORM_PARAM_RE = re.compile(
    r'(?:_orig_mod\.)?backbone\.(?:layers\.\d+\.norm2?|norm_f)\.'
)

# Matches the mixer-internal RMSNorm inside Mamba2 (MambaRMSNormGated at
# mixer.norm.*). Layer-index-gated against attn_layer_idx at match time
# so attention layers' mixer params are never picked up here.
_MAMBA_NORM_PARAM_RE = re.compile(
    r'(?:_orig_mod\.)?backbone\.layers\.(\d+)\.mixer\.norm\.'
)


def _classify_param(
    name: str, attn_layer_idx: Set[int]
) -> Tuple[str, Optional[int]]:
    """Return (kind, layer_idx).

    kind ∈ {'attn', 'mamba', 'mlp', 'other'}.
    'other' = non-block params (embedding, norm_f, lm_head); layer_idx=None.
    """
    m = re.search(
        r'(?:_orig_mod\.)?backbone\.layers\.(\d+)\.(norm2?|mixer|mlp)\.',
        name,
    )
    if not m:
        return "other", None
    layer_idx, comp = int(m.group(1)), m.group(2)
    if comp in ("mlp", "norm2"):
        return "mlp", layer_idx
    return ("attn" if layer_idx in attn_layer_idx else "mamba"), layer_idx


def apply_component_freeze(model, mamba_config, component: str, train_freeze: bool) -> dict:
    """Set requires_grad per the component / train_freeze polarity.

    MUST be called BEFORE fully_shard() so requires_grad lives on plain
    nn.Parameters, not DTensors.

    component:
        "attn"            -> attention mixer + its pre-norm
        "mamba"           -> every mamba mixer + its pre-norm
        "mlp"             -> every MLP + its pre-norm (norm2)
        "mamba_post_attn" -> only the mamba layers sitting immediately after
                             an attention layer (index = attn_idx + 1, when
                             that slot is itself a mamba layer)
        "norm"            -> all norms: per-Block pre-mixer (.norm.), pre-mlp
                             (.norm2.), and the final backbone.norm_f. Mixers
                             and MLPs are frozen.
        "mamba_norm"      -> ONLY the mixer-internal RMSNorm inside Mamba2
                             layers (MambaRMSNormGated at mixer.norm.*).
                             Excludes attn-layer mixers, pre-mixer norms,
                             pre-mlp norms, and norm_f.

    train_freeze:
        True  -> matches(component)  trainable, else frozen
        False -> matches(component)  frozen,    else trainable
    """
    if component not in _VALID_COMPONENTS:
        raise ValueError(
            f"component must be one of {_VALID_COMPONENTS}, got {component!r}"
        )
    attn_layer_idx = set(mamba_config.attn_layer_idx or [])
    n_layer = mamba_config.n_layer
    post_attn_mamba = {
        i + 1 for i in attn_layer_idx
        if (i + 1) < n_layer and (i + 1) not in attn_layer_idx
    }

    target_param_names: Set[str] = set()
    counts = {
        "target_param_names": target_param_names,
        "trainable": 0,
        "frozen": 0,
        "by_type": {"attn": 0, "mamba": 0, "mlp": 0, "other": 0},
    }
    for name, p in model.named_parameters():
        kind, layer_idx = _classify_param(name, attn_layer_idx)
        counts["by_type"][kind] += p.numel()
        if component == "mamba_post_attn":
            matches = (kind == "mamba") and (layer_idx in post_attn_mamba)
        elif component == "norm":
            # Norm cuts across attn/mamba/mlp kinds — match by name pattern.
            matches = bool(_NORM_PARAM_RE.search(name))
        elif component == "mamba_norm":
            # Mixer-internal RMSNorm only on mamba layers. Match by name +
            # layer-index gating so attn-layer mixers (different module shape,
            # but defensively excluded) never qualify.
            mn = _MAMBA_NORM_PARAM_RE.search(name)
            matches = bool(mn) and (int(mn.group(1)) not in attn_layer_idx)
        else:
            matches = (kind == component)
        is_target = matches if train_freeze else (not matches)
        if is_target:
            target_param_names.add(name)
            counts["trainable"] += p.numel()
        else:
            counts["frozen"] += p.numel()
            p.requires_grad_(False)
    return counts
