import re
from typing import Set

_VALID_COMPONENTS = {"attn", "mamba", "mlp"}


def _classify_param(name: str, attn_layer_idx: Set[int]) -> str:
    """Return one of {'attn', 'mamba', 'mlp', 'other'}.

    'other' = non-block params (embedding, norm_f, lm_head).
    """
    m = re.search(
        r'(?:_orig_mod\.)?backbone\.layers\.(\d+)\.(norm2?|mixer|mlp)\.',
        name,
    )
    if not m:
        return "other"
    layer_idx, comp = int(m.group(1)), m.group(2)
    if comp in ("mlp", "norm2"):
        return "mlp"
    return "attn" if layer_idx in attn_layer_idx else "mamba"


def apply_component_freeze(model, mamba_config, component: str, train_freeze: bool) -> dict:
    """Set requires_grad per the component / train_freeze polarity.

    MUST be called BEFORE fully_shard() so requires_grad lives on plain
    nn.Parameters, not DTensors.

    train_freeze=True  -> matches(component)  trainable, else frozen
    train_freeze=False -> matches(component)  frozen,    else trainable
    """
    if component not in _VALID_COMPONENTS:
        raise ValueError(
            f"component must be one of {_VALID_COMPONENTS}, got {component!r}"
        )
    attn_layer_idx = set(mamba_config.attn_layer_idx or [])

    counts = {
        "trainable": 0,
        "frozen": 0,
        "by_type": {"attn": 0, "mamba": 0, "mlp": 0, "other": 0},
    }
    for name, p in model.named_parameters():
        kind = _classify_param(name, attn_layer_idx)
        counts["by_type"][kind] += p.numel()
        matches = (kind == component)
        keep_trainable = matches if train_freeze else (not matches)
        if keep_trainable:
            counts["trainable"] += p.numel()
        else:
            p.requires_grad_(False)
            counts["frozen"] += p.numel()
    return counts
