"""Per-rank collective tracer: monkey-patches torch.distributed collectives so
every call is logged with a per-PG sequence number, op name, tensor sizes, and
a short Python stack trace.

When a rank hangs on an NCCL collective, the tail of the log file tells you
exactly which collective it was (matches NCCL's `SeqNum` in the watchdog
timeout message) and what Python call site enqueued it.

Usage:
    from fms_fsdp.utils.collective_tracer import install
    install(rank, log_dir="/gpfs/hshen/nccl_debug/<run_id>")

Output:
    - Per-rank file: <log_dir>/rank_<NNN>.log
    - Rank 0 also writes to stderr (set rank0_stderr=False to disable).

Notes:
    - Wrappers use (*args, **kwargs) pass-through so they work regardless of
      whether callers use positional or keyword arguments, and regardless of
      parameter renames across PyTorch versions (e.g. `output_tensor` vs
      `output` for `all_gather_into_tensor`).
    - The per-PG sequence counter on the Python side matches NCCL's internal
      SeqNum for the same PG, so the last line in the log before a hang names
      the exact collective whose SeqNum NCCL will report as timed out.
    - Stack traces are trimmed to user-relevant frames (torch internals
      stripped) and capped at 6 entries to keep log volume sane.
    - Every line is `flush=True` so partial output survives process kill.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from collections import defaultdict
from typing import Optional

import torch.distributed as dist

_installed = False
_rank: Optional[int] = None
_log_file = None
_rank0_stderr = True
_lock = threading.Lock()
_pg_seq: dict[str, int] = defaultdict(int)

# Frames from these files are stripped from the trace to surface user code.
_HIDE_FRAMES_FROM = (
    "torch/distributed/",
    "torch/_ops.py",
    "torch/overrides.py",
    "torch/distributed/_composable/",
    "torch/distributed/fsdp/",
    "/fms_fsdp/utils/collective_tracer.py",
)


def _pg_name(group) -> str:
    if group is None:
        return "default"
    try:
        return group.group_name
    except AttributeError:
        return f"pg@{id(group):x}"


def _short_stack(limit: int = 6) -> str:
    frames = traceback.extract_stack()
    # Drop the 2 innermost frames (this helper + the wrapper that called it).
    frames = frames[:-2]
    kept = [f for f in frames if not any(h in f.filename for h in _HIDE_FRAMES_FROM)]
    kept = kept[-limit:]
    return "; ".join(f"{os.path.basename(f.filename)}:{f.lineno}({f.name})" for f in kept)


def _log(op: str, group, in_numel: int, out_numel: int, extra: str = "") -> None:
    name = _pg_name(group)
    with _lock:
        seq = _pg_seq[name]
        _pg_seq[name] += 1
    line = (
        f"[rank={_rank}] [pg={name}] [seq={seq}] {op} "
        f"in={in_numel} out={out_numel}"
        f"{(' ' + extra) if extra else ''} | {_short_stack()}"
    )
    if _log_file is not None:
        print(line, file=_log_file, flush=True)
    if _rank0_stderr and _rank == 0:
        print(line, file=sys.stderr, flush=True)


def _pick(args, kwargs, pos, *names):
    """Pick an argument by position or any of the given kwarg names."""
    if len(args) > pos:
        return args[pos]
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return None


# --- originals (filled in at install time) ---
_orig: dict[str, object] = {}


def install(rank: int, log_dir: str = "/tmp", rank0_stderr: bool = True) -> None:
    """Install the collective tracer. Idempotent."""
    global _installed, _rank, _log_file, _rank0_stderr
    if _installed:
        return
    _installed = True
    _rank = rank
    _rank0_stderr = rank0_stderr

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"rank_{rank:03d}.log")
    _log_file = open(path, "w", buffering=1)
    print(f"[collective_tracer] rank={rank} logging to {path}", file=sys.stderr, flush=True)

    # --- all_gather_into_tensor ---
    if hasattr(dist, "all_gather_into_tensor"):
        _orig["all_gather_into_tensor"] = dist.all_gather_into_tensor
        def _agit(*args, **kwargs):
            output = _pick(args, kwargs, 0, "output_tensor", "output")
            input_ = _pick(args, kwargs, 1, "input_tensor", "input")
            group = _pick(args, kwargs, 2, "group")
            async_op = _pick(args, kwargs, 3, "async_op") or False
            in_n = input_.numel() if input_ is not None else -1
            out_n = output.numel() if output is not None else -1
            dt = input_.dtype if input_ is not None else "?"
            _log("all_gather_into_tensor", group, in_n, out_n, f"dtype={dt} async={async_op}")
            return _orig["all_gather_into_tensor"](*args, **kwargs)
        dist.all_gather_into_tensor = _agit

    # --- reduce_scatter_tensor ---
    if hasattr(dist, "reduce_scatter_tensor"):
        _orig["reduce_scatter_tensor"] = dist.reduce_scatter_tensor
        def _rst(*args, **kwargs):
            output = _pick(args, kwargs, 0, "output_tensor", "output")
            input_ = _pick(args, kwargs, 1, "input_tensor", "input")
            op = _pick(args, kwargs, 2, "op") or dist.ReduceOp.SUM
            group = _pick(args, kwargs, 3, "group")
            async_op = _pick(args, kwargs, 4, "async_op") or False
            in_n = input_.numel() if input_ is not None else -1
            out_n = output.numel() if output is not None else -1
            dt = input_.dtype if input_ is not None else "?"
            _log("reduce_scatter_tensor", group, in_n, out_n, f"dtype={dt} op={op} async={async_op}")
            return _orig["reduce_scatter_tensor"](*args, **kwargs)
        dist.reduce_scatter_tensor = _rst

    # --- all_reduce ---
    _orig["all_reduce"] = dist.all_reduce
    def _ar(*args, **kwargs):
        tensor = _pick(args, kwargs, 0, "tensor")
        op = _pick(args, kwargs, 1, "op") or dist.ReduceOp.SUM
        group = _pick(args, kwargs, 2, "group")
        async_op = _pick(args, kwargs, 3, "async_op") or False
        n = tensor.numel() if tensor is not None else -1
        dt = tensor.dtype if tensor is not None else "?"
        _log("all_reduce", group, n, n, f"dtype={dt} op={op} async={async_op}")
        return _orig["all_reduce"](*args, **kwargs)
    dist.all_reduce = _ar

    # --- all_gather (list form) ---
    _orig["all_gather"] = dist.all_gather
    def _ag(*args, **kwargs):
        tensor_list = _pick(args, kwargs, 0, "tensor_list")
        tensor = _pick(args, kwargs, 1, "tensor")
        group = _pick(args, kwargs, 2, "group")
        async_op = _pick(args, kwargs, 3, "async_op") or False
        in_n = tensor.numel() if tensor is not None else -1
        out_n = sum(t.numel() for t in tensor_list) if tensor_list else -1
        dt = tensor.dtype if tensor is not None else "?"
        _log("all_gather", group, in_n, out_n, f"dtype={dt} async={async_op}")
        return _orig["all_gather"](*args, **kwargs)
    dist.all_gather = _ag

    # --- broadcast ---
    _orig["broadcast"] = dist.broadcast
    def _bc(*args, **kwargs):
        tensor = _pick(args, kwargs, 0, "tensor")
        src = _pick(args, kwargs, 1, "src")
        group = _pick(args, kwargs, 2, "group")
        async_op = _pick(args, kwargs, 3, "async_op") or False
        n = tensor.numel() if tensor is not None else -1
        dt = tensor.dtype if tensor is not None else "?"
        _log("broadcast", group, n, n, f"src={src} dtype={dt} async={async_op}")
        return _orig["broadcast"](*args, **kwargs)
    dist.broadcast = _bc

    # --- batch_isend_irecv (RingComm / CP P2P) ---
    _orig["batch_isend_irecv"] = dist.batch_isend_irecv
    def _bii(*args, **kwargs):
        p2p_op_list = _pick(args, kwargs, 0, "p2p_op_list") or []
        total = sum(op.tensor.numel() for op in p2p_op_list)
        desc = ",".join(
            f"{op.op.__name__}(->{op.peer},n={op.tensor.numel()})"
            for op in p2p_op_list
        )
        grp = p2p_op_list[0].group if p2p_op_list else None
        _log("batch_isend_irecv", grp, total, total, f"n_ops={len(p2p_op_list)} {desc}")
        return _orig["batch_isend_irecv"](*args, **kwargs)
    dist.batch_isend_irecv = _bii

    # --- reduce (pointwise) ---
    _orig["reduce"] = dist.reduce
    def _rd(*args, **kwargs):
        tensor = _pick(args, kwargs, 0, "tensor")
        dst = _pick(args, kwargs, 1, "dst")
        op = _pick(args, kwargs, 2, "op") or dist.ReduceOp.SUM
        group = _pick(args, kwargs, 3, "group")
        async_op = _pick(args, kwargs, 4, "async_op") or False
        n = tensor.numel() if tensor is not None else -1
        dt = tensor.dtype if tensor is not None else "?"
        _log("reduce", group, n, n, f"dst={dst} dtype={dt} op={op} async={async_op}")
        return _orig["reduce"](*args, **kwargs)
    dist.reduce = _rd


def uninstall() -> None:
    """Restore all patched functions. Useful only for tests."""
    global _installed
    if not _installed:
        return
    for name, fn in _orig.items():
        setattr(dist, name, fn)
    _orig.clear()
    _installed = False
    if _log_file is not None:
        _log_file.close()
