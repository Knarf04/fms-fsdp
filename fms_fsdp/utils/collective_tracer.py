"""Per-rank collective tracer: monkey-patches torch.distributed collectives so
every call is logged with a per-PG sequence number, op name, tensor sizes, and
a short Python stack trace.

When a rank hangs on an NCCL collective, the tail of the log file tells you
exactly which collective it was (matches NCCL's `SeqNum` in the watchdog
timeout message) and what Python call site enqueued it.

Usage:
    from fms_fsdp.utils.collective_tracer import install
    install(rank, log_dir="/tmp")   # call ONCE, as early as possible

Output:
    - Per-rank file: /tmp/collective_trace_rank_<rank>.log
    - Rank 0 also writes to stderr (set rank0_stderr=False to disable).

Notes:
    - The per-PG sequence counter on the Python side matches NCCL's internal
      SeqNum for the same PG, so the last line in the log before a hang names
      the exact collective whose SeqNum NCCL will report as timed out.
    - Stack traces are trimmed to the user-relevant frames (torch internals
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
        # ProcessGroup in recent torch has a .group_name attribute.
        return group.group_name
    except AttributeError:
        return f"pg@{id(group):x}"


def _short_stack(limit: int = 6) -> str:
    frames = traceback.extract_stack()
    # Drop the 2 innermost frames (this helper + the wrapper that called it).
    frames = frames[:-2]
    # Strip torch / FSDP internal frames.
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
    path = os.path.join(log_dir, f"collective_trace_rank_{rank}.log")
    _log_file = open(path, "w", buffering=1)  # line-buffered
    print(f"[collective_tracer] rank={rank} logging to {path}", file=sys.stderr, flush=True)

    # --- all_gather_into_tensor ---
    if hasattr(dist, "all_gather_into_tensor"):
        _orig["all_gather_into_tensor"] = dist.all_gather_into_tensor
        def _agit(output, input, group=None, async_op=False):
            _log("all_gather_into_tensor", group, input.numel(), output.numel(),
                 f"dtype={input.dtype} async={async_op}")
            return _orig["all_gather_into_tensor"](output, input, group=group, async_op=async_op)
        dist.all_gather_into_tensor = _agit

    # --- reduce_scatter_tensor ---
    if hasattr(dist, "reduce_scatter_tensor"):
        _orig["reduce_scatter_tensor"] = dist.reduce_scatter_tensor
        def _rst(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
            _log("reduce_scatter_tensor", group, input.numel(), output.numel(),
                 f"dtype={input.dtype} op={op} async={async_op}")
            return _orig["reduce_scatter_tensor"](output, input, op=op, group=group, async_op=async_op)
        dist.reduce_scatter_tensor = _rst

    # --- all_reduce ---
    _orig["all_reduce"] = dist.all_reduce
    def _ar(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        _log("all_reduce", group, tensor.numel(), tensor.numel(),
             f"dtype={tensor.dtype} op={op} async={async_op}")
        return _orig["all_reduce"](tensor, op=op, group=group, async_op=async_op)
    dist.all_reduce = _ar

    # --- all_gather (list form, used by older code paths) ---
    _orig["all_gather"] = dist.all_gather
    def _ag(tensor_list, tensor, group=None, async_op=False):
        _log("all_gather", group, tensor.numel(),
             sum(t.numel() for t in tensor_list),
             f"dtype={tensor.dtype} async={async_op}")
        return _orig["all_gather"](tensor_list, tensor, group=group, async_op=async_op)
    dist.all_gather = _ag

    # --- broadcast ---
    _orig["broadcast"] = dist.broadcast
    def _bc(tensor, src, group=None, async_op=False):
        _log("broadcast", group, tensor.numel(), tensor.numel(),
             f"src={src} dtype={tensor.dtype} async={async_op}")
        return _orig["broadcast"](tensor, src, group=group, async_op=async_op)
    dist.broadcast = _bc

    # --- batch_isend_irecv (used by RingComm / CP P2P) ---
    _orig["batch_isend_irecv"] = dist.batch_isend_irecv
    def _bii(p2p_op_list):
        total = sum(op.tensor.numel() for op in p2p_op_list)
        desc = ",".join(
            f"{op.op.__name__}(->{op.peer},n={op.tensor.numel()})"
            for op in p2p_op_list
        )
        # P2P ops are per-op, so group is pulled from the first op.
        grp = p2p_op_list[0].group if p2p_op_list else None
        _log("batch_isend_irecv", grp, total, total, f"n_ops={len(p2p_op_list)} {desc}")
        return _orig["batch_isend_irecv"](p2p_op_list)
    dist.batch_isend_irecv = _bii

    # --- reduce (pointwise, used by state-passing in some modes) ---
    _orig["reduce"] = dist.reduce
    def _rd(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        _log("reduce", group, tensor.numel(), tensor.numel(),
             f"dst={dst} dtype={tensor.dtype} op={op} async={async_op}")
        return _orig["reduce"](tensor, dst, op=op, group=group, async_op=async_op)
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
