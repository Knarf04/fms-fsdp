"""
Convert Princeton-NLP ProLong MDS shards into fms-fsdp-compatible parquet.

ProLong layout:
- one row in MDS = one packed sequence (input_ids: int32[L])
- `indices`  = list of (start, end) tuples marking original document
  boundaries inside the packed window
- ultrachat additionally carries `mask: uint8[L]` and `length: uint64`

This converter:
- discovers MDS shard files for each subset under <local_mds>/<subset>/
- unpacks each row's `indices` into individual documents
- strips Llama-3 special tokens (128000-128255) from doc start/end
  (matches fms-fsdp ParquetHandler/ArrowHandler strip_tokens semantics)
- detokenizes with the local Llama-3 tokenizer (skip_special_tokens=True)
- writes parquet shards under <out_root>/<subset>/shard-NNNNN-PP.parquet
  with a `text` column (and optional `mask_runs` for ultrachat)
- runs one MDS shard per worker process; defaults to os.cpu_count()
  workers with TOKENIZERS_PARALLELISM=false to avoid thread oversubscription

Usage:
    python -m fms_fsdp.fsdp2.prolong_convert \
        --hf_dataset princeton-nlp/prolong-data-512K \
        --local_mds  /gpfs/hshen/cache/prolong-data-512K \
        --out_root   /gpfs/hshen/datasets/prolong/512k \
        --tokenizer_path /gpfs/hshen/tokenizer/llama3
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import List

# Llama-3 reserves ids 128000-128255 for special tokens.
LLAMA3_SPECIAL_LO = 128000
LLAMA3_SPECIAL_HI = 128255


# ---- per-worker globals (initialized in _init_worker) ---------------------

_TOKENIZER = None
_INCLUDE_MASK = False
_SHARD_TARGET_BYTES = 1_500_000_000


def _init_worker(tokenizer_path: str, include_mask: bool, shard_target_bytes: int):
    """Pool initializer. Builds the Llama-3 tokenizer once per process and
    forces the Rust backend to single-threaded mode so that 72 processes do
    not collectively spawn hundreds of OS threads."""
    global _TOKENIZER, _INCLUDE_MASK, _SHARD_TARGET_BYTES
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    _INCLUDE_MASK = include_mask
    _SHARD_TARGET_BYTES = shard_target_bytes


# ---- MDS reading ----------------------------------------------------------

def _list_subsets(local_mds: str) -> List[str]:
    """A 'subset' is any subdirectory of local_mds that contains an
    `index.json` (the MDS manifest)."""
    subs = []
    for entry in sorted(os.listdir(local_mds)):
        path = os.path.join(local_mds, entry)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "index.json")):
            subs.append(entry)
        else:
            # Some ProLong repos nest one level deeper (split / config).
            for entry2 in sorted(os.listdir(path)):
                inner = os.path.join(path, entry2)
                if os.path.isdir(inner) and os.path.isfile(os.path.join(inner, "index.json")):
                    subs.append(os.path.join(entry, entry2))
    return subs


def _list_mds_shards(subset_dir: str) -> List[str]:
    """Read index.json and return absolute paths of the .mds files in order."""
    idx_path = os.path.join(subset_dir, "index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    shards = idx.get("shards", [])
    files = []
    for s in shards:
        # Each shard may carry "raw_data": {"basename": ...} or "zip_data".
        for key in ("raw_data", "zip_data"):
            if key in s and "basename" in s[key]:
                files.append(os.path.join(subset_dir, s[key]["basename"]))
                break
    return files


# ---- per-shard worker -----------------------------------------------------

def _strip_specials(ids: list) -> list:
    """Drop Llama-3 special tokens from the start and end of a doc, mirroring
    fms-fsdp's strip_tokens logic for BOS/EOS markers in pretokenized data."""
    i, j = 0, len(ids)
    while i < j and LLAMA3_SPECIAL_LO <= ids[i] <= LLAMA3_SPECIAL_HI:
        i += 1
    while j > i and LLAMA3_SPECIAL_LO <= ids[j - 1] <= LLAMA3_SPECIAL_HI:
        j -= 1
    return ids[i:j], i, j


def _rle_mask(mask_slice) -> list:
    """Run-length-encode a 0/1 mask into [{start, end, value}, ...].
    `mask_slice` is a numpy array, list, or pyarrow array of ints."""
    out = []
    if len(mask_slice) == 0:
        return out
    # Convert to plain ints once
    ms = [int(x) for x in mask_slice]
    cur_val = ms[0]
    cur_start = 0
    for k in range(1, len(ms)):
        if ms[k] != cur_val:
            out.append({"start": cur_start, "end": k, "value": int(cur_val)})
            cur_start = k
            cur_val = ms[k]
    out.append({"start": cur_start, "end": len(ms), "value": int(cur_val)})
    return out


def _flush_parquet(rows, out_path, include_mask):
    import pyarrow as pa
    import pyarrow.parquet as pq

    text_col = pa.array([r["text"] for r in rows], type=pa.string())
    columns = {"text": text_col}
    if include_mask:
        runs_struct = pa.struct([
            pa.field("start", pa.int32()),
            pa.field("end", pa.int32()),
            pa.field("value", pa.int8()),
        ])
        columns["mask_runs"] = pa.array(
            [r.get("mask_runs", []) for r in rows],
            type=pa.list_(runs_struct),
        )
    table = pa.table(columns)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        row_group_size=1024,
    )
    size = os.path.getsize(out_path)
    return size


def _process_one_shard(task: dict) -> dict:
    """Worker entry point: convert a single MDS shard's rows into one or more
    parquet files. Returns a summary dict for the parent process to aggregate."""
    global _TOKENIZER, _INCLUDE_MASK, _SHARD_TARGET_BYTES

    subset_dir = task["subset_dir"]
    shard_index = task["shard_index"]
    out_subset_dir = task["out_subset_dir"]
    max_docs = task["max_docs"]

    try:
        # Public API; falls back to the internal path for older versions.
        try:
            from streaming import LocalDataset
        except ImportError:
            from streaming.base.local import LocalDataset
    except ImportError as e:
        return {"error": f"mosaicml-streaming not installed: {e}",
                "shard_index": shard_index, "subset_dir": subset_dir}

    try:
        ds = LocalDataset(local=subset_dir)
    except Exception as e:
        return {"error": f"failed to open MDS at {subset_dir}: {e}",
                "shard_index": shard_index, "subset_dir": subset_dir}

    # Discover the [start_row, end_row) interval for this shard from index.json.
    idx_path = os.path.join(subset_dir, "index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    shards = idx.get("shards", [])
    if shard_index >= len(shards):
        return {"error": f"shard_index {shard_index} oob ({len(shards)} shards)",
                "shard_index": shard_index, "subset_dir": subset_dir}
    shard_info = shards[shard_index]
    samples_per_shard = shard_info.get("samples", 0)
    # MDS index entries store cumulative sample offsets; sum prior shards.
    row_start = sum(s.get("samples", 0) for s in shards[:shard_index])
    row_end = row_start + samples_per_shard

    rows_buffer = []
    buffer_text_bytes = 0
    docs_emitted = 0
    total_tokens = 0
    lengths = []
    part_idx = 0
    output_paths = []

    def flush(rows_buffer, part_idx):
        if not rows_buffer:
            return None, 0
        out_path = os.path.join(
            out_subset_dir,
            f"shard-{shard_index:05d}-{part_idx:03d}.parquet",
        )
        size = _flush_parquet(rows_buffer, out_path, _INCLUDE_MASK)
        return out_path, size

    try:
        for row_i in range(row_start, row_end):
            try:
                row = ds[row_i]
            except Exception as e:
                # corrupt sample; skip
                continue

            input_ids = row.get("input_ids")
            indices = row.get("indices")
            mask = row.get("mask") if _INCLUDE_MASK else None
            if input_ids is None or indices is None:
                continue

            # Convert numpy/torch/pa arrays to a python list once.
            try:
                ids_list = input_ids.tolist()
            except AttributeError:
                ids_list = list(input_ids)
            mask_list = None
            if mask is not None:
                try:
                    mask_list = mask.tolist()
                except AttributeError:
                    mask_list = list(mask)

            for span in indices:
                # span is typically (start, end); some MDS variants serialize
                # as a 2-element numpy array.
                try:
                    start = int(span[0])
                    end = int(span[1])
                except Exception:
                    continue
                if end <= start:
                    continue

                doc_ids = ids_list[start:end]
                doc_ids_clean, lo_off, hi_off = _strip_specials(doc_ids)
                if not doc_ids_clean:
                    continue

                try:
                    text = _TOKENIZER.decode(
                        doc_ids_clean, skip_special_tokens=True
                    )
                except Exception:
                    # Decoding failure: skip this doc rather than fail the
                    # whole shard.
                    continue

                row_out = {"text": text}
                if _INCLUDE_MASK and mask_list is not None:
                    sub_mask = mask_list[start + lo_off : start + hi_off]
                    row_out["mask_runs"] = _rle_mask(sub_mask)

                rows_buffer.append(row_out)
                docs_emitted += 1
                total_tokens += len(doc_ids_clean)
                lengths.append(len(doc_ids_clean))
                buffer_text_bytes += len(text)

                if buffer_text_bytes >= _SHARD_TARGET_BYTES:
                    out_path, size = flush(rows_buffer, part_idx)
                    if out_path:
                        output_paths.append((out_path, size))
                    rows_buffer = []
                    buffer_text_bytes = 0
                    part_idx += 1

                if max_docs and docs_emitted >= max_docs:
                    break

            if max_docs and docs_emitted >= max_docs:
                break

        out_path, size = flush(rows_buffer, part_idx)
        if out_path:
            output_paths.append((out_path, size))

    except Exception as e:
        return {
            "error": f"unhandled: {e}\n{traceback.format_exc()}",
            "shard_index": shard_index,
            "subset_dir": subset_dir,
            "docs_emitted": docs_emitted,
            "lengths": lengths,
            "output_paths": output_paths,
        }

    return {
        "shard_index": shard_index,
        "subset_dir": subset_dir,
        "docs_emitted": docs_emitted,
        "total_tokens": total_tokens,
        "lengths": lengths,
        "output_paths": output_paths,
    }


# ---- driver ---------------------------------------------------------------

def _summarize_subset(name, results, t0, n_shards):
    docs = sum(r.get("docs_emitted", 0) for r in results)
    toks = sum(r.get("total_tokens", 0) for r in results)
    files = sum(len(r.get("output_paths", [])) for r in results)
    bytes_out = sum(sz for r in results for _, sz in r.get("output_paths", []))
    lengths = [L for r in results for L in r.get("lengths", [])]
    elapsed = time.time() - t0

    print(f"\n=== Subset summary: {name}")
    print(f"    shards processed : {n_shards}")
    print(f"    docs emitted     : {docs:,}")
    print(f"    total tokens     : {toks:,}")
    print(f"    parquet files    : {files} ({bytes_out / 1e9:,.2f} GB)")
    print(f"    elapsed          : {elapsed:,.1f}s ({docs / max(elapsed, 1e-9):,.0f} docs/s)")
    if lengths:
        sl = sorted(lengths)
        def pct(p): return sl[min(int(p / 100 * len(sl)), len(sl) - 1)]
        print(
            f"    doclen tokens    : "
            f"min={sl[0]:,}  p50={pct(50):,}  mean={sum(sl) // len(sl):,}  "
            f"p99={pct(99):,}  max={sl[-1]:,}"
        )
    return {
        "docs": docs, "tokens": toks, "files": files, "bytes_out": bytes_out,
        "n_shards": n_shards, "elapsed_sec": elapsed,
    }


def _str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_dataset", default="princeton-nlp/prolong-data-512K",
                    help="HF repo id (used only for documentation in summaries).")
    ap.add_argument("--local_mds", default="",
                    help="Local cache root containing the MDS subset folders. "
                         "If empty, defaults to /gpfs/hshen/cache/<basename(hf_dataset)>.")
    ap.add_argument("--out_root", required=True,
                    help="Output root, e.g. /gpfs/hshen/datasets/prolong/512k")
    ap.add_argument("--tokenizer_path", default="/gpfs/hshen/tokenizer/llama3")
    ap.add_argument("--subsets", default="all",
                    help="'all' or comma-separated subset names "
                         "(relative to --local_mds).")
    ap.add_argument("--shard_target_bytes", type=int, default=1_500_000_000,
                    help="Target uncompressed text-bytes per output parquet shard "
                         "(actual file size will be smaller post-zstd).")
    ap.add_argument("--num_workers", type=int, default=os.cpu_count() or 8,
                    help="Parallel workers (one MDS shard each). "
                         "Default = os.cpu_count() to maximize node utilization.")
    ap.add_argument("--include_mask", type=_str2bool, default=False,
                    help="Set true for ultrachat to write a `mask_runs` column.")
    ap.add_argument("--max_docs_per_subset", type=int, default=0,
                    help="0 = all; useful for dry runs.")
    ap.add_argument("--dry_run", type=_str2bool, default=False,
                    help="List the work plan and exit without converting.")
    ap.add_argument("--resume", type=_str2bool, default=True,
                    help="Skip subsets that already have a _DONE marker.")
    ap.add_argument("--summary_json", default="",
                    help="Optional path to write per-subset summary JSON.")
    args = ap.parse_args()

    local_mds = args.local_mds or os.path.join(
        "/gpfs/hshen/cache", os.path.basename(args.hf_dataset.rstrip("/"))
    )
    if not os.path.isdir(local_mds):
        sys.exit(f"local_mds not found: {local_mds}\n"
                 "Run prolong_download.py first or pass --local_mds.")

    requested_subsets = None if args.subsets.strip().lower() == "all" else [
        s.strip() for s in args.subsets.split(",") if s.strip()
    ]

    discovered = _list_subsets(local_mds)
    if requested_subsets is not None:
        unknown = [s for s in requested_subsets if s not in discovered]
        if unknown:
            sys.exit(f"Unknown subset(s) {unknown}. Discovered: {discovered}")
        subsets = requested_subsets
    else:
        subsets = discovered

    if not subsets:
        sys.exit(f"No MDS subsets discovered under {local_mds}")

    print(f"Source            : {args.hf_dataset}  ({local_mds})")
    print(f"Output root       : {args.out_root}")
    print(f"Tokenizer         : {args.tokenizer_path}")
    print(f"Subsets ({len(subsets)})  : {subsets}")
    print(f"Workers           : {args.num_workers}")
    print(f"Shard target bytes: {args.shard_target_bytes:,}")
    print(f"Include mask      : {args.include_mask}")
    if args.dry_run:
        for sub in subsets:
            d = os.path.join(local_mds, sub)
            shards = _list_mds_shards(d)
            print(f"  {sub}: {len(shards)} MDS shard(s)")
        return

    os.makedirs(args.out_root, exist_ok=True)
    overall_summary = {}

    # Use 'fork' on Linux so the tokenizer is reloaded fresh per worker
    # (initializer takes care of it); 'spawn' fallback on Windows/mac.
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")

    for sub in subsets:
        subset_dir = os.path.join(local_mds, sub)
        out_subset_dir = os.path.join(args.out_root, sub)
        done_marker = os.path.join(out_subset_dir, "_DONE")
        if args.resume and os.path.exists(done_marker):
            print(f"\n--- Skipping {sub}: _DONE marker present.")
            continue

        os.makedirs(out_subset_dir, exist_ok=True)
        shard_files = _list_mds_shards(subset_dir)
        n_shards = len(shard_files)
        if n_shards == 0:
            print(f"\n--- {sub}: no shard files in index.json, skipping.")
            continue

        # One task per MDS shard; budget max_docs across shards equally.
        per_shard_max = 0
        if args.max_docs_per_subset > 0:
            per_shard_max = max(1, math.ceil(args.max_docs_per_subset / n_shards))

        tasks = [
            {
                "subset_dir": subset_dir,
                "shard_index": i,
                "out_subset_dir": out_subset_dir,
                "max_docs": per_shard_max,
            }
            for i in range(n_shards)
        ]

        print(f"\n### Subset {sub}  ({n_shards} MDS shards) ###")
        t0 = time.time()
        results = []
        with ctx.Pool(
            processes=min(args.num_workers, n_shards) or 1,
            initializer=_init_worker,
            initargs=(args.tokenizer_path, args.include_mask, args.shard_target_bytes),
        ) as pool:
            for r in pool.imap_unordered(_process_one_shard, tasks, chunksize=1):
                if "error" in r:
                    print(f"  [warn] shard {r.get('shard_index')}: {r['error']}",
                          file=sys.stderr, flush=True)
                else:
                    n_done = sum(1 for x in results if "error" not in x) + 1
                    print(
                        f"  shard {n_done}/{n_shards} done "
                        f"(idx={r['shard_index']}, +{r['docs_emitted']:,} docs, "
                        f"+{r['total_tokens']:,} toks)",
                        flush=True,
                    )
                results.append(r)

        summary = _summarize_subset(sub, results, t0, n_shards)
        overall_summary[sub] = summary

        # Write _DONE marker on success (no errors at all).
        if all("error" not in r for r in results):
            with open(done_marker, "w") as f:
                json.dump({
                    "hf_dataset": args.hf_dataset,
                    "local_mds": subset_dir,
                    "tokenizer_path": args.tokenizer_path,
                    "include_mask": args.include_mask,
                    "n_shards_in": n_shards,
                    "n_files_out": summary["files"],
                    "docs": summary["docs"],
                    "tokens": summary["tokens"],
                    "bytes_out": summary["bytes_out"],
                    "elapsed_sec": summary["elapsed_sec"],
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }, f, indent=2)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"\nSummary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
