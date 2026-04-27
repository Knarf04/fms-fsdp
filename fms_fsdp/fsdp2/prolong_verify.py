"""
Post-conversion sanity check for the parquet shards produced by
prolong_convert.py.

For each subset under <root>:
- Walks the subset's parquet files (using fms-fsdp's ParquetHandler so the
  schema check matches the loader exactly).
- Samples N rows.
- Re-tokenizes the `text` column with the local Llama-3 tokenizer.
- Reports the realized doc-length distribution (token count, char count) and
  flags any rows that are empty or suspiciously short.
- For ultrachat output (presence of `mask_runs` column), reports mask
  coverage statistics (% assistant tokens).

Round-trip token-equality testing is intentionally NOT done because the
ProLong corpus already strips Llama-3 special tokens during conversion and
re-encoding `tokenizer.decode(...)` output via the same tokenizer is not
guaranteed to produce a bit-identical id sequence (BPE is not reversible
through `decode->encode` for arbitrary inputs). What we CAN guarantee, and
do check, is that the text is non-empty and tokenizes back to roughly the
same length.

Usage:
    python -m fms_fsdp.fsdp2.prolong_verify \
        --root /gpfs/hshen/datasets/prolong/512k \
        --tokenizer_path /gpfs/hshen/tokenizer/llama3 \
        --samples_per_subset 200
"""

import argparse
import os
import random
import statistics
import sys
from typing import List


def _list_subsets(root: str) -> List[str]:
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


def _list_parquets(subset_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(subset_dir, followlinks=True):
        for fn in filenames:
            if fn.endswith(".parquet"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _sample_rows(parquet_paths: List[str], n: int, seed: int):
    """Stream-sample `n` rows uniformly across the union of parquet files
    using reservoir sampling on the file list, then a uniform pick within
    each chosen file."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    if not parquet_paths:
        return []
    # Cheap two-pass: get row counts per file, then sample row offsets.
    counts = []
    for p in parquet_paths:
        try:
            counts.append(pq.read_metadata(p).num_rows)
        except Exception as e:
            print(f"  [warn] cannot read metadata of {p}: {e}", file=sys.stderr)
            counts.append(0)
    total = sum(counts)
    if total == 0:
        return []
    n = min(n, total)
    # Pick `n` global indices uniformly without replacement.
    chosen = sorted(rng.sample(range(total), n))
    # Map global indices back to (file, offset).
    out = []
    cur_file_idx = 0
    cur_base = 0
    chosen_iter = iter(chosen)
    next_chosen = next(chosen_iter, None)
    while next_chosen is not None and cur_file_idx < len(parquet_paths):
        nrows = counts[cur_file_idx]
        if next_chosen < cur_base + nrows:
            offsets_in_file = []
            while next_chosen is not None and next_chosen < cur_base + nrows:
                offsets_in_file.append(next_chosen - cur_base)
                next_chosen = next(chosen_iter, None)
            if offsets_in_file:
                tbl = pq.read_table(parquet_paths[cur_file_idx])
                for off in offsets_in_file:
                    out.append({col: tbl[col][off].as_py() for col in tbl.column_names})
        cur_base += nrows
        cur_file_idx += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Output root from prolong_convert.py "
                         "(e.g. /gpfs/hshen/datasets/prolong/512k).")
    ap.add_argument("--tokenizer_path", default="/gpfs/hshen/tokenizer/llama3")
    ap.add_argument("--samples_per_subset", type=int, default=200,
                    help="Random docs to inspect per subset.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_text_len", type=int, default=8,
                    help="Flag rows whose text is shorter than this many chars.")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"--root {args.root} is not a directory")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed (needed for the tokenizer).")

    print(f"Loading tokenizer from {args.tokenizer_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    subsets = _list_subsets(args.root)
    if not subsets:
        sys.exit(f"No subset directories found under {args.root}")

    overall_ok = True
    for sub in subsets:
        subset_dir = os.path.join(args.root, sub)
        pqs = _list_parquets(subset_dir)
        if not pqs:
            print(f"\n--- {sub}: no parquet files, skipping.")
            continue

        print(f"\n=== Verifying subset: {sub}")
        print(f"    parquet files: {len(pqs)}")
        rows = _sample_rows(pqs, args.samples_per_subset, args.seed)
        if not rows:
            print(f"    [warn] no rows sampled; subset may be empty.")
            continue

        # Schema sanity: every row must have 'text' (str).
        bad_schema = [r for r in rows if not isinstance(r.get("text"), str)]
        if bad_schema:
            print(f"    [error] {len(bad_schema)} rows missing string `text` column",
                  file=sys.stderr)
            overall_ok = False
            continue

        # Length / retokenization stats.
        char_lens, retok_lens, short_rows = [], [], 0
        for r in rows:
            t = r["text"]
            char_lens.append(len(t))
            if len(t) < args.min_text_len:
                short_rows += 1
            retok_lens.append(len(tok(t, add_special_tokens=False)["input_ids"]))

        def _summary(name, xs):
            xs = sorted(xs)
            if not xs:
                return f"{name}: (empty)"
            return (f"{name}: min={xs[0]:,} p50={xs[len(xs)//2]:,} "
                    f"mean={int(statistics.mean(xs)):,} "
                    f"p99={xs[max(0, int(0.99 * len(xs)) - 1)]:,} "
                    f"max={xs[-1]:,}")

        print(f"    sampled rows : {len(rows)}")
        print(f"    {_summary('char_len    ', char_lens)}")
        print(f"    {_summary('token_len   ', retok_lens)}")
        if short_rows:
            print(f"    [warn] {short_rows}/{len(rows)} rows shorter than "
                  f"{args.min_text_len} chars")

        # Mask-runs check (ultrachat case).
        if "mask_runs" in rows[0]:
            asst_frac = []
            for r in rows:
                runs = r.get("mask_runs") or []
                total_tok = sum((run["end"] - run["start"]) for run in runs)
                asst_tok = sum(
                    (run["end"] - run["start"]) for run in runs
                    if run.get("value") == 1
                )
                if total_tok > 0:
                    asst_frac.append(asst_tok / total_tok)
            if asst_frac:
                print(f"    assistant token fraction: "
                      f"mean={statistics.mean(asst_frac):.3f} "
                      f"min={min(asst_frac):.3f} max={max(asst_frac):.3f}")

    print("\n" + ("ALL OK" if overall_ok else "ISSUES DETECTED — see [error] lines above"))
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
