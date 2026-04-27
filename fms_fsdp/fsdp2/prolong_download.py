"""
Convenience wrapper around `huggingface_hub.snapshot_download` for the
ProLong long-context training corpora.

Usage:
    python -m fms_fsdp.fsdp2.prolong_download \
        --cache_dir /gpfs/hshen/cache \
        --datasets prolong-data-64K,prolong-data-512K,prolong-ultrachat-64K
"""

import argparse
import os
import sys
import time


_PROLONG_REPOS = {
    "prolong-data-64K":      "princeton-nlp/prolong-data-64K",
    "prolong-data-512K":     "princeton-nlp/prolong-data-512K",
    "prolong-ultrachat-64K": "princeton-nlp/prolong-ultrachat-64K",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="/gpfs/hshen/cache",
                    help="Local directory to mirror the HF snapshots into. "
                         "Each repo lands at <cache_dir>/<repo_short_name>/.")
    ap.add_argument("--datasets",
                    default=",".join(_PROLONG_REPOS.keys()),
                    help="Comma-separated short names: " + ",".join(_PROLONG_REPOS))
    ap.add_argument("--max_workers", type=int, default=16,
                    help="Concurrent download threads inside huggingface_hub.")
    ap.add_argument("--allow_patterns", default="",
                    help="Optional comma-separated glob patterns to restrict "
                         "which files are downloaded (e.g. 'tulu-v2/*').")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required. Install with: "
                 "pip install huggingface_hub")

    os.makedirs(args.cache_dir, exist_ok=True)
    short_names = [s.strip() for s in args.datasets.split(",") if s.strip()]

    allow_patterns = None
    if args.allow_patterns:
        allow_patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    for short_name in short_names:
        if short_name not in _PROLONG_REPOS:
            sys.exit(f"Unknown ProLong dataset {short_name!r}. "
                     f"Known: {list(_PROLONG_REPOS)}")
        repo_id = _PROLONG_REPOS[short_name]
        local_dir = os.path.join(args.cache_dir, short_name)
        print(f"\n=== Downloading {repo_id} → {local_dir}", flush=True)
        t0 = time.time()
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            max_workers=args.max_workers,
            allow_patterns=allow_patterns,
        )
        print(f"=== Done {short_name} in {time.time() - t0:,.0f}s", flush=True)


if __name__ == "__main__":
    main()
