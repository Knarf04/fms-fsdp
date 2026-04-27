"""
Convenience wrapper around `huggingface_hub.snapshot_download` for the
ProLong long-context training corpora.

These repos have thousands of small MDS shard files. Naive parallel
download trips two distinct failure modes:
  - 429 (rate limit; HF caps free at 1000 req/5min)
  - 5xx (HF-side transients on the xet-token endpoint)
plus generic network flakiness on long-running multi-hour transfers.

This wrapper resumes through all of them. snapshot_download is idempotent
(it re-uses the local file cache), so each retry just picks up where the
last attempt stopped.

Retry policy:
  - 429 / connection / timeout / generic 4xx : exp backoff starting at
    `--retry_initial_sleep` (default 30s), capped at 600s.
  - 5xx (HF backend hiccup)                  : longer backoff starting at
    `--retry_5xx_initial_sleep` (default 60s), capped at 1800s (30 min).
  - Wall-clock retry budget governed by `--retry_budget_minutes`
    (default 480 = 8 hours). Process exits non-zero only if the budget
    is exhausted.

Usage:
    HF_TOKEN=hf_xxx python -m fms_fsdp.fsdp2.prolong_download \
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
    ap.add_argument("--max_workers", type=int, default=4,
                    help="Concurrent download threads inside huggingface_hub. "
                         "Keep low (~4) to stay under HF's 1000-req/5min anon "
                         "rate limit; raise if you set HF_TOKEN.")
    ap.add_argument("--allow_patterns", default="",
                    help="Optional comma-separated glob patterns to restrict "
                         "which files are downloaded (e.g. 'arxiv/*').")
    ap.add_argument("--max_retries", type=int, default=10_000,
                    help="Hard ceiling on retry count. Effectively unlimited "
                         "by default; the real limit is --retry_budget_minutes.")
    ap.add_argument("--retry_initial_sleep", type=float, default=30.0,
                    help="Seconds to sleep before the first retry on 429 / "
                         "connection / 4xx errors; doubles each retry up to 600s.")
    ap.add_argument("--retry_5xx_initial_sleep", type=float, default=60.0,
                    help="Seconds to sleep before the first retry on 5xx "
                         "(HF backend) errors; doubles each retry up to 1800s.")
    ap.add_argument("--retry_budget_minutes", type=float, default=480.0,
                    help="Wall-clock retry budget per repo (minutes). Aborts "
                         "the script only if no successful snapshot completes "
                         "within this window.")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        sys.exit("huggingface_hub is required. Install with: "
                 "pip install huggingface_hub")
    # Catch network/connection issues alongside HF-specific HTTP errors.
    try:
        import requests.exceptions as _req_exc
        _NET_EXC = (
            _req_exc.ConnectionError,
            _req_exc.Timeout,
            _req_exc.ChunkedEncodingError,
            _req_exc.ReadTimeout,
        )
    except ImportError:
        _NET_EXC = ()

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

        sleep_4xx = args.retry_initial_sleep
        sleep_5xx = args.retry_5xx_initial_sleep
        attempt = 0
        budget_deadline = t0 + args.retry_budget_minutes * 60.0
        while True:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    max_workers=args.max_workers,
                    allow_patterns=allow_patterns,
                )
                break
            except (HfHubHTTPError,) + _NET_EXC as e:
                attempt += 1
                if attempt > args.max_retries:
                    print(f"  [error] hit --max_retries={args.max_retries}; aborting.",
                          file=sys.stderr, flush=True)
                    raise
                if time.time() >= budget_deadline:
                    print(f"  [error] retry budget "
                          f"({args.retry_budget_minutes:.0f} min) exhausted; aborting.",
                          file=sys.stderr, flush=True)
                    raise

                status = None
                if isinstance(e, HfHubHTTPError):
                    status = getattr(getattr(e, "response", None), "status_code", None)
                is_5xx = status is not None and 500 <= status < 600

                # Pick base wait. 5xx gets a longer floor & ceiling.
                if is_5xx:
                    wait = sleep_5xx
                    cap = 1800.0
                else:
                    wait = sleep_4xx
                    cap = 600.0
                # Honor Retry-After if HF set one (only present on HTTP errors).
                if isinstance(e, HfHubHTTPError):
                    try:
                        ra = e.response.headers.get("Retry-After")
                        if ra:
                            wait = max(wait, float(ra))
                    except Exception:
                        pass
                wait = min(wait, cap)

                # Don't sleep past the budget deadline.
                remaining = budget_deadline - time.time()
                wait = max(1.0, min(wait, remaining))

                kind = (f"HTTP {status}" if status is not None
                        else type(e).__name__)
                print(
                    f"  [warn] {kind} (attempt {attempt}). "
                    f"Sleeping {wait:.0f}s, then resuming snapshot. "
                    f"Budget left: {(budget_deadline - time.time()) / 60:.1f} min.",
                    file=sys.stderr, flush=True,
                )
                time.sleep(wait)
                # Grow the appropriate backoff for the next round.
                if is_5xx:
                    sleep_5xx = min(sleep_5xx * 2, cap)
                else:
                    sleep_4xx = min(sleep_4xx * 2, cap)
        print(f"=== Done {short_name} in {time.time() - t0:,.0f}s "
              f"(retries used: {attempt})", flush=True)


if __name__ == "__main__":
    main()
