# ProLong dataset conversion utilities

Three modules under `fms_fsdp/fsdp2/` convert Princeton-NLP's
[ProLong](https://huggingface.co/princeton-nlp) long-context corpora from
their distributed MDS (Mosaic Data Shard) format into parquet shards that the
existing fms-fsdp dataloader (`ParquetHandler` / `AutoHandler`) can read with
no pipeline changes.

| Module | Purpose |
|---|---|
| `prolong_download.py` | Download MDS shards from HF with rate-limit-aware retries |
| `prolong_convert.py`  | Unpack + detokenize MDS rows → fms-fsdp parquet shards |
| `prolong_verify.py`   | Sanity-check the converted parquet (schema, lengths, retokenize) |

All three are run as Python modules from the fms-fsdp repo root, e.g.:

```bash
python -m fms_fsdp.fsdp2.prolong_download …
python -m fms_fsdp.fsdp2.prolong_convert  …
python -m fms_fsdp.fsdp2.prolong_verify   …
```

## Why this exists

ProLong stores ~31 B-token long-context corpora as **packed** sequences (one
MDS row = one 64 K- or 512 K-token window with multiple original documents
concatenated, document boundaries marked by an `indices` field). The
fms-fsdp loader expects **one row = one document**, so we have to:

1. read MDS via `mosaicml-streaming`,
2. split each packed row back into its constituent documents using `indices`,
3. drop Llama-3 special tokens (128000–128255) at doc edges (matches the
   loader's `strip_tokens` semantics),
4. detokenize each doc with the local Llama-3 tokenizer (`skip_special_tokens=True`),
5. write parquet shards with a `text` column under a directory layout that
   mirrors ProLong's subset structure.

For background on why the upstream cosmopedia-v2 setup was inadequate (no
docs > 8 K, training a 512 K-context model on essentially short content),
see the original conversion plan at
`~/.claude/plans/given-the-crucial-state-mellow-music.md`.

## One-time install (remote node)

```bash
pip install mosaicml-streaming pyarrow huggingface_hub
# transformers + the Llama-3 tokenizer at /gpfs/hshen/tokenizer/llama3
# are already present on the training node.
```

## Step 1 — download the corpora

```bash
# Authenticate (PRO recommended; ProLong-64K alone is 3608 files).
huggingface-cli login    # paste a Read-scope token from huggingface.co/settings/tokens
huggingface-cli whoami   # confirm "[Pro User]" badge

# Download all three repos with retry-on-429-and-5xx baked in.
HF_TOKEN=$HF_TOKEN python -m fms_fsdp.fsdp2.prolong_download \
    --cache_dir /gpfs/hshen/cache \
    --datasets prolong-data-64K,prolong-data-512K,prolong-ultrachat-64K \
    --max_workers 8
```

Notable flags:

| Flag | Default | Purpose |
|---|---|---|
| `--max_workers` | 4 | Parallel HF downloads. Keep ≤ 8 even on PRO |
| `--max_retries` | 10 000 | Retry ceiling (effectively unlimited) |
| `--retry_initial_sleep` | 30s | Backoff base for 429 / connection / 4xx errors (caps at 600s) |
| `--retry_5xx_initial_sleep` | 60s | Backoff base for 5xx (caps at 1800s = 30 min) |
| `--retry_budget_minutes` | 480 | Wall-clock budget per repo (8 hours). Aborts only on exhaustion |
| `--allow_patterns` | (none) | Glob-restrict files, e.g. `'arxiv/*'` |

Re-running with the same args is safe and cheap — `snapshot_download` skips
files already present locally. The download is idempotent.

After it finishes, expect **~62 GB per long-context repo** in
`/gpfs/hshen/cache/<repo-name>/`.

## Step 2 — sanity-check what subsets exist

ProLong's dataset cards show display names that don't always match the actual
MDS subset directories. Discover the real names with `--dry_run true`:

```bash
python -m fms_fsdp.fsdp2.prolong_convert \
    --hf_dataset princeton-nlp/prolong-data-64K \
    --out_root   /gpfs/hshen/tmp/prolong_smoke \
    --dry_run    true
```

Real subset names you'll likely see:

- **prolong-data-64K** (11): `arxiv, book-65536, dclm-baseline, dolmawiki,
  fineweb-2023-50, fineweb-edu, openwebmath, stackexchange, textbooks,
  thestackv1_concat_by_repo-65536, tuluv2`
- **prolong-data-512K** (11+): same set plus 524 288-windowed variants of code/books
- **prolong-ultrachat-64K**: a single subset (chat data)

Use the discovery output as ground truth — the `--subsets` flag below must
match exactly.

## Step 3 — smoke-test on one small subset

```bash
mkdir -p /gpfs/hshen/tmp/prolong_smoke

python -m fms_fsdp.fsdp2.prolong_convert \
    --hf_dataset princeton-nlp/prolong-data-64K \
    --out_root   /gpfs/hshen/tmp/prolong_smoke \
    --subsets    tuluv2 \
    --max_docs_per_subset 5000 \
    --num_workers 8

python -m fms_fsdp.fsdp2.prolong_verify \
    --root /gpfs/hshen/tmp/prolong_smoke \
    --tokenizer_path /gpfs/hshen/tokenizer/llama3
```

Expected verify output: `ALL OK`, char-length and re-tokenized-length
distributions roughly within ±1 % of each other, and the per-subset
char/token mean ratio around 3-5 (English-text BPE territory).

**Note**: smoke-test parquet files are typically < 1 MB each because
`--max_docs_per_subset` truncates before the per-shard byte target is hit.
fms-fsdp's loader silently filters files < 1 MB
([dataset_utils.py:1209](../utils/dataset_utils.py#L1209)), so smoke output
is for QA only, not for training.

## Step 4 — full conversion

Default `--num_workers = os.cpu_count()` (= 72 on the training node) and
`--shard_target_bytes = 1.5 GB` keep output shards comfortably between
fms-fsdp's 1 MB minimum and 5 GB maximum guidance.

```bash
# Pretraining corpora (no mask).
python -m fms_fsdp.fsdp2.prolong_convert \
    --hf_dataset princeton-nlp/prolong-data-64K \
    --out_root   /gpfs/hshen/datasets/prolong/64k

python -m fms_fsdp.fsdp2.prolong_convert \
    --hf_dataset princeton-nlp/prolong-data-512K \
    --out_root   /gpfs/hshen/datasets/prolong/512k

# Chat SFT corpus — preserve the assistant-turn mask in a `mask_runs` column.
python -m fms_fsdp.fsdp2.prolong_convert \
    --hf_dataset princeton-nlp/prolong-ultrachat-64K \
    --out_root   /gpfs/hshen/datasets/prolong/ultrachat-64k \
    --include_mask true
```

Notable flags:

| Flag | Default | Purpose |
|---|---|---|
| `--num_workers` | `os.cpu_count()` | One MDS shard per worker. 72 on the target node |
| `--shard_target_bytes` | 1.5 GB | Approx. text bytes per output parquet (zstd-compressed file is smaller) |
| `--include_mask` | false | true for ultrachat → adds a `mask_runs` column (RLE) |
| `--max_docs_per_subset` | 0 | Cap per subset; 0 = all |
| `--resume` | true | Skip subsets with a `_DONE` marker file from a prior run |
| `--subsets` | "all" | Or comma-separated names matching MDS dirs |
| `--dry_run` | false | Print discovery and shard counts, then exit |

Each completed subset writes `<out_root>/<subset>/_DONE` with conversion
metadata; re-running the script picks up where it left off.

Per-subset throughput is roughly 100 docs/s/core, so ProLong-64K's ~39 M
docs take **~10 min wall-clock** on a 72-core node. ProLong-512K is roughly
3× slower per doc due to longer decode lengths (~30 min). Ultrachat is
small (~1.2 GB, finishes in seconds).

## Step 5 — point training at the converted data

The output layout:

```
/gpfs/hshen/datasets/prolong/64k/
├── arxiv/                             shard-NNNNN-PP.parquet, …
├── book-65536/
├── dclm-baseline/
├── dolmawiki/
├── fineweb-2023-50/
├── fineweb-edu/
├── openwebmath/
├── stackexchange/
├── textbooks/
├── thestackv1_concat_by_repo-65536/
└── tuluv2/
```

The existing training CLI works unchanged — only `--data_path` and
`--datasets` change:

```bash
torchrun … main_training_mamba_fsdp2.py \
    --data_path /gpfs/hshen/datasets/prolong/64k \
    --datasets  arxiv,book-65536,dclm-baseline,dolmawiki,fineweb-2023-50,fineweb-edu,openwebmath,stackexchange,textbooks,thestackv1_concat_by_repo-65536,tuluv2 \
    --weights   <ProLong recipe weights, see ProLong/training/configs> \
    --file_type auto \
    --col_name  text,content,contents,tokens \
    --tokenizer_path /gpfs/hshen/tokenizer/llama3 \
    --bos_token None \
    --eos_token 128000 \
    --target_doclen 8192 \
    …
```

Key compatibility points:

- `AutoHandler` ([dataset_utils.py:437-492](../utils/dataset_utils.py#L437-L492))
  routes `.parquet` → `ParquetHandler` automatically.
- `ParquetHandler` finds the `text` column (first match in
  `--col_name`'s default lookup order) and tokenizes at read time using
  the Llama-3 tokenizer at `--tokenizer_path` — same tokenizer ProLong
  used originally, so token IDs match what was on disk in MDS.
- The `mask_runs` column on ultrachat is **ignored** by today's
  `ParquetHandler`. The schema is preserved for forward-compat if we later
  thread mask-aware loss masking through the loader. For now, training on
  ultrachat = training on the whole conversation (user + assistant turns).
- `target_doclen` should match the dataset shape. ProLong docs are very
  long, so `target_doclen=8192` (instead of the previous `2048`) better
  reflects what we want the loader's probabilistic length filter to keep.
  See [the cosmopedia-v2 length analysis](#why-this-exists) for why this
  matters.

## Step 6 — distribution sanity check

After conversion, re-run the existing length-stats tool against the output
to confirm the long-tail is present (this is what motivated the conversion
in the first place — cosmopedia-v2 had no docs > 8 K):

```bash
python experiments/dataset_length_stats.py \
    --data_path /gpfs/hshen/datasets/prolong/64k \
    --datasets  arxiv,book-65536,dclm-baseline,dolmawiki,fineweb-2023-50,fineweb-edu,openwebmath,stackexchange,textbooks,thestackv1_concat_by_repo-65536,tuluv2 \
    --tokenizer_path /gpfs/hshen/tokenizer/llama3 \
    --num_workers 32 \
    --plot_dir /gpfs/hshen/plots/dataset_stats/prolong-64k
```

You should see meaningful mass above 8 K, 32 K, and 64 K depending on the
subset (arxiv, book-65536, fineweb-edu skew long; tuluv2, dclm-baseline are
shorter chat/web).

## Troubleshooting

- **Smoke-test parquet not loading at training time**: shard files < 1 MB
  are silently dropped by the loader. Re-run with no `--max_docs_per_subset`
  so the byte-target shard sizing kicks in.
- **`Unknown subset(s) [...]`**: the dataset cards lie about subset names.
  Run `--dry_run true` and copy the discovered names into your `--subsets`
  flag.
- **Repeated 429s during download**: confirm `huggingface-cli whoami` shows
  `[Pro User]`. If yes, lower `--max_workers` to 4. If no, upgrade to PRO
  (single $9 month is enough for one-shot data prep).
- **`mosaicml-streaming` import errors in the worker**: install on the
  remote node only; conversion-time dep, not a training-time dep.
- **Resuming a partial conversion**: `--resume true` (default) skips
  subsets that already have `_DONE`. To force re-conversion of a subset,
  delete its `_DONE` marker and the corresponding output dir.
