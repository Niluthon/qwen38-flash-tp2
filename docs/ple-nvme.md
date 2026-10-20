# The PLE table served from NVMe (patches 0013, 0013b, 0050)

With 0008 the INT8 PLE table still pins 48.9 GiB of host RAM. 0013 leaves it on the NVMe and keeps only a small pinned
row cache, which frees that RAM for a larger CPU KV tier (or makes the model fit a 128 GB unified-memory box at all).

## Design

1. **A host hook in `execute_model`, before the graph replay, on real batches only** (not dummy or profiling runs). It
   hashes the batch's n-grams on the host (bit-identical to the device function; a boot self-test checks it), resolves
   the row ids against a per-rank pinned row cache, reads the misses from the existing INT8 `.safetensors` with
   `O_DIRECT` (a row is one 4 KiB read, 8 KiB for the ~4 % that cross a page), and launches the existing gather into
   the static prefetch buffer. **No host I/O runs inside a captured region.**
2. **Each TP rank serves only its own rows** (on 4 ranks: two ranks hold bigram heads, two trigram). Nothing is shared
   across processes.
3. **Its own row cache**, not the page cache: pinned slots, a row→slot map, CLOCK eviction that protects the current
   and the previous step. Slot offsets are int64.
4. **0013b** adds a native reader (a C thread pool, one call per batch, GIL released; compiled with `gcc` at first use)
   and an optional one-chunk prefill lookahead read by a background thread per rank.

Switches: `B70_PLE_INT8_NVME*` in docs/switches.md. It needs `B70_PLE_INT8=1` and a table built by
`tools/build_int8_ple.py`.

## Measured (4× Arc Pro B70, TP4 + EP, native reader, 2 GiB pinned row cache per rank, 64 GiB CPU KV tier)

| | INT8 table in RAM | INT8 table on NVMe |
|---|---|---|
| host MemAvailable at idle | 62 GiB | 101 GiB (**~39 GiB freed**) |
| cold prefill 8K / 32K / 64K | 1.74 / 7.42 / 16.47 s | 1.75 / 7.41 / 16.32 s |
| decode, aggregate at 1 / 4 / 8 streams | 57.6 / 182 / 285 tok/s | 55.5 / 177 / 279 tok/s (−2 … −4 %) |
| fidelity vs BF16 reference (KL / top-1) | 0.0163 / 96.35 % | 0.0171 / 96.06 % (at the INT8 noise floor) |

The decode delta is exactly the cost of the per-step host sync the hook needs: a diagnostic mode that keeps the table
in RAM but adds the sync (`B70_PLE_INT8_NVME_SYNC_ONLY=1`) measured the same −2 … −5 %, and 0 on prefill. The NVMe read
path itself adds nothing measurable. Under real agent traffic the row cache hit 97–98 %, with ~2–5 misses per step and
a host bubble of ~0.5 ms p50 / ~1 ms p99. Under a mixed load of heavy and light sessions, light-session inter-token
latency was equal (~30 ms p50); the heavy sessions' TTFT tail was longer (p90 12.0 vs 9.0 s).

Off the GPU: host hash equal to the device function on 13.7 M ids; rows read through the cache byte-equal to the file
on 8,177 rows over 4 ranks including page-crossing rows; per-step host path 0.33 / 0.41 / 0.51 ms at 1 / 4 / 8
sequences; a 1,024-token prefill chunk on a trigram rank reads in 63 ms (py), 31 ms (uring), 23 ms (native).

## Notes

1. Keep the NVMe to this table while measuring; a disk KV tier on the same device competes for the same I/O.
2. The native reader is compiled at first use into `B70_PLE_INT8_NVME_NATIVE_DIR` (default `$TMPDIR/b70-ple-nvme`);
   `TMPDIR` must be writable.
3. A 128 GB unified-memory machine could not hold the BF16 table at all and would spend ~38 % of its memory on the
   INT8 one; this path is what makes the model practical there. Untested on such hardware.

## The checkpoint's FP8 shards in place (patch 0050)

0013 serves a table that has to be built first: on the two-card engine that is `tools/build_int8_ple_from_fp8.py`,
which reads the checkpoint's FP8 shards and writes a 52.5 GB INT8 copy (~3.5 min, docs/two-cards.md "Weights").
0050 skips the copy and reads the FP8 shards themselves, through the same host hook, row cache and readers.

### What it reads

The checkpoint (`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`) ships the PLE table as 128 tensors
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<i>.weight`, each F8_E4M3
[2,500,012, 160], spread over 10 files (`model-plefp8-0000{0..9}.safetensors`), plus one global BF16
`weight_scale` and the three I64 layout tensors (in `model-00001-of-00017.safetensors`).

| | INT8 table (0008 / 0013) | FP8 shards in place (0050) |
|---|---|---|
| file(s) on NVMe | 1 extra file, 52.5 GB, built at prepare time | none extra: the checkpoint's 10 shard files |
| bytes per row | 164 (160 × int8 + fp32 scale) | 160 (one global scale, applied after the gather) |
| rows in a 1 GiB-per-rank cache | 6,547,206 | 6,710,886 (+2.5 %) |
| data start | 4096-byte aligned (built that way) | arbitrary per tensor (unaligned safetensors headers) |
| values | INT8 re-quantised from the FP8 rows: rel. L2 error 0.66 % mean, 0.90 % p99 vs FP8 | the FP8 rows themselves (0 vs the checkpoint) |
| dequantisation on the card | `q × s` per row (0008) | 256-entry LUT, code × global scale (0007), bit-equal to the stock cast-multiply |

### Design

1. **Multi-file row store.** `PleNvmeRowStore(..., segments=[(file, data_start, first_row, rows), ...])`: one
   segment per shard tensor, covering [0, 320,001,536) in order without gaps; each file is opened once
   (`O_DIRECT`). Row g's file and byte offset come from a binary search over the segments' first rows. Reads are the
   same page-aligned 4 KiB (8 KiB for rows that cross a page) as before; unaligned data starts need nothing extra
   because every read is aligned down to its page.
2. **Per-read file descriptors in the native readers.** The C pool and the io_uring reader take an optional fd per
   read (`NULL` = 0013's single-file path, unchanged), so one batch still goes out in one call with the GIL
   released, whatever files its rows live in. A failed or short read takes the same Python retry path, on the row's
   own file.
3. **Same hook, same cache, same gather.** The 0013 host hook, CLOCK row cache, lookahead and stats are unchanged
   and sized by the storage row width (160 instead of 164). FP8 rows are gathered and all-reduced as bytes (no float8
   kernel on the XPU, as 0007) and dequantised by 0007's LUT after the all-reduce.
4. **`B70_PLE_FP8_PATH` takes the checkpoint directory** (or its `model.safetensors.index.json`): only the
   `…ple_embedding.…` tensors are mapped, and they must share one prefix. A single `.safetensors` file still works
   as in 0007, for the pinned FP8 path too.

### Boot checks (before any pinned memory is allocated)

- The shards resolve as in 0007: 128 shards 0..127, each F8_E4M3 [2,500,012, 160], one finite positive scale,
  layout tensors equal to the model's; anything else refuses to start.
- **Reader bytes = mmap bytes** on 4,096 random rows of the whole table plus the first and last row of every shard
  (catches wrong offsets, a wrong file, page-crossing bugs).
- **No NaN codes** (E4M3fn `0x7F` / `0xFF`) in a random sample of this rank's rows (`B70_PLE_INT8_NVME_BOOT_SAMPLE`,
  default 65,536). E4M3fn has no infinities.
- With `PLE_TABLE_PATH` set, 0007's BF16 cross-check runs as before.

### Use

The two-card engine's default since round 10 (`engines/two-cards/env/tp2-hotcold.env`); the patch-level default is
off. The lines:

```
B70_PLE_INT8=0
B70_PLE_INT8_NVME=0
B70_PLE_FP8=1
B70_PLE_FP8_NVME=1
B70_PLE_FP8_PATH=/models/qwen38-w4a16
```

Every `B70_PLE_INT8_NVME_*` tuning switch (cache size, I/O threads, reader, stats, lookahead, boot sample,
sync-only) applies unchanged; docs/switches.md. Boot log lines to look for: `PLE FP8 NVMe load (B70_PLE_FP8_NVME=1)
starting: …, 128 shards in 10 files`, `PLE NVMe host path ready`, `Serving FP8 PLE table from NVMe in place`.

### Verified so far (2026-10-20, host side only)

| check | result |
|---|---|
| `tests/b70/test_ple_fp8_nvme.py` (synthetic checkpoint: 7 shards in 3 files with unaligned data starts, plus the real checkpoint) | 21 / 21 pass |
| real checkpoint, readers `native` / `uring` / `py`: 2,000 random rows + both ends of all 128 shards vs buffered `pread` | byte-equal, no NaN codes |
| real checkpoint, end to end: 1,500 random FP8 rows read in place and re-quantised as `build_int8_ple_from_fp8.py` does vs the INT8 table 0013 serves today | byte-equal (same rows, same order) |
| 0013 / 0013b regression suite (`tests/b70/test_ple_int8_nvme.py`, real INT8 table, all readers) | 22 / 22 pass |
| full series on v0.30.0 (`scripts/make-tree.sh … --from-patches`) | 44 patches apply with `--fuzz=0`; 40 / 40 file hashes match |

Read latency per batch on this host's 980 PRO, native reader, 16 threads, `O_DIRECT`, random rows of one rank's half
(milliseconds per `read_rows` call, p50 of 30 calls, 5 for 8,192 rows; the engine was serving one request meanwhile):

| rows per call | FP8 shards in place, ms | INT8 table, ms |
|---|---|---|
| 8 (a decode step's misses) | 0.21 | 0.21 |
| 64 | 0.54–0.59 | 0.54–0.56 |
| 8,192 (a 1,024-token prefill chunk on one rank) | 50.3–50.6 | 50.7 |

### Measured on the engine (2026-10-20, docs/two-cards.md round 10)

Same boot sequence and load for both tables, dev tree with 0050, idle engine. Step times are bench_det ms/step for
the four fixed prompts; perplexity is the mean of two passes over four fixed texts.

| | INT8 table from NVMe | FP8 shards in place |
|---|---|---|
| 1 stream, ms/step | 34.7 / 31.8 / 32.2 / 32.1 | 34.7 / 31.7 / 31.8 / 32.1 |
| 4 streams, tok/s total · ms/step | 178.3 · 61.1 | 184.8 · 61.4 |
| prefill 3K / 12K / 30K, tok/s | 4,086 / 4,135 / 3,979 | 4,085 / 4,151 / 3,983 |
| row cache hit rate (whole run) | 0.905 | 0.902 |
| per-step host bubble p50 / p99, ms | 0.51 / 8.88 | 0.49 / 8.90 |
| perplexity vs INT8, % (4 texts) | — | +0.19 / −0.53 / −0.43 / +0.43 |

Equal within noise. The rebuilt image with 0050 baked in and these lines as defaults (no overrides): 1 stream
36.0 / 32.6 / 33.6 / 33.4 ms/step (second run 36.8 / 31.3 / 37.8 / 31.6), 4 streams 176.8–177.8 tok/s, prefill
4,021 / 4,143 / 3,973 tok/s, hit rate 0.911, perplexity within 0.21 % of the INT8 run. To go back to the INT8
table, use the fallback lines in `engines/two-cards/env/tp2-local.env.example` (the table must exist:
`BUILD_INT8_PLE=1 docker compose run --rm prepare`).
