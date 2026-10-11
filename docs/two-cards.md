# Two cards and ~64 GB of RAM: hot/cold experts (patch 0040) and the `tp2-hotcold` engine

The 4-card recipe keeps every routed expert in VRAM and the INT8 PLE table on NVMe. On 2× Arc Pro B70 (2 × 32 GB) the
routed experts alone are 58 GiB, so they cannot all live in VRAM; with 61 GiB of host RAM they cannot all live pinned in
RAM next to everything else either. `tp2-hotcold` splits them.

## Weights

`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` @ `ef554143`: Intel's AutoRound target tensors (the quantization block
equals `engines/serve-config-intel-autoround.json`) plus RadixArk's FP8 PLE table (`model-plefp8-*`, 128 shards, one
global scale) plus an INT4-g32 MTP draft (`runtime/mtp-int4-g32`, compressed-tensors). Three one-time steps:

```bash
# INT8 PLE table for 0008/0013, built from the FP8 shards (~3.5 min, 52.5 GB on NVMe)
CKPT=/models/qwen38-w4a16 OUT_DIR=/models/qwen38-w4a16-int8ple python3 tools/build_int8_ple_from_fp8.py build
CKPT=/models/qwen38-w4a16 OUT_DIR=/models/qwen38-w4a16-int8ple python3 tools/build_int8_ple_from_fp8.py verify
# serve snapshots: the target, and the MTP draft with its own quantization block
tools/fp8ple_snapshot.py /models/qwen38-w4a16 engines/serve-config-intel-autoround.json /models/qwen38-w4a16-snapshot
tools/fp8ple_snapshot.py --keep-quant /models/qwen38-w4a16/runtime/mtp-int4-g32 \
    engines/serve-config-intel-autoround.json /models/qwen38-w4a16-mtp-snapshot
```

`verify` on 262,045 sampled rows: every row re-quantises byte-exact; relative L2 error against the FP8 rows 0.66 %
mean, 0.90 % p99 (the same size as INT8 against BF16 in docs/ple-int8.md).

## Patch 0040: hot experts in VRAM, cold experts pinned in host RAM

The MoE runs TP 2 + EP 2: with the experts TP-sharded, `moe_intermediate_size` 640 / 2 = 320 is not a multiple of the
quantization group (128), so each rank owns 256 whole experts instead.

1. **Placement.** Per layer and rank, the `B70_MOE_HOT_EXPERTS` most-routed local experts (by
   `B70_MOE_HOTCOLD_RANKINGS`; the checkpoint ships `runtime/static_hot_cache_rankings.json`; experts it never saw
   follow by id) are the layer's ordinary parameters in VRAM. The rest are cold.
2. **Compute.** The XPU int4 grouped GEMM reads cold weights straight from pinned host memory through a UVA view (zero
   copy over PCIe). Each layer runs it twice, hot and cold, each with an `expert_map` that maps only its own experts
   (-1 elsewhere, the mechanism EP already uses), and sums the outputs. A row's result does not depend on where its
   experts live: hot + cold equals all-in-VRAM bit for bit (tests/kernels/moe/test_b70_hotcold_xpu.py).
3. **Loading.** `create_weights` gets the hot count; the weight loader writes cold experts into CPU twins of the same
   parameters; after loading, each layer's twins are repacked into the kernel layout (transpose, unsigned to signed
   int4) on the device, 16 experts at a time, and copied into pinned memory carved from 1 GiB chunks. torch's pinned allocator rounds every block up to a
   power of two (a 197 MB cold w13 would take 256 MB); exact 1 GiB chunks are not rounded.
4. **MTP layers are never split**; the draft stays in VRAM.

Cold-path cost, measured in isolation on one B70 (150 hot / 106 cold experts, random routing): 0.27 ms per layer at
1 row, ~1.9 ms at 7 rows, 12–20 ms for a 1,024–4,096-row prefill chunk (~13–19 GB/s over PCIe). Prefill therefore
prefers large chunks.

One trap, fixed in 0040 as published: vLLM builds the model under a default-device context, so the CPU twins must be
created with an explicit `device="cpu"`. Without it they landed in VRAM: torch reserved 35.6 GiB on a 30.3 GiB card
during loading (the driver overcommits), and the fragmented remainder cost ~3 GiB of KV cache per card.
`B70_MOE_HOTCOLD_MEMLOG=1` logs torch allocated / reserved and device free memory after every layer's split.

## The engine (`engines/tp2-hotcold.env`, `engines/serve-tp2.args`, `engines/run-tp2.sh`)

Container: `image/Dockerfile.tp2` (the official v0.30.0 XPU image by digest + kernels 0.1.14.1+b70.3); the source tree
and wu1ff's residency shim come from the work volume. Settings (the stable starting point, 2026-10-11): TP 2 + EP 2,
136 hot experts per layer and rank, MTP k=3 with the checkpoint's draft, `--gpu-memory-utilization 0.92`, 65,536-token
context, image input off (`--limit-mm-per-prompt` 0),
4 slots, 4,096-token batches, INT8 PLE from NVMe with a 2 GiB row cache, no CPU KV tier. It listens on 0.0.0.0:8000.

```bash
docker build -f image/Dockerfile.tp2 -t b70-flash-next:tp2 <dir holding the kernel wheel>
MODELS=~/models WORK=~/b70-work engines/run-tp2.sh
```

Launcher notes: `-w /opt/venv` (the image's `/workspace/vllm` checkout would shadow `PYTHONPATH` for `python -m`
subprocesses), `-v /dev/dri/by-path:/dev/dri/by-path:ro` (oneCCL opens it; `--device /dev/dri` does not create it).
`run-tp2.sh` sends one short warm-up request once `/health` answers (known issues item 13: with MTP, a first request
longer than one prefill chunk stops the engine). Variant measured on 2026-10-10: 112 hot experts, 0.95, 262,144-token
context and `--language-model-only` gave 489,214 KV tokens per card and 52–56 tok/s decode; 0.95 passes vLLM's startup
check here because each card has 29.2–29.4 GiB free at start (0.95 × 30.3 = 28.8 GiB).

## Measured (2× B70, 61 GiB RAM, compute-runtime as installed, 2026-10-10/11; single runs, ±10 % noise)

| | |
|---|---|
| VRAM per card | weights ~22.6 GiB (136 hot experts × 48 layers, dense, MTP draft); **144,945 KV tokens** (2.2 × a 65,536-token request) |
| host RAM | cold experts 15 GiB pinned per rank; PLE row cache 1 GiB per rank; ~45 GiB used in all, ~16 GiB available |
| decode, 1 stream | 57–58 tok/s with MTP (26–36 without MTP) |
| decode, 4 streams | 24 tok/s each, 97 total |
| prefill | ~2,600 tok/s at 3K–12K tokens; long first requests (17K with thinking, 24K) pass after the warm-up; on the 262K variant a 146K-token needle was found in 82 s |
| MTP acceptance | 2.48 tokens per step on a real chat answer (0.82 / 0.69 / 0.43 per position on synthetic text) |
| boot | ~4–5 min |

Not yet measured: quality benchmarks against the 4-card engine, a multi-day stability run, more than 4 slots.

## How to reproduce the measurements

Engine: `engines/restart-tp2.sh` (detached start; wait for `[run-tp2] warm-up request OK` in `$WORK/engine.log`).
Client scripts in `tools/bench_tp2/`, run in any container on the host network (e.g. the dev container, with this
repository mounted):

```bash
python3 tools/bench_tp2/bench2.py 1:20:512 2:20:256 4:20:256 4:20:512          # decode, 1/2/4 streams
python3 tools/bench_tp2/bench2.py 1:2800:64 1:11000:64 1:28000:64 1:55000:64 1:28000:256   # prefill, long-context decode
python3 tools/bench_tp2/bench_mtp.py                                           # real prompts, MTP acceptance
python3 tools/bench_tp2/repro.py 1050 1 600                                    # known issue 13 (must print OK)
```

`bench2.py` starts each cell only when the engine is idle and flags cells that other clients overlapped (it reads
the engine log named by `B70_ENGINE_LOG`); `bench.py` is the same without the gate. Measured on 2026-10-11, idle-gated:
decode 64.6 tok/s at 1 stream (512 tokens), 87.5 total at 2, 114.3 total at 4; prefill 2,648 / 2,679 / 2,596 / 2,443
tok/s at 3K / 12K / 30K / 59K tokens, decode after them 63 / 62 / 44-50 / 31 tok/s; real prompts 41-56 tok/s with MTP
acceptance 2.62 tokens per step (0.74 / 0.51 / 0.37 per draft position).

## How to test it

All commands run in a container from `b70-flash-next:tp2` (or the dev container) with the source tree on `PYTHONPATH`.

1. **The patch series.** `scripts/check-series.sh --git-dir <tree> --ref b70/tp2-hotcold` (patches/ equals the branch),
   `scripts/make-tree.sh b70/tp2-hotcold <dir> --from-patches` (applies on v0.30.0 and checks every touched file's
   sha256 in image/verify-overlay.sh).
2. **0040 on the CPU.** `pytest tests/model_executor/test_b70_hotcold.py`: ranking per EP rank, unranked experts,
   slot maps, the switch off by default, MTP layers not split, the loader routing cold experts into CPU twins.
3. **0040 on a card.** `ZE_AFFINITY_MASK=0 pytest tests/kernels/moe/test_b70_hotcold_xpu.py`: hot (VRAM) + cold (UVA
   views of the pinned arena) equals all experts in VRAM, bit for bit, at 1 / 7 / 64 / 1,024 rows. Needs a free card.
4. **The PLE table.** `tools/build_int8_ple_from_fp8.py verify`: byte-exact re-quantisation of 262,045 sampled rows,
   error against the FP8 source (0.66 % mean).
5. **The engine.** Boot `engines/run-tp2.sh`; check the boot lines `B70 hot/cold: 48 layers split`, `PLE NVMe host
   path ready`, `XPU KV cache size`; then a chat request, a `json_schema` request with MTP, a long-context needle, and
   a speed run (decode at 1 and 4 streams, prefill at 3K and 12K tokens). The numbers above came from that sequence.

Not done yet, and worth doing before relying on it:

- **Quality** against the 4-card engine on the same items (MMLU, TruthfulQA, the agentic coding set, 64K document
  questions). The target tensors are Intel's, but the PLE table is RadixArk's FP8 re-quantised to INT8, not devan's
  BF16-derived table, and the MTP draft is a different (INT4-g32) one.
- **Endurance**: a multi-day run under real load, watching host MemAvailable (13–16 GiB headroom) and preemptions.
- **Concurrency**: more than 4 slots, and the breakpoint where 4,096-token chunks of long prompts slow decode down.

## What to improve

1. **Measure, then set the hot set from this host's traffic.** The rankings come from the checkpoint author's runs.
   Counting routes per (layer, expert) at run time and re-ranking would raise the share of routes served from VRAM.
   A route counter would also show how much time decode spends on the cold path.
2. **Bigger prefill chunks.** Cold experts cost the same PCIe read per chunk whatever its size, so 8,192-token chunks
   should prefill faster (the CUDA 2× 3090 runtime saw that); it costs activation memory and KV.
3. **Overlap the cold path.** Hot and cold run one after the other on one queue; issuing the cold GEMM on a second
   queue would hide part of the PCIe time behind the hot one.
4. **CPU compute for cold experts at decode** (ktransformers style): host DRAM reads at ~60 GB/s against ~19 GB/s
   over PCIe, so a CPU int4 kernel could beat UVA reads at small batch sizes.
5. **Read the FP8 PLE shards in place** instead of a 52.5 GB INT8 copy: the 0013 reader would need multi-file row
   offsets and an FP8 dequant path (0007 has one).
6. **Put B70-0040 on the fork's release branch** so `make-tree.sh` for a release includes it.
