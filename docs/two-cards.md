# Two cards and ~64 GB of RAM: hot/cold experts (patch 0040) and the `tp2-hotcold` engine

The 4-card recipe keeps every routed expert in VRAM and the INT8 PLE table on NVMe. On 2× Arc Pro B70 (2 × 32 GB) the
routed experts alone are 58 GiB, so they cannot all live in VRAM; with 61 GiB of host RAM they cannot all live pinned in
RAM next to everything else either. `tp2-hotcold` splits them.

## Weights

[`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) @ `ef554143`: Intel's AutoRound target tensors (the quantization block
equals `engines/serve-configs/serve-config-intel-autoround.json`) plus RadixArk's FP8 PLE table (`model-plefp8-*`, 128 shards, one
global scale) plus an INT4-g32 MTP draft (`runtime/mtp-int4-g32`, compressed-tensors). This host runs revision
`c6af9fee`. Three one-time steps (`docker compose run --rm prepare` downloads the checkpoint and runs all three):

```bash
# INT8 PLE table for 0008/0013, built from the FP8 shards (~3.5 min, 52.5 GB on NVMe)
CKPT=/models/qwen38-w4a16 OUT_DIR=/models/qwen38-w4a16-int8ple python3 tools/build_int8_ple_from_fp8.py build
CKPT=/models/qwen38-w4a16 OUT_DIR=/models/qwen38-w4a16-int8ple python3 tools/build_int8_ple_from_fp8.py verify
# serve snapshots: the target, and the MTP draft with its own quantization block
tools/fp8ple_snapshot.py /models/qwen38-w4a16 engines/serve-configs/serve-config-intel-autoround.json /models/qwen38-w4a16-snapshot
tools/fp8ple_snapshot.py --keep-quant /models/qwen38-w4a16/runtime/mtp-int4-g32 \
    engines/serve-configs/serve-config-intel-autoround.json /models/qwen38-w4a16-mtp-snapshot
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

## The engine (`engines/two-cards/env/tp2-hotcold.env`, `engines/two-cards/args/serve-tp2.args`, `engines/two-cards/scripts/run-tp2.sh`)

Container: `image/Dockerfile.tp2`: the official v0.30.0 XPU image by digest, the vLLM of release 0.30.0-b70.2 (fork tag
`Lumnus/vllm@v0.30.0-b70.2`, commit `9d79d28d`, checked against `patches/vllm/` 0001–0032), the two-card patches
0040–0049 on top, kernels 0.1.14.1+b70.3 and wu1ff's residency shim; everything fetched and hash-checked at build time. Settings (2026-10-20, after the experiments below): TP 2 + EP 2,
146 hot experts per layer and rank, swapped by what decode reads (0043 + 0047, long-term ranking on the work volume),
the input embedding in pinned host RAM (0044), the 16-bit dense projections as FP8 W8A16 (0045), cold-expert prefetch
for prefill (0042), the fused decode MoE op (0048) with 70 % of the cold experts computed by the CPU pool (0049) and an
FP8 draft head, MTP k=3 with the checkpoint's draft, `--gpu-memory-utilization 0.96`, 262,144-token context, image
input off (`--limit-mm-per-prompt` 0), 4 slots, 4,096-token batches, INT8 PLE from NVMe with a 2 GiB row cache, no CPU
KV tier. It listens on 0.0.0.0:8000. The env switches are in `engines/two-cards/env/tp2-hotcold.env` and
`docs/switches.md`.

```bash
docker compose build                  # the image (image/Dockerfile.tp2)
docker compose run --rm prepare       # once: the weights steps above (scripts/prepare-models-tp2.sh)
docker compose up -d                  # compose.yaml: engine + one-shot warm-up; or: MODELS=~/models engines/two-cards/scripts/run-tp2.sh
cp engines/two-cards/env/tp2-local.env.example engines/two-cards/env/tp2-local.env   # optional: local overrides (B70_* switches, EXTRA_ARGS)
```

Launcher notes: `-w /opt/venv` (the image's `/workspace/vllm` checkout would shadow `PYTHONPATH` for `python -m`
subprocesses), `-v /dev/dri/by-path:/dev/dri/by-path:ro` (oneCCL opens it; `--device /dev/dri` does not create it).
`run-tp2.sh` sends one short warm-up request once `/health` answers (known issues item 13: with MTP, a first request
longer than one prefill chunk stops the engine). Variant measured on 2026-10-10: 112 hot experts, 0.95, 262,144-token
context and `--language-model-only` gave 489,214 KV tokens per card and 52–56 tok/s decode; 0.95 passes vLLM's startup
check here because each card has 29.2–29.4 GiB free at start (0.95 × 30.3 = 28.8 GiB).

## Measured (2× B70, 61 GiB RAM, compute-runtime as installed, 2026-10-10/11; single runs, ±10 % noise)

### Units in the experiment tables

Every speed table below follows one schema. Each header names the unit; cells are bare numbers.

- **1 stream** — `bench_det.py` at temperature 0: four fixed prompts, one at a time. Reported as **tok/s** for the
  prompt (mean over the four, when one number) or as **ms/step** per prompt (four numbers). Lower ms/step is faster.
- **4 streams** — the same four prompts in parallel: **tok/s** is the total across the four streams; where the step
  time is given it is **ms/step** of the whole batch.
- **agents** — `agent_load.py 3 12 2`: 12 coding-agent sessions at **2 concurrent requests** (temperature 0, frozen
  corpus). **tok/s** = all generated tokens over wall time, i.e. the total across the 2 concurrent requests.
- **KV tokens** — the XPU KV cache size from the boot log, per card. Higher = more context per request.
- **decode cold share** — the share of decode routes that hit cold experts (logged by `B70_MOE_ROUTE_COUNT`); lower
  is faster because cold experts come over PCIe.
- **fresh / after agents** — measured right after boot, and again after the two code-agent sessions (the hot set
  adapts to the traffic).
- One boot each, in order; ±5–10 % noise on single runs, ~3 % on ms/step. `bench_det` ms/step is the repeatable
  number; tok/s follows MTP acceptance and can swing.

| what | value |
|---|---|
| VRAM per card | weights ~22.6 GiB (136 hot experts × 48 layers, dense, MTP draft); 150 hot experts, embedding in host RAM, FP8 dense projections, prefetch slots 0.53 GiB; **201,794 KV tokens** at util 0.95 and a 131,072-token context (1.54 × a full request); 193,621 with 144 hot and BF16 dense; 173,610 with 136 hot and the embedding in VRAM; 144,945 at 0.92 / 65,536 without prefetch |
| host RAM | cold experts 15 GiB pinned per rank; PLE row cache 1 GiB per rank; ~45 GiB used in all, ~16 GiB available |
| decode, 1 stream | 57–58 tok/s with MTP (26–36 without MTP) |
| decode, 4 streams | 24 tok/s each, 97 total |
| prefill | ~3,950 tok/s at 3K–30K tokens with 0042 (~2,600 without), one request at a time (prefill with concurrent requests: not measured); long first requests (17K with thinking, 24K) pass after the warm-up; on the 262K variant a 146K-token needle was found in 82 s |
| MTP acceptance | 2.48 tokens per step on a real chat answer (0.82 / 0.69 / 0.43 per position on synthetic text) |
| boot | ~4–5 min |

Not yet measured: quality benchmarks against the 4-card engine, a multi-day stability run, more than 4 slots.

## Experiments, 2026-10-11 (one boot each; `tools/bench_tp2/bench_det.py`, 1 stream = mean of 4 prompts)

`bench_det.py` uses fixed real prompts at temperature 0, so MTP acceptance is the same between runs and the
engine's ms per step can be compared directly (repeatable within ~3 %). The word-salad cells of `bench2.py` swing
±30 % between identical runs at 1 stream (acceptance depends on the random text); do not compare configs with them.

| configuration | KV tokens | 1 stream tok/s (ms/step range) | 4 streams tok/s | prefill tok/s (3K / 12K / 30K prompts) | verdict |
|---|---|---|---|---|---|
| baseline: 136 hot, util 0.92, 64K | 144,854 | 54.8 (45–61) | 80.8 | 2,599 / 2,608 / 2,575 | reference |
| 128K context (else baseline) | 166,171 | real prompts 45.5 (= 64K) | — | 2,544 / 2,592 / 2,560 | no decode cost |
| PLE row cache 6 GiB + lookahead | 144,854 | 54.2 | 74–78 | 2,479 / 2,577 / 2,547 | no gain (cache already 99 % hits when warm) |
| MTP k=2 | 152,724 | 51.4 (40–54; acceptance −15–24 %) | 74 | 2,432 / 2,544 / 2,522 | worse; keep k=3 |
| 8,192-token batches | 76,671 | ~same | 73.5 | 2,497 / 2,774 / 2,952 | +8–16 % prefill for half the KV; no |
| rankings from 40 local requests | 143,029 | 46.0 (cold share 24 → 33 %) | 61 | — | overfit; keep the checkpoint's |
| `--kv-cache-dtype fp8` | 227,086 | 50.5 (acceptance −3–12 %) | 74 | 2,522 / 2,527 / 2,188 | +57 % KV, −5–8 % decode; only when KV-bound |
| 144 hot, util 0.95 | 154,438 | 55.3 (−3–4 % ms/step) | 84.7 | 2,613 / 2,690 / 2,652 | better everywhere |
| 144 hot, util 0.95, prefetch (0042) | 110,808 | 55.5 | 86.6 | 4,026 / 4,030 / 3,866 | prefill +50 %, answers identical |
| **136 hot, util 0.95, prefetch, 128K (the default now)** | **173,610** | **55.0** | **84–86** | **3,886 / 3,949 / 3,811** | needles 6/6 to 97K, mixed load OK |

Other findings:

- **Decode is bound by the cold experts.** With the checkpoint's rankings 24 % of decode routes (6–39 % by layer) hit
  cold experts on the benchmark prompts and 39 % on a Russian/code/JSON mix; ms/step follows that share (24 → 33 %
  cold: +6–14 ms/step). More hot experts help most: cold share on the benchmark prompts at 112 / 136 / 144 / 152 / 168
  hot = 33 / 24 / 22 / 19 / 15 %. A blend of the checkpoint ranking with local counts (`B70_MOE_ROUTE_COUNT`, λ 0.5–1)
  cut the mix's cold share 39 → 27–31 % without hurting the benchmark prompts (offline); it needs days of real traffic.
- **Not the bottleneck:** all-reduce between the two cards is 22 µs for decode-sized tensors (~2–3 ms per step);
  the PLE NVMe reads are 0.1 ms on the 980 PRO (O_DIRECT, idle or under decode), the per-step PLE bubble 0.1 ms p50
  with a warm cache (an occasional ~9 ms step, 1 %).
- **The prefix cache works** (it reuses 16,000 of 18,375 tokens: TTFT 7.9 → 1.15 s, 0.74 s with prefetch); a 0 %
  hit rate in the engine log only means the traffic did not repeat a ≥ 1,600-token prefix (Mamba 'align' blocks).
- VRAM budget per card at 0.92: weights 22.55 GiB, non-torch 1.84, peak activation 1.06, graphs 0.45, KV 2.42; vLLM's
  own suggestion is ~0.8 GiB more KV, which util 0.95 takes. Each card has PCIe 5.0 x8 (the board splits x16).

## Experiments, round 2 (2026-10-11): code prompts at 2 concurrent requests, KV never reduced

Rule for this round: no change may cost KV (173,610 tokens at 128K). Workload: `tools/bench_tp2/agent_load.py`,
coding-agent sessions (tool-use system prompt, a real source file from the vLLM tree, 5 turns: implement, test, fix,
refactor, explain), 12 sessions, **2 concurrent requests**, temperature 0 for A/B runs. Its tok/s = all generated
tokens / wall time of the run, i.e. the total over the 2 concurrent requests (~half of it per request), prefill included.

| item | result | verdict |
|---|---|---|
| **expert swapping (0043)** | 2 concurrent requests (code) 57.2 / 58.9 → **66.4 / 76.3 tok/s** total (+16 / +30 %, two runs; wall 245 / 286 → 175 / 204 s); decode cold share 30 % → 18–21 %; general prompts re-adapt within minutes (4 streams 88.5 vs 86); KV unchanged | **default** |
| hot and cold MoE on two queues at decode | captures and stays bit-identical, but no overlap: 53.9 vs 53.4 ms (4 rows, 48 layers), 198.3 vs 197.8 (16 rows) | not worth it |
| async scheduling (`--async-scheduling`) | KV 173,610 → 146,788; decode no faster (1 stream ~52 vs ~55, 4 streams 83 vs 84–86); 1 repeat-pattern output in 120 burst requests | not worth it |
| a more precise MTP draft | the draft is RTN INT4-g32 from RadixArk's NVFP4 (already 4-bit); no higher-precision MTP here; INT8 experts would cost ~1.9 GB VRAM per card | not available without KV loss |

What bounds decode, measured with the real kernels (`tools/bench_tp2/par_moe.py`, one B70, 136 hot / 120 cold, random
routing): 48 layers of hot experts take 4.9 ms at 4 rows, cold experts 48.4 ms; at 16 rows 11.0 vs 186.7 ms. The
cold reads run at ~21–23 GB/s, the PCIe 5.0 x8 limit, and keep the GPU busy meanwhile, so decode time is the cold
bytes divided by the link speed. The only lever is fewer cold bytes: swapping. An offline replay of 5 s route
snapshots (`tools/bench_tp2/sim_swap.py`) predicted the result: hot set by prefill counts 14 % decode cold share, by
decode counts 11.7 % (costs ~4 % decode to count), oracle per window 2 %.

Prefix reuse in agent sessions was 27–38 % of prompt tokens: with Mamba 'align' caching a GDN state is kept only at
1,600-token boundaries, and an MTP step often jumps over a boundary, so each turn recomputes up to ~1,600 tokens
besides its new text (≤ 0.4 s at ~3,950 tok/s).

## Experiments, round 3 (2026-10-11): free VRAM without touching KV

| item | result | verdict |
|---|---|---|
| **input embedding in pinned host RAM (0044)** | frees 0.59 GiB per card; 142 hot: KV 199,174, 2 concurrent requests (code) 70.6 / 81.2 tok/s total (136 hot: 66.4 / 76.3); 144 hot: KV 193,412, 73.8 / 79.3; needles 6/6, bursts clean | **default (144 hot)** |
| swap policy from an offline replay (0044 defaults) | prior weight 0.05 → 0.01, budget 4 → 8: decode cold share 14.3 → 12.2 % and 17.2 → 15.3 % on two traces (offline); live within noise | default |
| drop the MTP draft's own lm_head / embed_tokens | byte-identical to the target's, but vLLM already shares them and frees the copies: KV and weights unchanged | nothing to gain |
| fine-grained Mamba prefix cache | needs an identified draft KV group (log: none found) and excludes multi-module MTP; a custom version would save ≤ 0.4 s per agent turn | not applicable |
| decode profile | the torch profiler crashes the engine on XPU (known issues item 14); measured instead: lm_head 1.06 ms per call (4 per step), graph replay 1.3 µs per kernel, all-reduce 22 µs | see below |
| MTP draft experts on hot/cold | ~0.66 GiB per card back, but each of the 3 draft passes would then read cold experts (~0.5 ms each); estimated net ~0 | deferred |
| FP8 KV at a constant 174K tokens + ~10 more hot experts | +~5 % from hot experts against −5–8 % decode (acceptance) and −15 % long prefill; estimated wash | deferred |

**8-bit dense layers (0045; opt-in in round 3, the default since round 4).** `B70_DENSE_FP8` quantizes the 16-bit GDN projections (in_proj_qkv,
in_proj_z, out_proj) and attention q/k/v/o at load time to FP8 W8A16. Measured: weights −0.9 GiB per card, KV
193,621 → 251,771 tokens; perplexity on fixed texts (`tools/bench_tp2/ppl.py`) code 2.032–2.037 → 2.048, docs
7.58–7.61 → 7.638, Russian 1.539 → 1.546 (+0.4–0.7 %; run-to-run noise ~0.3 %); needles 6/6; general prompts after
chat 52.6 → 56.2 tok/s at 1 stream, 76.0 → 80.5 at 4 (one run each); the same 12 agent sessions at temperature 0
(`agent_judge.py`): answers 1,182 → 1,422 characters, responses at the 500-token cap 48 → 58 %, no loops. In isolation
`fp8_gemm_w8a16` beats BF16 at 1–4 rows (21 vs 37 µs) and loses at 512 (211 vs 132 µs), yet prefill stayed ~3,950
tok/s. Verdict: a KV increase with a small speed gain and a measurable behaviour shift; off until a real quality
benchmark (MMLU, the coding set) passes — it did in round 4. The agent-session comparison in this paragraph used the
unfrozen corpus and is superseded by round 4. Shared experts and the INT4 W4A16 path (`int4_gemm_w4a16`) are untested.

Where the non-cold ~20–28 ms of a decode step goes (estimates from the measurements above): lm_head ×4 ~4.3 ms,
hot experts ~5 ms, all-reduce ~3 ms, graph replay ~2–4 ms, and the dense attention / GDN projections, which are
BF16 (~3.2 GiB per card read every forward, ~5.4 ms). Next candidates: 8-bit dense weights (XPU has W8A16 FP8 and
INC wNa16 linear kernels; would halve those reads and free ~1.6 GiB per card for ~14 more hot experts; needs offline
quantization and a quality comparison) and a 32K-vocabulary lm_head for the draft passes only (~2.3 ms per step,
outputs unchanged since the target verifies every token).

## Experiments, round 4 (2026-10-11): parallel agents, draft head, quality gate for FP8 dense

From here on the agent load reads a frozen copy of its source files (`AGENT_CORPUS`, default `/work/agent-corpus`).
Before that it picked files from the live vLLM tree by size, so every patch that changed a file's size changed the
sessions: the round-3 FP8 agent comparison (prompt totals 207K vs 229K) was confounded and is superseded here.

| item | result | verdict |
|---|---|---|
| **FP8 dense layers (0045) + 150 hot experts** | quality: MMLU 87.33 → 87.23 % on 3,000 paired questions (29 vs 26 discordant, exact McNemar p = 0.79), HumanEval pass@1 157 → 158 / 164 (`run_eval.py`, `he_exec.py`, `compare.py`); KV 193,621 → 201,794; 2 concurrent requests (code) 84.5 / 87.5 → 90.5 / 98.7 tok/s total (frozen corpus, same prompts); general prompts 53.5 → 57.1 tok/s at 1 stream (steps ~3.5 ms shorter); prefill unchanged; 0 / 120 bad burst outputs | **default** |
| 8 slots (`--max-num-seqs 8`, capture sizes to 32) | KV 193,621 → 178,324; 4 concurrent requests (code) 105.8 → 106.4 tok/s total, 8 concurrent 118.6 → 127.7 (+8 %, wall −13 %); prefix reuse at 8 concurrent 30 → 14 % (KV pressure); one unexplained short answer in 400 eight-wide burst requests | opt-in (tp2-local.env.example) |
| draft-only reduced vocabulary (0046, not committed) | step −2…4 ms, but 5–20 % of real output tokens were outside a 51K-token vocabulary (code-heavy corpus) and acceptance fell 3–11 %: net −3 % (1 stream), −7 % (4 streams); <0.2 % misses needs ~98K of the 124K rows per card (~0.7 ms/step for ~0.5 GiB) | not worth it |
| adaptive draft length | vLLM's adaptive verification supports only DSpark, excludes local-argmax drafting, and variable-length verify batches are the padded-graph case of known issue 6 | not applicable |
| cold-expert prefetch at decode | a UVA-reading kernel does not overlap with compute (two-queue test), and the decode graphs cannot express copies whose source experts are chosen on the GPU each step | blocked |

## Experiments, round 5 (2026-10-12): a fixed pool for cold experts at decode, and 154 hot experts

**Fixed virtual pool (measured, not built).** The idea: per layer, a few VRAM slots at fixed addresses that the
captured decode kernels read through the expert maps (graph-safe), filled by copy-engine DMA from the routes of
earlier steps; experts the pool misses fall back to the UVA path, so results stay exact. To avoid waits inside the
graph, the pool would be double-buffered with a two-step lag (fill for step t+1 during step t from the routes of step
t−1). A per-step route trace (0046, `B70_MOE_ROUTE_TRACE`) of 4,120 decode steps (code prompts, 2 concurrent requests) with swapping off
(`tools/bench_tp2/pool_reuse.py`, `pool_churn.py`):

| pool slots per layer | lag | cold routes served by the pool |
|---|---|---|
| 4 | 1 step | 29–31 % |
| 4 | 2 steps | 26–29 % |
| 8 | 2 steps | 41–43 % |

A step needs ~4 cold experts per layer. Per step, P = 4 avoids 133 MB of stalling UVA reads (~5.8 ms) for 94 MB of
DMA; with swapping on (the default) the leftover cold experts are rarer and the saving shrinks to ~3 ms, against
~1–2 ms for counting routes inside the decode graph plus host work at every step boundary, and ~0.47 GiB of VRAM
(~−40K KV). Net estimate +1–4 %: not worth it. The lag is not the problem (2–3 points); the cold experts left after
0043 seldom repeat from step to step.

**154 hot experts (the alternative use of that VRAM).** KV 201,794 → 170,152; 2 concurrent requests (code) 90.5 / 98.7 → 92.1 /
98.5 tok/s total; general prompts 57.1 → 57.9 tok/s at 1 stream, 81.4 → 84.8 at 4; prefill unchanged; 0 / 120 bad burst
outputs. +1–4 %, mostly within noise, for 31.6K KV tokens: the default stays at 150 (154 is a line in
`engines/two-cards/env/tp2-local.env.example`). Returns from more hot experts have flattened at this point.

## Experiments, round 6 (2026-10-19): rank experts by what decode reads (0047)

The idea: rank experts by use and keep them in the tier their use deserves (VRAM ← pinned RAM). 0043 already swaps,
but it scored experts by prompt (prefill) routes, while prefill batches ≥ 512 tokens read the whole cold set from the
0042 slots anyway: only decode pays for a cold expert. 0047 counts the decode graphs' routes (two kernels per layer:
`remainder` + `index_add_` from a static ones buffer, instead of four), scores by them (prefill at weight 0.1), and
keeps a long-term ranking on the work volume that places the hot set at the next boot.

Setup: 146 hot, 262,144-token context, util 0.96 (this host's settings), KV 279,001 tokens in every run (278,888 in B).
One boot each, in this order; per boot `bench_det.py`, then `agent_load.py 3 12 2` twice (AGENT_TEMP=0, frozen
corpus), then `bench_det.py` again (general prompts after the code sessions). Harness: `engine-data/ab/ab.sh`
(local, not committed).

| run | 1 stream tok/s (mean of the 4 prompts) | 4 streams tok/s | agents tok/s (2 concurrent, total) | 1 stream tok/s after agents | 4 streams tok/s after agents | decode cold share (agents) |
|---|---|---|---|---|---|---|
| A: 0043 as before | 55.3 | 98.4 | 90.5 / 94.1 | 53.8 | 69.9 | — (not counted) |
| B: A + decode counting (scores unchanged) | 55.4 | 100.5 | 89.9 / 91.6 | 52.8 | 71.7 | 8–10 % |
| C: 0047 decode scores, no saved state | 55.7 | 137.5 | 87.5 / 99.5 | 54.2 | 114.5 | 6–8 % |
| **D: C + the state C saved (the default now)** | **60.7** | 110.6 | 86.6 / **100.2** | 55.3 | **113.2** | 6–8 % |
| E: D's settings from the rebuilt image (0047 baked in, no overrides; KV 279,114) | 62.7 | 110.6 | 91.6 / 98.4 | 57.0 | 120.8 | — |

- Counting inside the decode graphs costs nothing measurable now: the four fixed prompts' ms/step, A 49.5 / 59.3 /
  46.9 / 44.9 vs B 49.5 / 59.1 / 47.9 / 45.5.
- With prefill scores the hot set follows the prompts: after the code sessions, general prose decodes with 17–34 %
  cold routes (B) and 4 streams drop to 70 tok/s; with decode scores 12–13 % and 113–115 tok/s.
- In the code sessions themselves: decode cold share 8–10 → 6–8 %, the adapted run 94.1 → 99.5–100.2 tok/s (+6 %). The
  first run of each boot (cold prefix cache, mostly prefill) is within noise (86.6–90.5).
- The saved state makes the first requests after a restart faster: 1 stream fresh 55.3 → 60.7 tok/s (the code prompt
  58.6 → 75.8). It is ranked by this host's traffic, so it also carries its bias (here: code); 0043/0047 keep adapting.
- Swap rounds grow with decode scores (~8,300 swaps per rank in 13 minutes in C; 15–40 ms per round every 5 s,
  ≤ 0.8 % of the time).
- Prefill cold share rises (7 → 9–10 %): expected and harmless with 0042 for batches ≥ 512 tokens.

Not tried: a third tier on NVMe for the coldest experts. The cold set already fits in RAM (~15 GiB per rank, ~16 GiB
still free), NVMe reads (3–7 GB/s) are slower than the UVA path (21–23 GB/s), and the decode graphs cannot fetch from
disk, so an expert on NVMe must never be routed while graphs run. It would free RAM, not add speed.

## Experiments, round 7 (2026-10-19): where decode bandwidth goes, a fused decode MoE op and an FP8 draft head (0048)

Bandwidth per component on one B70 (`tools/bench_tp2/decode_bw.py`, `moe_bw.py`; XPU graphs, random data; the
B70 compresses constant data, which inflates GB/s). A plain 2 GiB read reaches 595 GB/s (98 % of 608).

| component, decode shapes | time per call | rate (or % of 608 GB/s where noted) |
|---|---|---|
| BF16 dense (GDN in_proj/out_proj, attention q/o), 4–16 rows | 30–57 µs per layer | 85–92 % |
| BF16 lm_head shard (124,160 × 2,560) | 1.09 ms | 96 % |
| FP8 W8A16 dense (0045) at 4–16 rows | 24–50 µs | 50–56 % (it saves VRAM, not time) |
| small replicated layers (HC, router, shared expert) | 5–21 µs | 50–77 % (launch-bound) |
| hot experts, stock (2 `XpuFusedMoe` calls per layer), 1 / 4 / 8 / 16 rows | 83 / 155 / 208 / 338 µs per layer | 30 / 59 / 70 / 80 % |
| cold experts over PCIe (UVA), 4 / 8 / 16 rows | 381 / 500 / 982 µs per layer | 20 GB/s — the PCIe limit, not a % of VRAM bandwidth |

Dense layers already run at the memory limit. On each card the losses come from the MoE's fixed costs: each
`XpuFusedMoe` call costs about 56 µs (6 kernels) before it reads a weight byte, and there are two calls per layer.
The cold experts are bound by PCIe; only fewer cold reads help there (0047, more hot slots). TP 2 doubles the
bandwidth across the two cards, but per-card fixed costs, the all-reduces and PCIe do not shrink with it.

0048 adds two things:

- **The fused op** (`kernels/b70-moe-decode`, `B70_MOE_DECODE_OP`) runs hot and cold experts in three kernels and
  reads each expert once per layer. Per layer: 150 → 66 µs at 1 row, 391 → 241 µs at 4 rows, 1030 → 740 µs at 16
  rows (cold share 0.08).
- **The FP8 draft head** (`B70_DRAFT_HEAD_FP8`) cuts each of the three draft passes per step from 1.08 ms to 0.56 ms.

Same harness as round 6; one boot each, in this order, sharing the learned state.

| run | KV tokens | 1 stream tok/s (ms/step range) | 4 streams tok/s | agents tok/s (2 concurrent, total) | 1 stream tok/s after agents | 4 streams tok/s after agents |
|---|---|---|---|---|---|---|
| F: image (0047) | 279,001 | 61.4 (45.5–47.9) | 112.4 | 83.3 / 96.9 | 56.3 | 113.7 |
| G: + op, ≤ 8 rows | 279,001 | 73.1 (37.4–41.3) | 113.2 | 105.8 / 107.2 | 68.8 | 115.9 |
| H: G + FP8 draft head | 271,647 | 76.6 (35.5–38.1) | 112.8 | 110.3 / 116.2 | 70.9 | 113.4 |
| **I: op ≤ 16 rows + FP8 draft head** | 271,421 | **76.6** (35.8–39.4) | **141.8** | **105.9 / 115.4** | 68.4 | **144.3** |

- 1 stream +25 %, 4 streams +26 %, code agents +12–27 %. Acceptance is unchanged (2.2–3.6 per step in every run).
  A sanity prompt at temp 0 answers correctly.
- G and H show no 4-stream gain because the op was capped at 8 rows (4 streams × 4 tokens = 16). The first 16-row
  version was slower than stock (register pressure at 16 rows per pass); 4 rows per pass, with more passes served
  from cache, fixed it.
- Both are on by default in `tp2-hotcold.env` (I's settings). The image compiles the op itself in a builder stage
  (icpx 2026.0 from Intel's apt repository, against the image's torch), so nothing is prepared on the host.

## Experiments, round 8 (2026-10-19): cold reads at decode (item 3) and a CPU spike for cold experts (item 4)

**Where the cold bytes come from.** Measured on this host with the engine stopped (`engine-data/spike4/`, local):

| path | GB/s |
|---|---|
| one card reading pinned host RAM (UVA kernel, copy kernel or copy engine; alone or both cards at once) | 28.7 per card |
| the 0048 op on cold experts only | 25.6–26.3 |
| CPU cores reading RAM (8 or 16 threads) | 64–69 (the cores' fabric limit, not the RAM's) |
| CPU while both cards stream from RAM | 44.5, with the cards still at 27.5 each: ~100 GB/s of RAM in total |

**Item 3, kernel side.** A one-work-group plan kernel now lists the batch's local experts, cold first, so no
work-group is wasted on route slots of other ranks or repeated experts, and 128 "cold worker" sub-groups loop over the
cold experts while the rest of the GPU takes the hot ones (one job each). Without the workers, the cold work-groups
filled most of the GPU's 2,048 thread slots while they waited on PCIe. Per layer at cold share 0.08: 4 rows 241 →
226 µs, 8 rows 397 → 353 µs, 16 rows 740 → 638 µs; hot only, 16 rows: 473 → 340 µs. That is 82–91 % of the time the
cold bytes alone need at 26 GB/s. Worker counts: 64 too few (slower at all sizes), 512 too many (they crowd out the
hot experts); 128 is the default.

| run (one boot each, KV ~271.6K) | 1 stream ms/step (the 4 prompts) | 4 streams tok/s · ms/step | agents tok/s (2 concurrent, total) | 1 stream ms/step after agents | 4 streams tok/s · ms/step after agents |
|---|---|---|---|---|---|
| J: image (0048) | 41.1 / 38.1 / 42.0 / 38.5 | 135.1 · 80.8 | 100.8 / 113.5 | 46.2 / 37.2 / 46.1 / 37.7 | 141.2 · 77.4 |
| K: J + plan + cold workers | 41.1 / 37.6 / 41.9 / 37.7 | 142.0 · 76.4 | 111.6 / 118.4 | 45.9 / 36.5 / 44.3 / 38.3 | 146.4 · 73.7 |

4 streams +4–5 %, 1 stream ~1 % (one cold expert or two per layer leave little to overlap), agents +4–10 %.

**Item 4, spike (measured, not built).** Cold experts are bound by PCIe (26 of 28.7 GB/s per card), while RAM still
has room: the CPU can add ~44 GB/s next to both cards. Two questions decided whether a CPU path is possible:

- *A CPU kernel fast enough.* AVX-512 BF16 (`vpermw` nibble → bf16 lookup, `vdpbf16ps` on (even, odd) pairs, one
  scale FMA per 128-group), same weight layout as the GPU, persistent spin pool, 8 threads: 58–64 GB/s of expert
  weights for 1–4 rows per expert, 31–33 GB/s at 8 rows, 17 GB/s at 16 (compute-bound). At decode a cold expert is
  routed by 1–2 rows, so it runs memory-bound. Relative error vs fp64 ~1e-7.
- *A GPU ↔ CPU handshake inside a captured decode graph.* A device counter makes each replay's request unique. The
  GPU writes it to a host flag; a host thread answers; the GPU spins on a done flag. Plain, `volatile`, system-scope
  atomic loads, RMW atomics, fences and uncached Level Zero host memory all kept returning a stale cached value
  (waits timed out whenever the CPU needed > ~15 µs). A load with an **uncached read hint**
  (`annotated_ptr` + `read_hint<cache_control<uncached, L1, L3>>`) works: 48 handshakes per replay, all served, no
  timeouts, ~3 µs each, and the GPU waits exactly the CPU's work time.

Estimate if built: CPU and GPU share a layer's cold experts (≈ 26 + 22–30 GB/s per rank, ~100 GB/s of RAM in total),
which roughly halves the cold time: 4 rows ~226 → ~130 µs per layer (hot-bound), 16 rows ~640 → ~350 µs, i.e.
+10–15 % at 1 stream and +20–25 % at 4 streams before overheads. Needed: the plan kernel assigns some cold experts to
the CPU and copies their rows to host memory, a per-rank pool pinned to one CCD each, a wait kernel with a timeout
that falls back to the GPU path, and the CPU's partial sums added in the gather.

## Experiments, round 9 (2026-10-19/20): the CPU computes part of the cold experts (0049)

One-card microbench, µs per layer (48 layers in an XPU graph, cold share 0.08); the CPU path uses 8 threads on one
CCD, at the given share of the cold experts:

| rows (tokens) | GPU only | + CPU, 30 % | + CPU, 70 % (default) | + CPU, 100 % |
|---|---|---|---|---|
| 1 | 66 µs | 63 | 48 | 45 |
| 4 | 226 | 190 | 142 | 143 |
| 8 | 353 | 262 | 204 | 211 |
| 16 | 638 | 477 | 343 | 359 |

70 % is the default (100 % leaves the GPU's PCIe link idle and the pool becomes the bound). Errors vs the stock
path are the same as the GPU-only op (max rel ~1e-2, bf16 level); zero GPU-wait timeouts. With the pool stalled
300 ms and a 100 ms bound, the fallback recomputed the experts on the GPU, reported the timeout and switched the
CPU path off: results stayed correct and the graphs did not hang.

Two cards, one boot each (the same harness as rounds 6–8; L = the image with 0048's op, M = the op + CPU path at
70 %, one pool per rank pinned to its own CCD):

| run | 1 stream ms/step (the 4 prompts) | 4 streams tok/s · ms/step | agents tok/s (2 concurrent, total) | 1 stream ms/step after agents | 4 streams tok/s · ms/step after agents |
|---|---|---|---|---|---|
| L: GPU only | 41.1 / 37.4 / 41.5 / 38.3 | 146.6 · 77.0 | 103.9 / 120.1 | 46.8 / 36.8 / 46.0 / 38.5 | 149.5 · 73.2 |
| **M: + CPU path** | **34.6 / 31.8 / 34.4 / 31.6** | **170.1 · 62.6** | **119.4 / 129.8** | 35.6 / 30.8 / 36.2 / 30.4 | **179.2 · 60.5** |

1 stream −16 % step time, 4 streams −19 %, agents +8–15 %, acceptance unchanged, decode cold share unchanged, zero
timeouts in the engine logs. Paired perplexity on four fixed texts (CPU on vs off): +0.4 / −0.6 / +0.3 / +1.2 % —
within run-to-run noise; the CPU path rounds at the same points as the GPU one.

The handshake: per layer the plan kernel copies the CPU's experts' activation rows and route weights into a host
mailbox (its own counter makes every graph replay's request unique), the GPU runs the hot and the other cold
experts, one wait work-group polls the done flag with uncached loads (the only load that sees the CPU's store
inside a captured graph; ~3 µs, bounded), and the gather sums the CPU's results. A timeout sets a device flag,
the wait work-group computes the experts itself, and the plan switches the path off from the next call.

Not tried: overlap the next layer's CPU work behind the current one (the mailbox scheme is per layer; the CPU is
busy ~100–150 µs of a ~260 µs layer), and per-layer lookahead needs the next layer's routing, which decode cannot
know. 0043's swaps still move cold experts between RAM tiers; the CPU pool reads whatever is pinned.

## Excluded or not worth it (as of 2026-10-11)

| item | why |
|---|---|
| more hot experts at the same util (144, 152), 64K context with 152 hot | cost KV (excluded: long coding sessions need the KV) |
| 8,192-token prefill batches | +8–16 % prefill for half the KV; 0042 gives +50 % for none |
| GPU power cap changes | excluded (host stability) |
| PLE row cache 6 GiB, PLE lookahead | no gain: 99 % hits when warm, 0.1 ms bubble |
| MTP k=2 | acceptance −15–24 % beats the shorter step: −6 % |
| FP8 KV | +57 % KV but decode −5–8 %, prefill at 30K −15 %; only when KV-bound |
| rankings rebuilt once from a small sample | overfit (cold share 24 → 33 % on other prompts); 0043 adapts instead |
| two queues for hot/cold at decode | PCIe-bound, nothing to overlap |
| async scheduling | KV −27K, no gain |
| a better MTP draft | none available without VRAM (KV) |
| draft-only reduced vocabulary | real output uses too much of the vocabulary; acceptance loss beats the saved reads |
| adaptive draft length | DSpark-only in vLLM; conflicts with local-argmax drafting |
| cold-expert prefetch at decode | needs data-dependent copies inside the decode graphs |
| fixed VRAM pool for cold experts (P = 4, two-step lag) | catches 26–29 % of cold routes; ~+1–4 % net for ~0.47 GiB |
| 154 hot experts | +1–4 % for −31.6K KV |
| deduplicating the draft's lm_head / embed | already shared by vLLM |
| fine-grained Mamba prefix cache | not applicable to multi-module MTP |
| all-reduce tuning | 22 µs per call, ~2–3 ms per step |
| CPU compute for cold experts | ~~breaks the decode graphs~~ → built as 0049 (round 9): the CPU pool + mailbox handshake work and are on by default |

## How to reproduce the measurements

Engine: `engines/two-cards/scripts/restart-tp2.sh` (detached start; wait for `[run-tp2] warm-up request OK` in `engine-data/engine.log`).
Client scripts in `tools/bench_tp2/`, run in any container on the host network (e.g. the dev container, with this
repository mounted):

```bash
python3 tools/bench_tp2/bench2.py 1:20:512 2:20:256 4:20:256 4:20:512          # decode, 1/2/4 streams
python3 tools/bench_tp2/bench2.py 1:2800:64 1:11000:64 1:28000:64 1:55000:64 1:28000:256   # prefill, long-context decode
python3 tools/bench_tp2/bench_mtp.py                                           # real prompts, MTP acceptance
python3 tools/bench_tp2/repro.py 1050 1 600                                    # known issue 13 (must print OK)
python3 tools/bench_tp2/bench_det.py                                           # decode: fixed prompts, temp 0, ms/step
python3 tools/bench_tp2/quality.py <tag>        # needles 32K-97K + short answers -> /work/quality-<tag>.json (diff two)
python3 tools/bench_tp2/prefix_test.py 6000     # prefix cache: TTFT and cached tokens for a repeated 18K prompt
python3 tools/bench_tp2/stress_mixed.py         # 3 long prefills during 2 decodes; all must be ok
AGENT_TEMP=0 python3 tools/bench_tp2/agent_load.py 3 12 2   # code prompts: 12 sessions x 5 turns, 2 concurrent requests
python3 tools/bench_tp2/burst_check.py 30       # known issue 6: 30 bursts of 4 requests, degenerate outputs
python3 tools/bench_tp2/sim_swap.py "<dir>/r0-*.json"   # offline replay of hot-set policies (route snapshots)
python3 tools/bench_tp2/ppl.py <tag>            # perplexity on fixed texts (code, docs, Russian) via prompt logprobs
python3 tools/bench_tp2/run_eval.py <tag> 3000  # MMLU (3,000 sampled) + HumanEval completions -> /work/eval/<tag>-*.json
docker run --rm --network none -v $PWD/engine-data/eval:/eval --entrypoint /opt/venv/bin/python b70-flash-next:tp2 /eval/he_exec.py /eval/<tag>-humaneval.json
python3 tools/bench_tp2/compare.py <tagA> <tagB>   # paired MMLU (McNemar) and HumanEval differences
python3 tools/bench_tp2/pool_reuse.py trace-rankRANK.jsonl   # B70_MOE_ROUTE_TRACE output: pool hit rate by size/lag
python3 tools/bench_tp2/pool_churn.py trace-rank0.jsonl      # pool DMA volume vs UVA reads avoided
AGENT_DUMP=a.json AGENT_TEMP=0 python3 tools/bench_tp2/agent_load.py 3 12 2; python3 tools/bench_tp2/agent_judge.py a.json
torchrun --nproc-per-node 2 tools/bench_tp2/ar_bench.py   # 2-card all-reduce latency (engine stopped; tp2 image + env)
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
5. **The engine.** Boot `engines/two-cards/scripts/run-tp2.sh`; check the boot lines `B70 hot/cold: 48 layers split`, `PLE NVMe host
   path ready`, `XPU KV cache size`; then a chat request, a `json_schema` request with MTP, a long-context needle, and
   a speed run (decode at 1 and 4 streams, prefill at 3K and 12K tokens). The numbers above came from that sequence.

Not done yet, and worth doing before relying on it:

- **Quality** against the 4-card engine on the same items (MMLU, TruthfulQA, the agentic coding set, 64K document
  questions). The target tensors are Intel's, but the PLE table is RadixArk's FP8 re-quantised to INT8, not devan's
  BF16-derived table, and the MTP draft is a different (INT4-g32) one.
- **Endurance**: a multi-day run under real load, watching host MemAvailable (13–16 GiB headroom) and preemptions.
- **Concurrency**: more than 4 slots, and the breakpoint where 4,096-token chunks of long prompts slow decode down.

## What to improve

1. ~~Measure, then set the hot set from this host's traffic~~: 0041 counts routes, 0043 swaps experts as traffic changes.
2. ~~Bigger prefill chunks~~ and ~~overlap the cold path in prefill~~: done as 0042 (prefetch into VRAM slots).
3. ~~Overlap the cold path at decode~~: the fused op (0048) with the plan kernel and cold workers (round 8) runs at
   82–91 % of the PCIe bound; what is left needs fewer bytes over PCIe (item 4, or more hot experts).
4. ~~CPU compute for cold experts at decode~~: built as 0049 and on by default (round 9): 1 stream −16 % step time,
   4 streams −19 %, agents +8–15 %; the timeout fallback keeps the graphs safe.
5. **Read the FP8 PLE shards in place** instead of a 52.5 GB INT8 copy: the 0013 reader would need multi-file row
   offsets and an FP8 dequant path (0007 has one).
6. **Put B70-0040 on the fork's release branch** so `make-tree.sh` for a release includes it.
