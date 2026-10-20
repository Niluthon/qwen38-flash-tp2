# Runtime switches (`B70_*`)

Every feature the b70 series adds is behind an environment variable and **off by default**. With none of them set,
the image behaves as vLLM v0.30.0 plus wu1ff's patch set (0001–0005), apart from the ungated parts listed at
the end. Set the variables in the serving container's environment; `vllm serve` flags are unchanged.

A variable that is not listed here is ignored. Older builds used a `LUMNUS_` prefix for the same switches; those
names are no longer read, so a deployment that still sets them runs with the feature off.

## PLE n-gram table (0006, 0007, 0008, 0013, 0013b, 0050)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_PLE_DIRECT_PINNED` | unset (off) | 0006 | `1`: fill the pinned PLE slabs straight from the memory-mapped table file, per TP rank, instead of first loading a pageable copy per rank. Measured boot host-RAM peak on 4 ranks with the BF16 table: ~191 GiB → ~158 GiB. |
| `B70_PLE_FP8` | unset (off) | 0007 | `1`: keep the table as FP8 E4M3 (one global scale) in the pinned slabs; needs `B70_PLE_FP8_PATH`. Exclusive with `B70_PLE_INT8`. |
| `B70_PLE_FP8_PATH` | — | 0007, 0050 | `.safetensors` with a `table` tensor (or the checkpoint's `…shard_<i>.weight` keys) plus one `weight_scale`. Since 0050 also a checkpoint directory or its `model.safetensors.index.json`: then only the `…ple_embedding.…` tensors are used, from every file that holds one (they must share one prefix). |
| `B70_PLE_FP8_CROSSCHECK` | `1` | 0007 | `0` skips the boot cross-check against the BF16 table (only runs when `PLE_TABLE_PATH` is set). |
| `B70_PLE_FP8_MAX_REL_ERR` | `0.25` | 0007 | refuse to start above this relative L2 error on 4,096 random rows. |
| `B70_PLE_FP8_DEQUANT` | `lut` | 0007 | `cast` uses the stock cast-and-multiply instead of the 256-entry lookup table (bitwise equal). |
| `B70_PLE_INT8` | unset (off) | 0008 | `1`: keep the table as INT8, one fp32 scale per row (164 B/row, 48.9 GiB over 4 ranks instead of 95.4 GiB BF16); needs `B70_PLE_INT8_PATH`. Build the table with `tools/build_int8_ple.py` (docs/ple-int8.md). |
| `B70_PLE_INT8_PATH` | — | 0008 | the INT8 table `.safetensors`. Its metadata must carry the format tag `lumnus-ple-int8-rowscale/v1` (see below). |
| `B70_PLE_INT8_CROSSCHECK` | `1` | 0008 | `0` skips the boot cross-check against the BF16 table. |
| `B70_PLE_INT8_MAX_REL_ERR` | `0.02` | 0008 | refuse to start above this relative L2 error (expected ~0.0066). |
| `B70_PLE_INT8_NVME` | unset (off) | 0013 | `1` (with `B70_PLE_INT8=1`): serve the INT8 table from NVMe through a small pinned row cache instead of pinning all of it (docs/ple-nvme.md). |
| `B70_PLE_INT8_NVME_PATH` | `B70_PLE_INT8_PATH` | 0013 | the table file to read with `O_DIRECT`. |
| `B70_PLE_INT8_NVME_CACHE_GIB` | `8` | 0013 | pinned row cache, GiB **in total over all TP ranks**. We run `8` (2 GiB per rank on 4 ranks); the hit rate on agent traffic was 97–98 %. |
| `B70_PLE_INT8_NVME_IO_THREADS` | `16` | 0013 | read threads per rank (also the io_uring queue depth with `READER=uring`). |
| `B70_PLE_INT8_NVME_STATS` | `0` | 0013 | `1`: per-rank stats line (hits, misses, reads, host bubble). |
| `B70_PLE_INT8_NVME_STATS_S` | `60` | 0013 | stats period, seconds. |
| `B70_PLE_INT8_NVME_SYNC_ONLY` | `0` | 0013 | diagnostic: keep the table in RAM but add the per-step host sync the NVMe path needs, to measure that cost alone. |
| `B70_PLE_INT8_NVME_BOOT_SAMPLE` | `65536` | 0013 | rows compared byte-for-byte against the memory-mapped file at boot. |
| `B70_PLE_INT8_NVME_READER` | `py` | 0013b | `native` = a C thread pool (one call per batch, GIL released); `uring` = io_uring via raw syscalls. `native` is the one we run. |
| `B70_PLE_INT8_NVME_LOOKAHEAD` | `0` | 0013b | `1`: read the next prefill chunk's rows in the background during the current step. |
| `B70_PLE_INT8_NVME_LOOKAHEAD_TOKENS` | scheduler's `max_num_batched_tokens` | 0013b | tokens predicted per step (unset or `0` = the default). |
| `B70_PLE_INT8_NVME_NATIVE_DIR` | `$TMPDIR/b70-ple-nvme` | 0013b | where the C reader is compiled at first use (needs `gcc`, present in the base image). |
| `B70_PLE_INT8_NVME_NATIVE_LIB` | — | 0013b | use this prebuilt reader `.so` instead of compiling. |
| `B70_PLE_FP8_NVME` | unset (off) | 0050 | `1` (with `B70_PLE_FP8=1`): serve the FP8 table from NVMe in place, straight from the shards `B70_PLE_FP8_PATH` names (160 B/row, the checkpoint's own values), through the 0013 row cache and host hook; no INT8 copy and no pinned table (docs/ple-nvme.md). Every `B70_PLE_INT8_NVME_*` tuning switch above applies to it too (cache size, I/O threads, stats, sync-only, boot sample, reader, lookahead, native dir/lib); `B70_PLE_INT8_NVME` and `B70_PLE_INT8_NVME_PATH` do not. |

`PLE_TABLE_PATH` (wu1ff's variable, the BF16 table) is still read: it is the source table for BF16 serving and the
reference for the FP8/INT8 cross-checks.

**The INT8 table format tag stays `lumnus-ple-int8-rowscale/v1`.** It is a data-contract string written into the
table file's metadata by `tools/build_int8_ple.py` and compared by equality in 0008 and 0013; renaming it would make
every existing table unloadable. It is not a switch.

## Chat API defaults (0009, 0010, 0012)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_THINKING_BUDGET` | unset (off) | 0009 | per-effort cap on reasoning tokens, keyed on the effort the **client** asked for: `minimal=512,low=512,medium=2048,high=4096,xhigh=8192,max=12288,ultra=12288,default=8192`. `default` applies when no effort was sent. Nothing applies when thinking is off (`enable_thinking=false` or an effort of `none`/`off`/`false`/`0`/`disabled`), and a request's own `thinking_token_budget` wins. |
| `B70_DEFAULT_PRESENCE_PENALTY` | unset | 0009 | server default for `presence_penalty` (we use `0.7`); a request's own value wins. |
| `B70_DEFAULT_REPETITION_DETECTION` | unset | 0010 | server default for vLLM's built-in n-gram repetition stop, `max=<n>,min=<n>,count=<n>` (we use `max=1,min=1,count=128`: stop when one token repeats 128 times). Catches the token-1023 "duct" NaN loop. A request's own `repetition_detection` wins. Chat endpoint only. |
| `B70_REASONING_EFFORT_ALIAS` | unset (off) | 0012 | `1`: map an OpenRouter-style `{"reasoning": {"effort": X}}` onto `reasoning_effort` when the request did not set that field. |

Effort-alias map (0012): `reasoning.effort` is lower-cased and trimmed; `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` and `ultra` become `reasoning_effort` unchanged; any other value is ignored. The budget table (0009)
is then looked up with that effort.

## KV offload (0014a–f)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_OFFLOAD_TRACE` | unset (off) | 0014a/f | `1`: boot line, a state line every period, and the mechanism events only (`junction-set`, `zeroed_by`), once per request per value. `2`: also every per-request lookup/store/load/hand-off/req-done line. Log prefix `B70-OFFLOAD`, logger `vllm.b70_offload`. |
| `B70_OFFLOAD_TRACE_PERIOD_S` | `60` | 0014a | state-line period (a daemon thread, so it also fires when idle). |
| `B70_OFFLOAD_JUNCTION` | unset (off) | 0014b | `1`: when a Mamba/GDN group cuts an offload hit below what the full-attention groups proved, pin a shared-prefix junction there so the recompute stores that state. A "same document, new question" revisit misses once, then hits (docs/offload-fix.md). |
| `B70_OFFLOAD_GDN_BACKSTEP` | `0` | 0014c | `N`: also retain the Mamba/GDN states at the N block boundaries below the replay boundary, so the first revisit hits too. Cost per request: N × one Mamba state set (~116 MB for Qwen3.8-Flash-Next) in the CPU tier. |
| `B70_OFFLOAD_EMPTY_ADVANCE_GUARD` | unset (off) | 0014d | `1`: guard for vllm#56795 (do not advance past keys still write-pending under another request). |
| `B70_OFFLOAD_GROUP_EVICT` | unset (off) | 0014e/f | `1`: select the vllm#51787 backport (request-scoped recency, tail-before-head eviction across all KV groups). Unset = upstream v0.30.0 LRU/ARC, byte for byte. |

Candidate default for the offload fix ("0014jb"): `B70_OFFLOAD_TRACE=1 B70_OFFLOAD_JUNCTION=1
B70_OFFLOAD_GDN_BACKSTEP=1 B70_OFFLOAD_EMPTY_ADVANCE_GUARD=0 B70_OFFLOAD_GROUP_EVICT=0`.

## Hot/cold routed experts (0040)

For XPU W4A16 MoE (INT4, `XPUExpertsWNA16`) when VRAM cannot hold every routed expert (docs/two-cards.md).

| variable | default | patch | effect |
|---|---|---|---|
| `B70_MOE_HOTCOLD` | unset (off) | 0040 | `1`: keep the `B70_MOE_HOT_EXPERTS` most-routed local experts per layer in VRAM and the rest pinned in host RAM, read by the same kernel through UVA. Exact (hot + cold = all in VRAM, bit for bit). MTP layers are never split. |
| `B70_MOE_HOT_EXPERTS` | all local experts (= off) | 0040 | hot experts per layer **per rank**. Each one costs 48 × 2.42 MiB = 116 MiB of VRAM per rank on Qwen3.8-Flash-Next and frees the same in host RAM. We run `150` on 2× B70 (EP 2: 256 local experts) with `B70_EMBED_UVA=1` and `B70_DENSE_FP8`. |
| `B70_MOE_HOTCOLD_RANKINGS` | unset (experts by id) | 0040 | JSON `{"<layer>": [global expert ids, hottest first]}`; experts it does not list follow by id. The albucino checkpoint ships `runtime/static_hot_cache_rankings.json`. |
| `B70_MOE_HOTCOLD_MEMLOG` | unset (off) | 0040 | `1`: log torch allocated / reserved and device free memory after every layer's split (otherwise every 12th layer). |
| `B70_MOE_COLD_PREFETCH` | unset (off) | 0042 | `1`: batches with at least `B70_MOE_COLD_PREFETCH_MIN_TOKENS` tokens (prefill) run the cold experts from two VRAM slots instead of UVA reads; layer L+1's cold experts are DMA'd on a copy stream while layer L computes. Same kernel and bytes (outputs unchanged). Costs two layers' cold experts of VRAM (0.53 GiB per card at 136–144 hot). On 2× B70: prefill ~2,600 → ~3,950 tok/s. |
| `B70_MOE_COLD_PREFETCH_MIN_TOKENS` | `512` | 0042 | smallest batch that uses the slots (min 64); smaller batches, and every decode graph (capture sizes ≤ 16), keep the UVA path. |
| `B70_MOE_ROUTE_COUNT` | unset (off) | 0041, 0043, 0047 | `prefill`: count routes per (layer, global expert) in eager batches (prefill, mixed) only, nothing added to the decode graphs; `1`: also inside the decode graphs (two kernels per layer since 0047, cost within noise; four before, ~4 %). Log line per period: the share of this rank's prefill and decode routes served by cold experts. Eager counting starts with the first real batch (the profiling run, and so the KV size, is unchanged). |
| `B70_MOE_ROUTE_COUNT_S` | `60` | 0041 | period of the log line / dump, seconds (read on the host after the per-step sync of the NVMe PLE path, so it needs an NVMe PLE path: `B70_PLE_INT8_NVME=1` or `B70_PLE_FP8_NVME=1`). |
| `B70_MOE_ROUTE_COUNT_PATH` | unset | 0041 | write cumulative counts to `<path>-rank<r>.json` (`{"<layer>": [count per global expert]}`), the input for re-ranking. |
| `B70_MOE_ROUTE_TRACE` | unset | 0046 | with `B70_MOE_ROUTE_COUNT=1`: write `<path>-rank<r>.jsonl`, the cold set per layer and then one line per decode step with the experts each layer routed to. A diagnostic for expert-cache studies (`tools/bench_tp2/pool_reuse.py`). |
| `B70_MOE_HOTCOLD_SWAP` | unset (off) | 0043 | `1`: the hot set follows the traffic. Every period each rank scores its local experts from the prefill route counts (implies `B70_MOE_ROUTE_COUNT=prefill`) and swaps experts between VRAM and pinned RAM in place (weights, scales and expert maps; no graph recapture). Exact: outputs do not depend on placement. We run it on 2× B70. |
| `B70_MOE_HOTCOLD_SWAP_S` | `5` | 0043 | period, seconds (one round costs 7–16 ms on 2× B70). |
| `B70_MOE_HOTCOLD_SWAP_DECAY` | `0.95` | 0043 | per-period decay of the scores (half-life ~70 s at 5 s). |
| `B70_MOE_HOTCOLD_SWAP_PRIOR` | `0.01` | 0044 | weight of the rankings-file order in the score (0.05 before 0044; an offline replay of route traces from code prompts at 2 concurrent requests preferred 0.005–0.01). |
| `B70_MOE_HOTCOLD_SWAP_BUDGET` | `8` (`4` in 0043) | 0043, 0044 | most swaps per layer and period; a swap needs the cold score to beat the hot one by 10 %. |
| `B70_MOE_HOTCOLD_SWAP_SOURCE` | `prefill` | 0047 | what the swap scores count: `prefill` = eager-batch routes (0043); `decode` = the decode graphs' routes (needs `B70_MOE_ROUTE_COUNT=1`, else falls back to `prefill` with a warning) plus `B70_MOE_HOTCOLD_SWAP_PREFILL_W` x the prefill routes, each normalised per layer and period. Prefill batches ≥ 512 tokens read all cold experts from the 0042 slots, so only decode pays for a cold expert. Two-card default: `decode`. |
| `B70_MOE_HOTCOLD_SWAP_PREFILL_W` | `0.1` | 0047 | weight of the prefill routes with `SOURCE=decode`. |
| `B70_MOE_HOTCOLD_STATE` | unset (off) | 0047 | `<prefix>`: keep a long-term score per expert and write it to `<prefix>-rank<r>.json`; at boot, a state with enough periods places the hot set (instead of the rankings file alone) and seeds the swap scores. Two-card default: `/work/hotcold-state`. Delete the files to start from the checkpoint's ranking again. |
| `B70_MOE_HOTCOLD_STATE_DECAY` | `0.999` | 0047 | long-term decay per active swap period (idle time does not count; half-life ~58 min of traffic at 5 s). |
| `B70_MOE_HOTCOLD_STATE_S` | `300` | 0047 | seconds between state writes. |
| `B70_MOE_HOTCOLD_STATE_MIN_PERIODS` | `12` | 0047 | a state with fewer active periods is ignored at boot. |
| `B70_MOE_HOTCOLD_STATE_PRIOR` | `0.05` | 0047 | weight of the rankings-file order when the state places the hot set (tie-break for experts the state never saw). |
| `B70_MOE_DECODE_OP` | unset (off) | 0048 | path to `libb70_moe_decode.so` (`1` = the image's copy next to `b70_hotcold.py`, built in the image's `b70moe` stage; or a self-built one, `kernels/b70-moe-decode/build.sh`): batches of at most `B70_MOE_DECODE_OP_MAX_ROWS` tokens run hot and cold experts in one fused SYCL op (3 kernels, each expert read once) instead of two `XpuFusedMoe` calls. silu without a clamp only; if the library does not load, a warning and the stock path. Two-card default: `1`. |
| `B70_MOE_DECODE_OP_MAX_ROWS` | `16` | 0048 | largest batch (tokens) for the fused op (at most 16). |
| `B70_DRAFT_HEAD_FP8` | unset (off) | 0048 | `1` (XPU): the MTP draft passes use an FP8 W8A16 copy of the lm_head shard (1.08 → 0.56 ms per pass); the target verifies with its BF16 head, so outputs are unchanged. +318 MB per card (KV −7,600 tokens at 262K). Two-card default: `1`. |
| `B70_MOE_CPU` | unset (off) | 0049 | `1`: the decode op gives `B70_MOE_CPU_PCT` % of each layer's cold experts to a host pool that computes them in pinned RAM (AVX-512 BF16, same bf16 roundings) while the GPU does the hot and the other cold experts; the GPU waits per layer on a mailbox in host memory (uncached loads, ~3 µs). Needs the CPU-path kernel (the image builds it) and an AVX-512 BF16 CPU. On a GPU-wait timeout the op computes the experts itself and switches the path off (a stderr line). Anything but `1` = off: every cold expert stays on the GPU path. Each rank logs `B70 decode op: CPU path on` / `off (B70_MOE_CPU=…)` / `requested but this CPU has no AVX-512 BF16` (then it is set to `0`). Two-card default: `1`; with Compose also settable from the host: `B70_MOE_CPU=0` in the root `.env` or the shell overrides both env files. |
| `B70_MOE_CPU_PCT` | `50` | 0049 | percent of a layer's cold experts computed by the CPU pool (70 measured best; 100 was slightly worse at 16 rows). |
| `B70_MOE_GLUE` | unset (off) | 0051 (dev tree only: `kernels/b70-moe-decode/vllm-0051-moe-glue.patch`) | `1`: at decode (<= `B70_MOE_DECODE_OP_MAX_ROWS` tokens) the 0047 route count runs inside the decode op's plan kernel (`moe_decode_counted`, replaces remainder + `index_add_`) and top-k softmax runs as `b70moe.topk_softmax` (same ids, weights within a few fp32 ulp, degenerate rows like the stock op). Needs `B70_MOE_DECODE_OP` and a library built from the current `kernels/b70-moe-decode`. ~1.5 % per decode step (docs/two-cards.md, round 11). |
| `B70_MOE_GLUE_PARTS` | `count,topk` | 0051 | which glue pieces `B70_MOE_GLUE=1` enables. |
| `B70_MOE_CPU_MIN_ROWS` | `1` | 0049 | batches with fewer tokens than this skip the CPU path. |
| `B70_MOE_CPU_THREADS` | `8` | 0049 | threads of the per-rank pool (poller included). |
| `B70_MOE_CPU_FIRST_CORE` | `8 × TP rank` | 0049 | first core the pool pins its threads to (cores first … first + `B70_MOE_CPU_THREADS` − 1): unset = each rank its own 8-core CCD on a 16-core host; one number = the same first core for every rank; a comma list = one per TP rank, e.g. `0,6` for two pools of 6 threads on a 12-core CPU. |
| `B70_MOE_CPU_TIMEOUT_MS` | `500` | 0049 | GPU wait bound per layer before the timeout fallback. |
| `B70_MOE_CPU_STALL_US` | `0` | 0049 | tests: stall the pool this long per request, to exercise the timeout fallback. |

## Input embedding in host RAM (0044)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_EMBED_UVA` | unset (off) | 0044 | `1` (XPU): after loading, the target's `embed_tokens` moves into pinned host RAM behind a UVA view (0.59 GiB per card at TP 2); lookups read a few KB per token over PCIe. lm_head and the MTP draft are untouched. Spend the freed VRAM on `B70_MOE_HOT_EXPERTS`. |

## FP8 dense layers (0045; on in tp2-hotcold since round 4)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_DENSE_FP8` | unset (off) | 0045 | a regex: 16-bit linear layers of an INC checkpoint whose names match are quantized at load time to FP8 E4M3 (one scale per output channel) and run by the XPU weight-only kernel `fp8_gemm_w8a16`; lm_head never. Tested pattern: `(linear_attn\.(in_proj_qkv\|in_proj_z\|in_proj_qkvz\|out_proj)\|self_attn\.(qkv_proj\|q_proj\|k_proj\|v_proj\|o_proj))$` (docs/two-cards.md, round 3). On 2× B70: MMLU 87.33 → 87.23 % (3,000 paired questions, p = 0.79), HumanEval 157 → 158 / 164, perplexity +0.4–0.7 %; −0.9 GiB per card (spent on hot experts). |

## Hyper-connection down GEMM (0052; on by default)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_HC_KSPLIT_VIEWS` | `1` (on) | 0052 | prefill rows (> 64) of the hyper-connection down GEMMs run as 0003's two K-half GEMMs over strided views of the one weight instead of contiguous copies: −0.68 GiB per card (+46K KV tokens), results bitwise equal, decode untouched (docs/two-cards.md, round 12). `0`: the copies of 0003. |

## Ungated parts (active even with every switch unset)

1. 0009: `reasoning_effort` also accepts `"ultra"` (API widening only).
2. 0013: the KV-offload host-tensor allocation log line moves from DEBUG to INFO and prints `is_pinned()`.
3. 0014f: `ReqContext` records key positions (a dict write per key, never read with `B70_OFFLOAD_GROUP_EVICT` unset),
   and the policy base class gains optional hooks that are never called with the flag unset.
4. 0018: when one pinned allocation for the CPU KV pool is refused (≥ ~31 GiB on a B70), the pool is stored as
   equal chunks instead of failing at boot. A pool that fits in one allocation is unchanged.
5. 0019: on dense-QSA configs, checkpoint tensors under `self_attn.indexer` are skipped at load (checkpoints without
   them, such as devan's, are unaffected).

## Added in 0.30.0-b70.2 (0020–0032)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_MTP_DRAFT_PREFILL_NO_PIECEWISE` | unset (off) | 0021 | `1`: on XPU, the MTP draft model keeps only FULL (uniform-decode) graphs; mixed prefill batches run the draft eagerly. We serve with `1` (the PIECEWISE draft prefill capture failed on XPU). |
| `B70_MTP_UNIFORM_DRAFTS` | `1` (on) | 0026 | never schedule a partial draft list on XPU (grammar-trimmed drafts crashed the stock 0.1.14 GDN kernel). `0` turns it off; with kernels that carry upstream #600 (our b70.2/b70.3) it is not needed. |
| `B70_GDN_MODE` | `index64` when `libgdn_index64.so` is present, else `official` | 0023, 0027 | which GDN op runs: `index64` (wu1ff's library), `official` (the stock op: 64-bit with our kernels, 32-bit with stock 0.1.14.1), `split-index64`, `split-official` (debug). Re-read every 2 s. |
| `B70_PROMPT_LOGPROBS_CHUNK` | `128` | 0029, 0029b | rows per prompt-logprobs chunk (bounds the [rows, vocab] fp32 transient). `0` = vLLM's own chunking (one pass on V1, 1,024 rows on V2). |
| `B70_OFFLOAD_H2D_DIRECT` | unset (off) | 0031 | `1`: CPU→GPU KV loads copy straight from the pinned offload pool instead of the kernel's staged batch copy. For kernels without B70-K1; with our 0.1.14.1+b70.3, leave it off. |

Debug tools in 0023/0023c (off unless their flag file exists): a NaN tracer in the GDN wrapper and at the sampling
boundary, and a probe that zeroes block 0 around FULL replays. The flag files are fixed paths
(`/run/b70-probes/gdn-mode`, `/run/b70-probes/gdn-nancheck`, `/run/b70-probes/zero-null-block`,
`/run/b70-probes/uniform-drafts-off`), re-read every 2 s; with no such files the code does nothing. (Before the
0.30.0-b70.2 release the fork branches used another directory; only the path changed.)

Ungated in b70.2: 0020 and 0022 (MTP loader indexer skip; vllm#55506 port), 0028 (acts only on an INC config), 0032
(log once). Every patch that adds a log line logs once per process when it first acts (e.g. `B70-0029b`, `B70-0031`),
so a deployment can confirm what is active.

## Measurement only (dev tree, not in the image)

| variable | default | source | effect |
|---|---|---|---|
| `B70_STEP_TRACE` | unset (off) | `kernels/b70-step-trace` (`vllm-step-trace.patch`) | path to `libb70_step_trace.so`: host-stamped marks at module boundaries and around every all-reduce, captured into the decode graphs; ~+1.5–2 ms per step. Report: `tools/bench_tp2/step_trace_report.py`. |
| `B70_STEP_TRACE_DIR` / `_EVERY` / `_DEPTH` | `/work/step-trace` / `256` / `2` | same | dump directory, engine steps between dumps, module depth below a decoder layer that gets marks. |
| `B70_STEP_TRACE_WRAP` | unset | same | `module:Class.method,...`: marks around methods that are not module forwards. |
| `B70_STEP_TRACE_OPS` | unset | same | as `_WRAP`, plus a mark around every op the method dispatches (TorchDispatchMode; ~+10 ms per step on the MoE block). |
| `B70_STEP_TRACE_CORE_<dev>` | `23 + 8 * dev` | same | core of the poller thread that stamps the marks. |
