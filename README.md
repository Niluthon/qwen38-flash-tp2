# qwen38-flash-tp2

Run **Qwen3.8-Flash-Next** (W4A16) with vLLM on **two Intel Arc Pro B70** cards (2 × 32 GB) and ~64 GB of host RAM.
This repository holds the vLLM patch series, the kernel build, the engine settings and the measurements; one Docker
image is built from them. It is the two-card line of [Lumnus/b70-flash-next](https://github.com/Lumnus/b70-flash-next);
the one-card and four-card setups are not part of this repository.

The engine serves an OpenAI-compatible API (`/v1/chat/completions`, tool calls, reasoning, structured output) on
port 8000, model name `qwen38-flash-next`.

## How to run

Needs: 2× Arc Pro B70 with the `xe` driver, ~64 GB RAM, ~185 GB free on an NVMe disk, Docker Compose v2.24+, an x86-64
CPU (AVX-512 BF16 makes decode faster, see [CPU](#cpu)).

```bash
git clone https://github.com/Niluthon/qwen38-flash-tp2 && cd qwen38-flash-tp2
cp .env.example .env               # optional: MODELS=/path/on/nvme (default ~/models)
docker compose build               # the image: vLLM release 0.30.0-b70.2 + two-card patches 0040-0050, 0052, 0053 + kernels + shim
docker compose run --rm prepare    # once: download (~129 GB) and prepare the weights
docker compose up -d               # serving on port 8000 after ~5 min
docker compose logs warmup         # wait for "warm-up request OK" before sending long prompts
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen38-flash-next", "messages": [{"role": "user", "content": "Hello"}]}'
```

Stop with `docker compose down`. **Full guide: [HOW-TO-RUN.md](HOW-TO-RUN.md)** (machine check, daily commands,
settings, weights you already have, your own model, troubleshooting).

## Speed of the current build

Image `b70-flash-next:tp2` with patches 0001–0050, 0052 and 0053, default settings, no overrides. Host: Ryzen 9 7950X, 61 GiB DDR5,
2× B70 on PCIe 5.0 x8. Measured 2026-10-20 on an idle engine (docs/two-cards.md, round 10, run P; KV pool from round
12, run Q4: expandable segments in the torch allocator, 0052 and 0053, speed unchanged).

Units used below:

- **tok/s** — generated tokens per second. For one request it is what a user sees; for several it is the total of all.
- **ms/step** — time of one decode step (one forward pass of the model plus 3 MTP draft passes). Lower is faster.
  It is the stable number for comparing builds (repeats within ~3 %); tok/s also depends on the text.
- **tokens/step** — tokens accepted per step (1 + accepted MTP drafts, at most 4). tok/s = tokens/step ÷ ms/step × 1000.

**Decode, one request at a time** (`tools/bench_tp2/bench_det.py`: four fixed prompts, temperature 0):

| prompt | tok/s | ms/step | tokens/step |
|---|---|---|---|
| explain how a refrigerator works | 71.6 | 36.0 | 2.57 |
| write a Python function | 106.6 | 32.6 | 3.49 |
| write a short story | 68.8 | 33.6 | 2.31 |
| compare TCP and UDP | 90.0 | 33.4 | 3.01 |
| **mean** | **84.3** | **33.9** | **2.85** |

**Load, prefill and capacity:**

| measurement | value | unit |
|---|---|---|
| decode, 4 requests at once, total of the 4 (same prompts) | 176.8 | tok/s |
| decode, 4 requests at once, step time | 62.5 | ms/step |
| coding agents, 12 sessions at 2 concurrent requests, total (`agent_load.py`, two runs) | 119.3 / 137.1 | tok/s |
| prefill, one request, 3K / 12K / 30K-token prompt | 4,021 / 4,143 / 3,973 | tok/s (prompt tokens) |
| KV cache pool, shared by all requests (each card holds half of the attention heads of every token) | 400,513 | tokens |
| context length, one request at most | 262,144 | tokens |
| requests served at once (more are queued) | 4 | requests |
| boot to serving | 4–5 | min |

**[BetterBench](https://github.com/GGZ14/BetterBench)** (20 passes per category, temperature 0.7, 2026-10-20, commit
`816cc78`, i.e. before 0050, which measured speed-neutral):

| measurement | value | unit |
|---|---|---|
| decode, 1 request, weighted median over 8 categories | 94.1 | tok/s |
| time to first token, 1 request, p50 | 281 | ms |
| gap between streamed updates, p99 (stutter) | 40.8 | ms |
| decode total at 1 / 2 / 4 / 8 concurrent requests | 81.6 / 121.9 / 166.1 / 165.4 | tok/s |
| prefill median at 1.5K / 6K / 12K / 24K / 47K-token prompts | 2,368 / 3,883 / 3,921 / 3,863 / 3,689 | tok/s (prompt tokens) |

**CPU path on vs off** (the same image, one boot each, 2026-10-20, `bench_det.py`; off = `B70_MOE_CPU=0` or a CPU
without AVX-512 BF16, see [CPU](#cpu)):

| CPU path | 1 request, ms/step (4 prompts) | 1 request, tok/s (4 prompts) | 4 requests, tok/s total | 4 requests, ms/step |
|---|---|---|---|---|
| on (default) | 33.9 / 31.4 / 32.1 / 31.6 | 77.7 / 109.0 / 72.0 / 96.0 | 180.2 | 62.1 |
| off | 40.1 / 35.7 / 38.8 / 36.7 | 67.0 / 97.5 / 57.5 / 79.5 | 142.1 | 75.1 |
| on vs off | −12 to −17 % | +12 to +25 % | +27 % | −17 % |

Prefill does not use the CPU path (large batches take 0042's prefetch path).

Quality: MMLU 87.7 % (3,000 sampled questions), HumanEval 158 / 164, measured 2026-10-20 on the current build's
numerics (round 12: FP8 target head, 0053; with the BF16 head 87.5 % and 158 / 164). Single runs vary by ±5–10 % in
tok/s.

## CPU

The decode MoE op (0048/0049) lets the CPU compute 70 % of the cold experts each step (`B70_MOE_CPU=1`, on by
default). That code needs **AVX-512 BF16** (`avx512_bf16` in `/proc/cpuinfo`):

| CPU | CPU path | what happens |
|---|---|---|
| AMD Zen 4 / Zen 5 (Ryzen 7000 / 9000, Threadripper 7000 / 9000, EPYC 9004 / 9005) | yes | as measured above (Zen 4) |
| Intel Xeon 4th gen (Sapphire Rapids) and newer, Xeon 3rd gen Cooper Lake | yes | not tested here |
| Intel Core 12th gen and newer, Core Ultra (no AVX-512), AMD Zen 3 and older, older Xeons | no | the engine runs; the op checks the CPU at start and keeps every cold expert on the GPU (the "without the CPU path" numbers above) |
| non-x86 (ARM) | — | the image is x86-64 only |

The library is compiled for generic x86-64; only the CPU-path function uses AVX-512 instructions, and it runs only after
the start-up check (`__builtin_cpu_supports("avx512bf16")`), so it cannot fault on another CPU. Each card logs its
state at start: `docker compose logs engine | grep "CPU path"` shows `CPU path on`, `CPU path off (B70_MOE_CPU=0)` or
`CPU path requested but this CPU has no AVX-512 BF16`.

**Turn it off** (the container then runs with every cold expert on the GPU):

```bash
echo B70_MOE_CPU=0 >> .env && docker compose up -d     # stays off; remove the line (or set 1) to turn it back on
B70_MOE_CPU=0 docker compose up -d                     # or for this start only (the shell wins over .env)
B70_MOE_CPU=0 engines/two-cards/scripts/run-tp2.sh     # without Compose
```

`B70_MOE_CPU` from `.env` or the shell wins over `engines/two-cards/env/tp2-local.env`, which wins over the defaults in
`tp2-hotcold.env`.

Core layout matters on other CPUs: each card's pool pins 8 threads, card 0 from core 0, card 1 from core 8
(`B70_MOE_CPU_FIRST_CORE`, default 8 × rank; one CCD each on a 16-core Ryzen). With fewer than 16 physical cores the
pools land on SMT siblings or share cores with each other and with vLLM. Measure `B70_MOE_CPU=1` against `=0` with
`bench_det.py` on such a host, and size the pools to the CPU in `tp2-local.env`, e.g. on a 12-core CPU
`B70_MOE_CPU_THREADS=6` and `B70_MOE_CPU_FIRST_CORE=0,6` (one first core per card). Details: docs/switches.md.

## What the engine runs

Weights: [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)
(Intel AutoRound target tensors, FP8 PLE table, INT4 MTP draft).

Settings (`engines/two-cards/env/tp2-hotcold.env`, `engines/two-cards/args/serve-tp2.args`): tensor parallel 2 + expert
parallel 2; 146 hot experts per layer and card in VRAM that follow what decode reads (0040, 0043, 0047), the rest
pinned in host RAM; cold experts prefetched into VRAM for prefill (0042); the fused decode MoE op with the CPU path
(0048, 0049) and an FP8 draft head; the input embedding in host RAM (0044); FP8 W8A16 dense projections (0045); the
checkpoint's FP8 PLE shards read in place from NVMe with a 2 GiB row cache (0013 + 0050); MTP k=3; GPU memory 0.96;
4 sequences; a warm-up request at start (known issues item 13). Without Compose:
`engines/two-cards/scripts/run-tp2.sh` / `restart-tp2.sh`. Experiments and numbers: [docs/two-cards.md](docs/two-cards.md).

## What is in the series

The image applies the vLLM release `0.30.0-b70.2` (patches 0001–0032, fork tag `Lumnus/vllm@v0.30.0-b70.2`) and the
two-card patches 0040–0050, 0052 and 0053 on top.

| patch | what | switch |
|---|---|---|
| 0001–0005 | wu1ff's pack (XPU gate, pinned-host PLE, HC K-split, GDN spec fusion, GDN index64 hook) | always on |
| 0006 | load the PLE table straight into pinned memory | `B70_PLE_DIRECT_PINNED=1` |
| 0007 | PLE table in FP8 | `B70_PLE_FP8=1` |
| 0008 | PLE table in INT8 with a per-row scale | `B70_PLE_INT8=1` |
| 0009 | per-effort thinking budgets; server default presence penalty | `B70_THINKING_BUDGET`, `B70_DEFAULT_PRESENCE_PENALTY` |
| 0010 | server default repetition stop (bounds the token-1023 NaN loop) | `B70_DEFAULT_REPETITION_DETECTION` |
| 0012 | accept OpenRouter-style `{"reasoning":{"effort":…}}` | `B70_REASONING_EFFORT_ALIAS=1` |
| 0013, 0013b | serve the INT8 PLE table from NVMe | `B70_PLE_INT8_NVME=1` |
| 0014a–f | hybrid-model KV-offload fixes (CPU KV tier; not used on two cards) | `B70_OFFLOAD_*` |
| 0018 | chunked pinned CPU KV pool (CPU KV tier; not used on two cards) | ungated, acts only then |
| 0019, 0020 | dense-QSA configs skip `self_attn.indexer` tensors | ungated |
| 0021 | MTP draft prefill with FULL graphs only on XPU | `B70_MTP_DRAFT_PREFILL_NO_PIECEWISE=1` |
| 0022 | port of vllm-project/vllm#55506 (Mamba spec-decode block tables by request slot) | ungated |
| 0023, 0023c/0024t | debug: GDN op mode switch, NaN tracers, a null-block probe | off by default |
| 0026 | MTP: never schedule a partial draft list on XPU | `B70_MTP_UNIFORM_DRAFTS=0` turns it off |
| 0027, 0032 | wu1ff's `libgdn_index64.so` optional | automatic; `B70_GDN_MODE` |
| 0028 | the PLE embedding accepts Intel's AutoRound (INC) quantization config | ungated |
| 0029, 0029b | prompt logprobs in 128-row chunks | `B70_PROMPT_LOGPROBS_CHUNK` |
| 0031 | CPU→GPU KV loads straight from the pinned pool (CPU KV tier; not used on two cards) | `B70_OFFLOAD_H2D_DIRECT=1` |
| 0040 | routed experts split between VRAM (hot) and pinned host RAM (cold) | `B70_MOE_HOTCOLD=1`, `B70_MOE_HOT_EXPERTS`, `B70_MOE_HOTCOLD_RANKINGS` |
| 0041 | route counter per (layer, expert) | `B70_MOE_ROUTE_COUNT` |
| 0042 | prefill runs cold experts from two prefetched VRAM slots | `B70_MOE_COLD_PREFETCH=1` |
| 0043 | the hot set follows the traffic (expert swapping) | `B70_MOE_HOTCOLD_SWAP=1` |
| 0044 | input embedding in pinned host RAM; swap defaults | `B70_EMBED_UVA=1` |
| 0045 | 16-bit dense projections as FP8 W8A16 | `B70_DENSE_FP8=<regex>` |
| 0046 | debug: per-step decode route trace | `B70_MOE_ROUTE_TRACE=<path>` |
| 0047 | rank experts by what decode reads; long-term state on the work volume | `B70_MOE_HOTCOLD_SWAP_SOURCE=decode`, `B70_MOE_HOTCOLD_STATE` |
| 0048 | fused decode MoE op (SYCL, `kernels/b70-moe-decode`); FP8 draft head | `B70_MOE_DECODE_OP=1`, `B70_DRAFT_HEAD_FP8=1` |
| 0049 | the CPU computes part of the cold experts at decode | `B70_MOE_CPU=1`, `B70_MOE_CPU_PCT` |
| 0050 | the PLE table read in place from the checkpoint's FP8 shards on NVMe | `B70_PLE_FP8_NVME=1`, `B70_PLE_FP8_PATH` |

Every switch with its default and measurements: [docs/switches.md](docs/switches.md).

## Where to find what

| path | what |
|---|---|
| `HOW-TO-RUN.md` | step-by-step guide |
| `compose.yaml`, `.env.example` | the engine with Docker Compose (engine, warm-up, `prepare`) |
| `engines/two-cards/` | `env/tp2-hotcold.env` (defaults), `env/tp2-local.env.example` (your overrides), `args/serve-tp2.args` (`vllm serve` flags), `scripts/` (compose entrypoint, `run-tp2.sh`, `restart-tp2.sh`) |
| `engines/serve-configs/` | the snapshots' `config.json` (`serve-config-intel-autoround.json` is the one the engine uses) |
| `image/` | `Dockerfile.tp2` (the image), `verify-overlay.sh` and `verify-tree.py` (hash checks of every patched file) |
| `patches/vllm/`, `patches/vllm-tests/` | the vLLM series 0001–0050, 0052 and 0053; CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | our kernel series on vllm-xpu-kernels 0.1.14.1 (b70.3) ([docs/kernels.md](docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file |
| `kernels/` | standalone SYCL ops (`b70-moe-decode`, built by the image) and diagnostics |
| `scripts/` | `prepare-models-tp2.sh` (weights), `make-tree.sh`, `check-series.sh`, `export-series.sh` |
| `tools/` | snapshot and PLE-table builders; `bench_tp2/` client benchmarks |
| `docs/` | [two cards](docs/two-cards.md) · [switches](docs/switches.md) · [known issues](docs/known-issues.md) · [PLE on NVMe](docs/ple-nvme.md) · [PLE INT8](docs/ple-int8.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [offload fix](docs/offload-fix.md) |

`PROVENANCE.md` records where every piece comes from, with hashes. `CHANGELOG.md` lists changes.

## Credits

This work stands on other people's:

1. **Lumnus** — [b70-flash-next](https://github.com/Lumnus/b70-flash-next) and the
   [vllm](https://github.com/Lumnus/vllm) fork.
2. **albucino** — the [`Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)
   weights.
3. **wu1ff — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller)** (MIT). The Qwen3.8-Flash-Next B70
   pack: the XPU gate, the pinned-host PLE table, the HC down-GEMM K-split, the GDN spec-metadata fusion, the 64-bit
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. Patches 0001–0005 are
   wu1ff's changes re-derived as source diffs; the residency shim is copied from wu1ff's public image by digest.
4. **Intel** — the [`Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
   target tensors, made with [AutoRound](https://github.com/intel/auto-round);
   [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels), whose upstream fixes #600, #563, #564, #578 and
   #586 are in our kernel build (by Guancheng Fu, Chaojun Zhang, Qiming Zhang and Tony Lin);
   [llm-scaler](https://github.com/intel/llm-scaler) and the XPU work in
   [vLLM #55068](https://github.com/vllm-project/vllm/pull/55068).
5. **devan-carlin** — the early community vLLM port of the model to XPU (`devan-carlin/vllm@xpu-qwen4exp`) and the
   BF16 PLE table.
6. **TSUMUGI-XE** — [intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968), the analysis of
   Level Zero peer residency mirroring device memory into host memory, and the first shim that works around it.
7. **The Qwen team** — Qwen3.8-Flash-Next itself (license: qwen-community-1.0, see each checkpoint's card).
8. **[vLLM](https://github.com/vllm-project/vllm)**, on which everything here is built. Patch 0014e is a backport of
   vllm-project/vllm#51787; patch 0022 ports vllm-project/vllm#55506.

## License

Apache-2.0 (LICENSE). wu1ff's binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify
Apache-2.0 code. Model weights are not included and keep their own licenses. See NOTICE.
