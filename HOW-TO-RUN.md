# How to run it (two cards)

A step-by-step guide to serving Qwen3.8-Flash-Next on **2× Intel Arc Pro B70** with Docker Compose. It covers a fresh
machine, daily use, reusing weights you already have, and bringing your own model. This repository has only the
two-card setup. Speed of the current build: [README](README.md#speed-of-the-current-build).

**Short version** (each step is explained below):

```bash
git clone https://github.com/Niluthon/qwen38-flash-tp2 && cd qwen38-flash-tp2
docker compose build                 # ~20 GB image, builds everything itself
docker compose run --rm prepare      # once: ~129 GB download to ~/models (set MODELS in .env to change)
docker compose up -d                 # ~5 min until it serves on port 8000
docker compose logs -f warmup        # waits for the engine, one short request, then exits: "warm-up request OK"
```

**Contents:** [1. Check the machine](#1-check-the-machine) · [2. First start](#2-first-start) ·
[3. Use it](#3-use-it) · [4. Daily use](#4-daily-use) · [5. Change settings](#5-change-settings) ·
[6. Weights you already have](#6-weights-you-already-have) · [7. Your own model](#7-your-own-model) ·
[8. Troubleshooting](#8-troubleshooting)

## 1. Check the machine

| need | check with | expected |
|---|---|---|
| 2× Arc Pro B70 with the `xe` kernel driver (we run Linux 7.0) | `ls /dev/dri` | `card*` and `renderD*` entries, and a `by-path` directory |
| ~64 GB RAM | `free -g` | `total` around 62 or more |
| an x86-64 CPU; **AVX-512 BF16** for the fastest decode (optional) | `grep -o -m1 avx512_bf16 /proc/cpuinfo` | `avx512_bf16` (AMD Zen 4/5, Intel Xeon 4th gen+). Without it the engine runs ~16–19 % slower at decode; see [README, CPU](README.md#cpu) |
| 16 or more CPU cores for the CPU path's default pinning | `lscpu \| grep '^Core'` | `Core(s) per socket: 16` or more; with fewer, compare `B70_MOE_CPU=1` and `=0` (section 5) |
| ~185 GB free on an **NVMe** disk, for the weights (the PLE table is read from it while serving) | `df -h /path/for/models` | 185G or more available |
| ~20 GB free for Docker's image store | `df -h /var/lib/docker` | 20G or more available |
| Docker Engine with the Compose plugin **v2.24 or newer** | `docker compose version` | `v2.24.0` or higher |
| your user may run Docker | `docker run --rm hello-world` | "Hello from Docker!" without `sudo` |

No Docker yet: install Docker Engine from [docs.docker.com/engine/install](https://docs.docker.com/engine/install/)
(the Compose plugin comes with it), then `sudo usermod -aG docker $USER` and log in again.

Nothing else is installed on the host: vLLM, the patches, the GPU kernels, the Level Zero runtime and the weights all
come through the steps below. Internet access is needed for the build and the download (Docker Hub, ghcr.io,
github.com, huggingface.co); serving runs offline.

## 2. First start

```bash
# a. get the repository
git clone https://github.com/Niluthon/qwen38-flash-tp2 && cd qwen38-flash-tp2

# b. (optional) choose where the weights go; the default is ~/models
cp .env.example .env
nano .env                                  # e.g. MODELS=/mnt/nvme/models

# c. build the image: vLLM of release 0.30.0-b70.2 + two-card patches + kernels + shim, all checked (~20 GB image)
docker compose build

# d. download and prepare the weights, once: ~129 GB download, then the serve snapshots (seconds)
docker compose run --rm prepare

# e. start the engine (~5 minutes until it serves)
docker compose up -d                       # engine, plus the warmup service (its own service: it waits for the engine)
docker compose logs -f engine              # wait for "Application startup complete"; Ctrl-C only stops following
```

Every step can be repeated: `build` reuses what it already built, `prepare` skips finished work and resumes an
interrupted download.

**f. Check it.** `docker compose ps` lists `engine` as `healthy`, and `docker compose logs warmup` ends with
`warm-up request OK`. Wait for that line before sending long prompts (see [Troubleshooting](#8-troubleshooting)).
`warmup` is independent of `engine` (no `depends_on`, none of the engine's devices or volumes): it starts with the
engine, polls `/health` itself and exits 0 after that one request — so `docker compose up -d` and
`docker compose restart engine warmup` cannot fail it. It gives up after `WARMUP_TIMEOUT` seconds (1200 by default).

## 3. Use it

The engine serves an OpenAI-compatible API on port 8000 of this machine, model name `qwen38-flash-next`, no API key.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen38-flash-next",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}]
}'
```

- **Clients** (coding agents, chat UIs, SDKs): base URL `http://<this machine>:8000/v1`, any API key string.
- **Thinking** is on by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`,
  or limit it with `"reasoning_effort": "low"` (or the OpenRouter form `"reasoning": {"effort": "low"}`).
- **Tool calls** and **structured output** (`response_format` with a JSON schema) work.
- **Limits**: 262,144-token context, 4 requests at a time (more are queued), text only (no images).
- **Speed** on our host (tok/s = generated tokens per second):
  - one request: ~70–108 tok/s, depending on how well the MTP draft guesses the text (~31–36 ms per decode step);
  - four requests at once: ~177 tok/s in total (~62 ms per step);
  - reading the prompt (prefill): ~4,000 prompt tokens/s, so a 30K-token prompt takes ~7.5 s before the first token.

  Tables and method: [README](README.md#speed-of-the-current-build).
- The port is open on all network interfaces; firewall it if the machine is reachable from outside.

## 4. Daily use

| command | what it does |
|---|---|
| `docker compose up -d` | start, or apply changed settings (the engine restarts) |
| `docker compose down` | stop |
| `docker compose logs -f engine` | follow the log (also in `engine-data/engine.log`) |
| `docker compose ps` | state; `healthy` = serving |
| `git pull && docker compose build && docker compose up -d` | update to a newer version |

The engine does **not** restart by itself after a crash or a reboot; start it again with `docker compose up -d`.

## 5. Change settings

Keep tracked files unchanged; put your changes in git-ignored files, then run `docker compose up -d`:

| file | for | start from |
|---|---|---|
| `.env` | host paths (`MODELS`, `ENGINE_DATA`), image name, `HF_TOKEN`; `B70_MOE_CPU=0` turns the CPU path for cold experts off (wins over the env files) | `.env.example` |
| `engines/two-cards/env/tp2-local.env` | engine switches (hot experts, PLE cache, chat defaults, ...) and `EXTRA_ARGS`, extra `vllm serve` flags | `engines/two-cards/env/tp2-local.env.example` (each setting explained) |
| `engines/two-cards/args/serve-tp2.local.args` | the full flag list, when you need to **remove** a flag (`EXTRA_ARGS` can only add or override) | `cp engines/two-cards/args/serve-tp2.args engines/two-cards/args/serve-tp2.local.args`, then `SERVE_ARGS_FILE=/b70-engines/two-cards/args/serve-tp2.local.args` in `engines/two-cards/env/tp2-local.env` |

Examples for `engines/two-cards/env/tp2-local.env`:

```bash
EXTRA_ARGS=--max-model-len 65536                       # shorter context
EXTRA_ARGS=--port 8080 --served-model-name my-model    # another port / model name
B70_MOE_HOT_EXPERTS=154                                # more experts in VRAM: a bit faster, less KV cache
B70_MOE_CPU=0                                          # cold experts on the GPU only (CPUs with < 16 cores: measure both)
B70_MOE_CPU_THREADS=6                                  # smaller CPU pools, e.g. a 12-core CPU ...
B70_MOE_CPU_FIRST_CORE=0,6                             # ... with one first core per card
```

After anything that changes memory use, check the KV cache the engine reports at start:
`docker compose logs engine | grep "KV cache size"`. Every switch: [docs/switches.md](docs/switches.md).

## 6. Weights you already have

`prepare` expects these names inside the `MODELS` directory (set in `.env`; default `~/models`):

| path in `MODELS` | what | made by |
|---|---|---|
| `qwen38-w4a16/` | the checkpoint [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE), **all** files incl. `runtime/` (MTP draft, expert rankings) | download |
| `qwen38-w4a16-int8ple/ple_ngram_int8_rowscale.safetensors` | optional: the INT8 PLE table, only for the INT8 fallback (the engine reads the checkpoint's FP8 PLE shards in place, patch 0050) | `BUILD_INT8_PLE=1 docker compose run --rm prepare` |
| `qwen38-w4a16-snapshot/`, `qwen38-w4a16-mtp-snapshot/` | what the engine loads: symlinks into the checkpoint + `config.json` | made in seconds |

Everything must sit **inside** `MODELS` itself: the container sees only that directory, so a symlink pointing outside
it does not work.

**You downloaded the checkpoint with `hf download --local-dir` (or git clone).** Move or rename it to
`$MODELS/qwen38-w4a16` (a `mv` on the same disk is instant), then:

```bash
docker compose run --rm prepare                          # checks it against the pinned revision, fetches what is missing
SKIP_DOWNLOAD=1 docker compose run --rm prepare          # or: use it exactly as it is, no network
```

`SKIP_DOWNLOAD=1` is for a download of another revision that you want to keep; it only checks that `config.json` and
`runtime/mtp-int4-g32` are there.

**It is in the Hugging Face cache** (`~/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE`).
The cache is made of links into a blob store, so copy it with the links resolved (needs the space twice until you
delete the cache copy), then continue as above:

```bash
cp -rL ~/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/*/ "$MODELS/qwen38-w4a16"
```

**You already built the INT8 PLE table** (with `tools/build_int8_ple_from_fp8.py`). The engine no longer needs it
(0050 reads the FP8 shards in the checkpoint); delete it to free 52.5 GB, or keep it for the INT8 fallback
(docs/ple-nvme.md). With `BUILD_INT8_PLE=1`, `prepare` verifies a table found at
`$MODELS/qwen38-w4a16-int8ple/ple_ngram_int8_rowscale.safetensors` on a sample of rows instead of rebuilding it
(with its `.ok` file next to it, even that check is skipped).

**Old snapshots** made from a checkpoint at another path: delete them, `prepare` makes new ones.

**Your folders have other names and you do not want to rename them.** Skip `prepare` and point the engine at your
paths in `engines/two-cards/env/tp2-local.env` (container paths, `/models` = your `MODELS`). You then make the snapshots yourself with
`tools/fp8ple_snapshot.py` ([docs/two-cards.md](docs/two-cards.md#weights)).

```bash
MODEL_DIR=/models/<target snapshot>
B70_MOE_HOTCOLD_RANKINGS=/models/<checkpoint>/runtime/static_hot_cache_rankings.json
B70_PLE_INT8_PATH=/models/<table dir>/ple_ngram_int8_rowscale.safetensors
B70_PLE_INT8_NVME_PATH=/models/<table dir>/ple_ngram_int8_rowscale.safetensors
EXTRA_ARGS=--speculative-config '{"method":"mtp","num_speculative_tokens":3,"model":"/models/<draft snapshot>"}'
```

## 7. Your own model

**What can run here.** The engine and its patches are built for **Qwen3.8-Flash-Next**. On two cards the routed
experts (58 GiB) do not fit in VRAM, so the split between VRAM and host RAM (patch 0040) is required, and it works only
for checkpoints whose experts load through vLLM's GPTQ path: **Intel AutoRound (`quant_method: auto-round`) W4A16**.

| checkpoint | on two cards |
|---|---|
| [`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) | **the default**; tested |
| `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (the target tensors above, with a BF16 PLE table) | should work; **not tested on two cards** (recipe below) |
| your own AutoRound W4A16 quant of Qwen3.8-Flash-Next | as for Intel's; same quantization format |
| AWQ / llm-compressor (`compressed-tensors`, e.g. `wtdcode/...-AWQ-W4A16`) | **no**: the expert split does not cover this format and the experts do not fit |
| other model families | **no**: the settings and patches are for this model; use upstream vLLM |

**What the engine needs from a model**, whatever its source:

1. **A serve snapshot** (`MODEL_DIR`): the checkpoint's files, an index without the PLE table tensors and the unused
   `self_attn.indexer.*`, and `config.json` = the dense-QSA serve config with the checkpoint's own quantization block.
   `tools/fp8ple_snapshot.py` (FP8 PLE checkpoints) and `tools/intel_snapshot.py` (BF16 PLE checkpoints) make it;
   they refuse a checkpoint whose config differs in anything else.
2. **The PLE table.** A checkpoint with FP8 PLE shards needs nothing extra: `B70_PLE_FP8_PATH` = the checkpoint
   directory, read in place from NVMe (`B70_PLE_FP8=1` + `B70_PLE_FP8_NVME=1`, patch 0050). Otherwise an INT8 PLE
   table (`B70_PLE_INT8_PATH`, `B70_PLE_INT8_NVME_PATH`), built from the checkpoint's own PLE table:
   `tools/build_int8_ple_from_fp8.py` from FP8 shards, `tools/build_int8_ple.py` from a BF16 `.pt` table.
3. **Expert rankings** (`B70_MOE_HOTCOLD_RANKINGS`, optional): which experts start in VRAM. Leave it empty if your
   checkpoint has none; the hot set then starts by expert id and patch 0043 moves the busiest experts into VRAM as
   traffic comes in.
4. **An MTP draft** (`--speculative-config`, optional): a separate draft model (`"model": "/models/..."`), or no
   `model` key to use the checkpoint's own MTP layers (in BF16 they take more VRAM, so fewer hot experts or less KV
   cache), or no speculative decoding at all (remove the flag via `serve-tp2.local.args`, section 5; slower).

**Example: Intel's AutoRound checkpoint** (not tested on two cards). Run the commands
from the repository directory; they work inside the image, on your `MODELS` directory:

```bash
# 1. download: Intel's checkpoint without its 102 GB BF16 PLE shard, and devan's BF16 PLE table (the INT8 source)
#    into the same folder (the table builder reads layout tensors from the checkpoint next to it)
docker compose run --rm --entrypoint bash prepare -c '
  hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --revision 4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897 \
     --local-dir /models/intel-w4a16 --exclude model-00016-of-00017.safetensors
  hf download devan-carlin/Qwen3.8-Flash-Next-W4A16 ple_table_qwen4exp.pt \
     --revision 40b8f18df4d4a32cb6e687a51c78207e5e438522 --local-dir /models/intel-w4a16'

# 2. INT8 PLE table (reads the 95 GiB BF16 table once), snapshot, serve config
docker compose run --rm --entrypoint bash prepare -c '
  BF16_PT=/models/intel-w4a16/ple_table_qwen4exp.pt OUT_DIR=/models/intel-int8ple /opt/venv/bin/python /b70-tools/build_int8_ple.py build
  /opt/venv/bin/python /b70-tools/intel_snapshot.py snapshot /models/intel-w4a16 /models/intel-w4a16/ple_table_qwen4exp.pt /models/intel-snapshot
  cp /b70-engines/serve-configs/serve-config-intel-autoround.json /models/intel-snapshot/config.json'
```

Then in `engines/two-cards/env/tp2-local.env`:

```bash
MODEL_DIR=/models/intel-snapshot
B70_PLE_INT8_PATH=/models/intel-int8ple/ple_ngram_int8_rowscale.safetensors
B70_PLE_INT8_NVME_PATH=/models/intel-int8ple/ple_ngram_int8_rowscale.safetensors
B70_MOE_HOTCOLD_RANKINGS=
EXTRA_ARGS=--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`docker compose up -d`, then check the start-up log for `B70 hot/cold: 48 layers split`, `PLE NVMe host path ready`
and the `KV cache size`. If start-up runs out of VRAM, lower `B70_MOE_HOT_EXPERTS` (each expert is 116 MiB per card).
Intel's checkpoint is described in [docs/weights.md](docs/weights.md).

## 8. Troubleshooting

| symptom | cause / what to do |
|---|---|
| `docker compose build` fails while downloading | network or registry hiccup; run it again (finished layers are kept) |
| `env_file` or `required` error from compose | Compose older than v2.24; update Docker |
| `prepare` says `/models is not writable` | `MODELS` in `.env` must be a directory Docker can write to |
| `prepare` stops in the download | run it again; it resumes. Rate-limited by Hugging Face: set `HF_TOKEN` in `.env` |
| the engine stops right after the first request | the first request after a start was long (known issue 13). The `warmup` service sends a short one first; wait for `warm-up request OK` |
| `warmup` exits 1 with `warm-up: /health did not answer within 1200s` | the engine never became healthy in that window: read `docker compose logs engine` (section above). Give it longer with `WARMUP_TIMEOUT=3600 docker compose up -d warmup` |
| engine stays `starting` for over 15 min, or exits | read `docker compose logs engine`; check that nothing else uses the cards (the engine takes 95 % of their memory) and that `prepare` finished |
| `No such file` for a path under `/models` | the weights are not where the engine looks: section 6 |
| no `/dev/dri`, or `group render not found` | the GPU driver is not loaded on the host; check `ls /dev/dri` and the kernel log |
| decode slower than in the README: `python3 tools/bench_tp2/bench_det.py` shows ~36–40 ms/step (one request) instead of ~31–36 ms/step, or ~142 instead of ~180 tok/s for 4 requests | check `docker compose logs engine \| grep "CPU path"`: `CPU path on` for both ranks means it runs; `off (B70_MOE_CPU=…)` means it is switched off (`.env`, shell or `tp2-local.env`); `no AVX-512 BF16` means the CPU cannot run it (all cold experts stay on the GPU). On with fewer than 16 cores: size the pools (`B70_MOE_CPU_THREADS`, `B70_MOE_CPU_FIRST_CORE=0,6`, section 5) or compare with `B70_MOE_CPU=0` |

More: [docs/known-issues.md](docs/known-issues.md); details and measurements: [docs/two-cards.md](docs/two-cards.md).
