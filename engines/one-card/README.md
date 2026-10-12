# One card: 1× Arc Pro B70 + 96 GB RAM

> **These settings are not tested.** They have never run on a card; the numbers are estimates.

The two-card engine (`engines/two-cards/`) cut down to one card: tensor parallel 1, no expert parallel, the same
image, weights and patches. The numbers below are estimated from the two-card engine's boot log; tune them with the
boot log (below).

| file | what |
|---|---|
| `env/tp1-hotcold.env` | the environment: one card (`ZE_AFFINITY_MASK=0`), 96 hot experts, `SERVE_ARGS_FILE` → `serve-tp1.args`; everything else as two cards |
| `env/tp1-local.env.example` | template for your git-ignored `env/tp1-local.env` |
| `args/serve-tp1.args` | `vllm serve` flags: `--tensor-parallel-size 1`, no `--enable-expert-parallel`, `--max-model-len 65536`; the rest (MTP k=3, 4 sequences, util 0.95) as two cards |

## Use it

On the one-card PC, follow [HOW-TO-RUN.md](../../HOW-TO-RUN.md) (build, `prepare`, start), with these two lines in
the root `.env` before starting:

```bash
ENGINE_ENV=engines/one-card/env/tp1-hotcold.env
ENGINE_LOCAL_ENV=engines/one-card/env/tp1-local.env
```

## Sizing (estimates)

All 512 routed experts per layer belong to the one card. Each expert (all 48 layers) is 116 MiB, in VRAM (hot) or
pinned host RAM (cold).

| VRAM, at `--gpu-memory-utilization 0.95` (~28.8 GiB usable) | GiB |
|---|---|
| non-expert weights incl. the MTP draft (2 × the two-card per-card 5.3 GiB) | ~10.6 |
| activations, graphs, profiling peak (as two cards) | ~3.0 |
| cold-expert prefetch slots (2 layers × 416 cold × 2.42 MiB) | ~2.0 |
| KV cache for one 65,536-token context (KV per token is 2× two cards' per-card) | ~1.9 |
| **96 hot experts** | **~11.1** |

| host RAM (96 GB = ~89 GiB) | GiB |
|---|---|
| 416 cold experts, pinned | ~47.1 |
| PLE row cache 2, input embedding 1.2 (pinned) | ~3.2 |
| engine processes, OS | ~10–12 |
| **left** | **~27** |

So VRAM, not RAM, sets the hot count: RAM would also take every expert at 0 hot.

## First boot: what to check

`docker compose up -d && docker compose logs -f engine`, then in the log:

1. `B70 hot/cold: 48 layers split (96 hot in VRAM, 416 cold)`: the split works without expert parallel (the code
   path the two-card engine never used; if it fails here, the error is in patch 0040).
2. `Model loading took … GiB`: the real weight size; ~10.6 + 11.1 expected.
3. `KV cache size: … tokens`: must cover `--max-model-len`. If vLLM refuses to start for lack of KV memory, lower
   `B70_MOE_HOT_EXPERTS` or `--max-model-len` (in `env/tp1-local.env`). If there is plenty, raise the hot count:
   every 116 MiB of spare KV is one more hot expert.
4. Host RAM: `free -g` after start should leave ≥ 15 GiB available.

Expect slower decode than two cards: ~19 % of experts in VRAM (two cards: ~59 %) and every cold read over one PCIe
link. Turning MTP off (`serve-tp1.local.args` without `--speculative-config`) frees ~3.9 GiB = ~33 more hot experts;
which is faster has to be measured.
