# Engines

The engine configuration for 2× Arc Pro B70 and ~64 GB RAM (`tp2-hotcold`): environment files, `vllm serve` flag
files and launch scripts. How to run it: [HOW-TO-RUN.md](../HOW-TO-RUN.md).

```
engines/
  serve-configs/   the snapshots' config.json: serve-config-intel-autoround.json (what the engine uses);
                   serve-config-awq.json is only the input of `tools/intel_snapshot.py serve-config`
  two-cards/
    env/             tp2-hotcold.env (defaults), tp2-local.env.example (template for your git-ignored tp2-local.env)
    args/            serve-tp2.args (vllm serve flags; your optional serve-tp2.local.args goes here too)
    scripts/         compose-serve.sh (compose entrypoint), run-tp2.sh / restart-tp2.sh (without compose)
```

| engine | weights | slots | PLE table | CPU KV tier | what it runs |
|---|---|---|---|---|---|
| `tp2-hotcold` (`docker compose up -d`, or `run-tp2.sh`) | albucino FP8-PLE (Intel AutoRound target) | 4 | the checkpoint's FP8 shards, read in place from NVMe (0050) | none | TP 2 + EP 2, 146 hot experts per layer in VRAM following decode traffic, the rest pinned in RAM, fused decode MoE op with 70 % of the cold experts on the CPU, FP8 dense projections, MTP k=3, 262K context, warm-up at start ([docs/two-cards.md](../docs/two-cards.md)) |

## Compose

`compose.yaml` at the repository root runs `tp2-hotcold`: `docker compose up -d` starts it detached, and the separate
`warmup` service sends the short first request (it waits for `/health` itself, then exits 0 — no `depends_on`, none of
the engine's devices or volumes, so `up -d` and `restart engine warmup` never race the engine; `WARMUP_TIMEOUT`,
1200 s by default). `docker compose logs -f engine` follows the log (also in `engine-data/engine.log`),
`docker compose down` stops it. Defaults come from `tp2-hotcold.env` and `serve-tp2.args`
(shared with `run-tp2.sh`); put local changes in `tp2-local.env` (git-ignored; template `tp2-local.env.example`):
environment switches override the defaults, and `EXTRA_ARGS` appends `vllm serve` flags (the last value of a flag wins).
Host paths (`MODELS`, `ENGINE_DATA`) and the image go in a root `.env` (template `.env.example`). `compose-serve.sh` is the
services' entrypoint (`serve`, `health`, `warmup`).

## Notes

1. **Chat template.** The engine uses the checkpoint's own template.
2. **Liveness.** Probe with a 1-token chat completion; `/health` and `/v1/models` stay 200 while the engine is wedged.
   Allow ≥180 s under load: a busy engine queues, it is not down.
3. **compute-runtime.** The image does not change the host driver.
4. **GPU hangs.** If a card keeps timing out driver jobs after a GT reset, see
   [docs/known-issues.md](../docs/known-issues.md) (rebind the card, not an FLR).
5. **CPU.** The CPU path for cold experts needs AVX-512 BF16 and by default pins 8 threads per card from cores 0 and 8;
   on other CPUs see [README, CPU](../README.md#cpu).
