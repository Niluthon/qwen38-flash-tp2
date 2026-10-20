# b70-moe-decode: fused hot+cold INT4 MoE for decode (patch 0048)

One SYCL op, `torch.ops.b70moe.moe_decode`, for batches of at most 16 tokens. It runs the hot experts (in VRAM) and
the cold experts (pinned host RAM, read over PCIe through UVA) together in three kernels: up projection with
silu·mul, down projection with the route weight, then a gather. Each expert's weights are read once per layer. The
stock path makes two `XpuFusedMoe` calls per layer, and each call costs about 56 µs before it reads any weight byte.
Patch 0048 calls this op when `B70_MOE_DECODE_OP` is set (docs/switches.md); the two-card engine sets it to `1`.

The image builds the library itself: stage `b70moe` of `image/Dockerfile.tp2` installs icpx 2026.0 (the runtime's
SYCL version) from Intel's apt repository and runs `compile.sh` against the image's torch. For development without a
rebuild:

    kernels/b70-moe-decode/build.sh     # -> build/libb70_moe_decode.so
    cp kernels/b70-moe-decode/build/libb70_moe_decode.so engine-data/
    # engines/two-cards/env/tp2-local.env:  B70_MOE_DECODE_OP=/work/libb70_moe_decode.so

`build.sh` runs `compile.sh` in a container that has oneAPI 2026.0 and torch 2.13.0+xpu (`COMPILE_IMAGE`, default
`k8v4-compile-2026:latest`; `COMPILE_SETVARS` = its setvars.sh).

Check correctness and timing on one card. The script compares against the stock path, runs 48 layers in an XPU
graph, and takes the cold share as its argument:

    docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --group-add render \
      --ulimit memlock=-1 -e ZE_AFFINITY_MASK=0 -v $PWD:/r -w /r --entrypoint /opt/venv/bin/python \
      b70-flash-next:tp2 kernels/b70-moe-decode/test_moe_decode.py kernels/b70-moe-decode/build/libb70_moe_decode.so 0.08

Measured on one B70, per layer (stock → this op): 1 row 150 → 66 µs, 4 rows 391 → 241 µs, 8 rows 603 → 397 µs,
16 rows 1030 → 740 µs. Relative L2 against stock is about 8e-3 (bf16 rounding).

`test_draft_head.py` checks the FP8 draft head of `B70_DRAFT_HEAD_FP8` on the real lm_head shard. Run it with
`PYTHONPATH` set to a tree with 0048, and the models directory mounted at `/models`.

## MoE glue ops (0051, `B70_MOE_GLUE`)

- `moe_decode_counted(..., counts)`: moe_decode plus the 0047 route count in the plan kernel
  (`counts[id mod len] += 1` per route slot; replaces remainder + `index_add_`, ~12 µs per layer at 4 rows).
- `topk_softmax(topk_weights, topk_ids, gating_output, renormalize, is_padding=None)`: the stock `_moe_C` op's
  semantics (ties to the lower index, padding rows -1, rows with a NaN/+inf logit or only -inf logits: ids 0, NaN
  weights), 6.3 µs vs 11.9 per call. Row validity is decided on the raw bf16 bits: this library builds with
  `-ffast-math`, which removes float NaN tests.

`test_glue.py <lib.so>` (one card, engine stopped) checks both against the stock ops and times them; the vLLM side
is `vllm-0051-moe-glue.patch` (apply to a dev tree; not in the image's patch list).
