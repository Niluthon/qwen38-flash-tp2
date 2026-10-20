# b70-step-trace: where a decode step's time goes, on the running two-card engine

Measurement only. Gives per-module GPU time of the real decode step (XPU graphs, both ranks), which no profiler
gives us: graph replays are one opaque launch for unitrace, unitrace hangs the two-card engine as soon as tracing is
resumed (even with the 0049 CPU path off), and the torch profiler crashes the engine.

How it works: a mark is a single-work-item kernel that bumps a device counter, writes a 32-bit code into a ring in
host USM and publishes the counter (system-scope release, the 0049 mailbox pattern). A pinned host thread spins on
the counter and stamps every new entry with CLOCK_MONOTONIC. The B70 has no device-scope clock
(`ext_oneapi_clock_device` is missing; the sub-group clock is per EU and useless across kernels), hence host stamps.
Marks are captured into the graphs like any op, and the queue is in order, so the time between two stamps is the GPU
time of the work between them plus ~1.35 µs (one mark). The GPU->host latency is the same for every mark and cancels.

Card check (`test_step_trace.py`, one B70): a 1 GiB copy between two marks reads 3.949 ms, XPU events 3.970 ms;
1.35 µs median per mark inside a graph; 5,020 of 5,020 records, none lost.

## Use

    kernels/b70-step-trace/build.sh                       # -> build/libb70_step_trace.so (oneAPI 2026.0 container)
    cp kernels/b70-step-trace/build/libb70_step_trace.so engine-data/
    cd engine-data/dev-tree && git apply ../../kernels/b70-step-trace/vllm-step-trace.patch   # once
    # engines/two-cards/env/tp2-local.env:
    #   PYTHONPATH=/work/dev-tree
    #   B70_STEP_TRACE=/work/libb70_step_trace.so
    #   B70_STEP_TRACE_DIR=/work/step-trace/<run>    B70_STEP_TRACE_EVERY=128    B70_STEP_TRACE_DEPTH=2
    #   B70_STEP_TRACE_WRAP=<module:Class.method,...>  (optional: marks around non-forward methods)
    #   B70_STEP_TRACE_OPS=<module:Class.method,...>   (optional: also a mark around every op those methods
    #                                                   dispatch; ~+10 ms per step on the MoE block)
    docker compose up -d --force-recreate engine warmup
    python3 tools/bench_tp2/bench_det.py 400               # or any load
    docker run --rm -v $PWD:/r -w /r --entrypoint /opt/venv/bin/python b70-flash-next:tp2 \
      tools/bench_tp2/step_trace_report.py engine-data/step-trace/<run> --rank 0 --tokens 4,16

The patch (`vllm-step-trace.patch`) adds `vllm/model_executor/b70_step_trace.py` and two calls in
`vllm/v1/worker/gpu/model_runner.py` (install after load_model, a dump check per execute_model). Hooks: begin per
forward of the target and the draft model, marks at start/end of every module down to depth 2 below a decoder layer,
before/after every TP all-reduce. Pollers sit on cores 23 and 31 (SMT siblings of a CPU-pool core of each rank;
`B70_STEP_TRACE_CORE_<dev>`).

Overhead: ~1,700 marks per 1-stream step, +1.5–2 ms per step (bench_det 34.4–37.7 vs 32.6–36.0 ms/step without).
Each module's self time contains one mark (~1.35 µs), which matters only for the smallest modules.

The MoE wrap list used in round 11:

    vllm.model_executor.layers.fused_moe.router.fused_moe_router:FusedMoERouter.select_experts,
    vllm.model_executor.layers.fused_moe.modular_kernel:FusedMoEKernelModularImpl._prepare,
    vllm.model_executor.layers.fused_moe.modular_kernel:FusedMoEKernelModularImpl._fused_experts,
    vllm.model_executor.layers.fused_moe.modular_kernel:FusedMoEKernelModularImpl._finalize,
    vllm.model_executor.layers.fused_moe.experts.xpu_moe:XPUExpertsWNA16.b70_apply_hotcold,
    vllm.model_executor.layers.fused_moe.b70_hotcold:HotColdPlan.apply_decode_op
