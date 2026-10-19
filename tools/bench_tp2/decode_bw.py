"""Decode-step component microbench on one B70 (engine stopped): bandwidth efficiency of the kernels the decode
graph runs, at the real shapes of Qwen3.8-Flash-Next TP2 (per card), 48 distinct weight copies per shape so nothing
is served from L2, timed as XPU graph replays. Prints us per call, GB/s of weight bytes, % of 608 GB/s."""
import sys, time, torch
import torch.nn.functional as F
import vllm_xpu_kernels._xpu_C  # noqa: F401  registers torch.ops._xpu_C
d = torch.device("xpu")
PEAK = 608.0
L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
ROWS = [int(r) for r in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["4", "16"])]

def graph_time(fn, n=20):
    s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s):
            fn()
    g.replay(); torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(n): g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / n

def report(name, sec, nbytes, calls):
    us = sec / calls * 1e6; gbs = nbytes / sec / 1e9
    print(f"{name:44s} {us:8.1f} us/call  {gbs:6.0f} GB/s  {100*gbs/PEAK:4.0f}%   ({nbytes/calls/1e6:.1f} MB/call)", flush=True)

# 1. roofline: read-only reduction and copy
big = torch.rand((2 << 30) // 4, device=d)  # random: B70 compresses constant data
out = torch.empty(1, device=d)

report("roofline: sum of 2 GiB (read)", graph_time(lambda: torch.sum(big, dim=0, out=out[0])), big.numel() * 4, 1)
half = big[: big.numel() // 2]; other = big[big.numel() // 2 :]
report("roofline: copy 1 GiB (read+write)", graph_time(lambda: other.copy_(half)), half.numel() * 8, 1)
del big, half, other; torch.xpu.empty_cache()

def dense_set(N, K, dtype):
    if dtype == "bf16":
        return [torch.randn(N, K, device=d, dtype=torch.bfloat16) * 0.02 for _ in range(L)]
    return [(torch.randn(K, N, device=d) * 0.02).to(torch.float8_e4m3fn) for _ in range(L)]

SHAPES = [  # (name, N_out, K_in) per card
    ("GDN in_proj_qkv (TP2)", 5120, 2560), ("GDN in_proj_z (TP2)", 3072, 2560), ("GDN out_proj (TP2)", 2560, 3072),
    ("attn q_proj (TP2)", 6144, 2560), ("attn o_proj (TP2)", 2560, 3072),
    ("HC down+inject (replicated)", 336, 10240), ("HC up (replicated)", 10240, 320),
    ("shared expert gate/up (TP2?)", 640, 2560), ("shared expert down", 2560, 320),
    ("router gate (replicated)", 512, 2560),
]
for M in ROWS:
    print(f"\n=== M = {M} rows ===", flush=True)
    for name, N, K in SHAPES:
        x = torch.randn(M, K, device=d, dtype=torch.bfloat16)
        wb = dense_set(N, K, "bf16")
        report(f"{name} bf16", graph_time(lambda: [F.linear(x, w) for w in wb]), N * K * 2 * L, L)
        del wb
        w8 = dense_set(N, K, "fp8"); s8 = torch.rand(1, N, device=d, dtype=torch.float32)
        try:
            report(f"{name} fp8 w8a16", graph_time(lambda: [torch.ops._xpu_C.fp8_gemm_w8a16(x, w, s8, None) for w in w8]),
                   N * K * L, L)
        except Exception as e:
            print(f"{name} fp8: {str(e)[:100]}")
        del w8; torch.xpu.empty_cache()
    # lm_head: vocab 248320 / 2 per card
    x = torch.randn(M, 2560, device=d, dtype=torch.bfloat16)
    W = torch.randn(124160, 2560, device=d, dtype=torch.bfloat16) * 0.02
    report("lm_head shard bf16 (124160x2560)", graph_time(lambda: F.linear(x, W)), W.numel() * 2, 1)
    del W; torch.xpu.empty_cache()
    W8 = (torch.randn(2560, 124160, device=d) * 0.02).to(torch.float8_e4m3fn); S8 = torch.rand(1, 124160, device=d)
    report("lm_head shard fp8 w8a16", graph_time(lambda: torch.ops._xpu_C.fp8_gemm_w8a16(x, W8, S8, None)), W8.numel(), 1)
    del W8; torch.xpu.empty_cache()
