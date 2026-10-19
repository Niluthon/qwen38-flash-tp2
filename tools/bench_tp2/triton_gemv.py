"""Triton skinny GEMM (M <= 16) prototype for B70 decode: y[M,N] = x[M,K] @ W, W either FP8 E4M3 [K,N] with a
per-column fp32 scale (the fp8_gemm_w8a16 layout of 0045) or BF16 [N,K] (F.linear layout). Split-K with fp32
atomics into a zeroed buffer, then a cast. Compared against the stock kernels on 48 distinct weights per shape."""
import sys, time, torch
import torch.nn.functional as F
import triton, triton.language as tl
import vllm_xpu_kernels._xpu_C  # noqa: F401
d = torch.device("xpu"); PEAK = 608.0; L = 48


@triton.jit
def gemv_fp8_kn(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, K_PER: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = pid_k * K_PER
    for kk in range(0, K_PER, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, w.to(tl.bfloat16))
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * s[None, :]
    tl.atomic_add(y_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def gemv_bf16_nk(x_ptr, w_ptr, y_ptr, M, N, K, K_PER: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = pid_k * K_PER
    for kk in range(0, K_PER, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
    tl.atomic_add(y_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def run(kind, x, w, s, y32, cfg):
    M, K = x.shape
    N = w.shape[1] if kind == "fp8" else w.shape[0]
    BN, BK, SK, warps = cfg
    k_per = triton.cdiv(K, SK)
    k_per = triton.cdiv(k_per, BK) * BK
    grid = (triton.cdiv(N, BN), triton.cdiv(K, k_per))
    y32.zero_()
    if kind == "fp8":
        gemv_fp8_kn[grid](x, w, s, y32, M, N, K, k_per, BM=16, BN=BN, BK=BK, num_warps=warps)
    else:
        gemv_bf16_nk[grid](x, w, y32, M, N, K, k_per, BM=16, BN=BN, BK=BK, num_warps=warps)
    return y32.to(torch.bfloat16)


def graph_time(fn, n=20):
    s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s): fn()
    g.replay(); torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): g.replay()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n


SHAPES = [("GDN in_proj_qkv", 5120, 2560, "fp8"), ("GDN in_proj_z", 3072, 2560, "fp8"), ("GDN out_proj", 2560, 3072, "fp8"),
          ("attn q_proj", 6144, 2560, "fp8"), ("HC down+inject", 336, 10240, "bf16"), ("HC up", 10240, 320, "bf16"),
          ("router", 512, 2560, "bf16"), ("lm_head shard fp8", 124160, 2560, "fp8")]
CFGS = [(64, 128, 1, 4), (128, 128, 1, 4), (64, 256, 1, 4), (128, 64, 2, 4), (64, 128, 2, 4), (64, 128, 4, 4),
        (32, 128, 4, 4), (32, 256, 4, 4), (256, 64, 1, 8), (128, 128, 4, 8), (32, 128, 8, 4), (16, 256, 8, 4)]
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
only = sys.argv[2] if len(sys.argv) > 2 else ""
for name, N, K, kind in SHAPES:
    if only and only not in name: continue
    n_w = 1 if "lm_head" in name else L
    x = torch.randn(M, K, device=d, dtype=torch.bfloat16)
    if kind == "fp8":
        ws = [(torch.randn(K, N, device=d) * 0.05).to(torch.float8_e4m3fn) for _ in range(n_w)]
        s = torch.rand(1, N, device=d, dtype=torch.float32) * 0.01 + 0.001
        ref_fn = lambda: [torch.ops._xpu_C.fp8_gemm_w8a16(x, w, s, None) for w in ws]
        nbytes = N * K * n_w
    else:
        ws = [torch.randn(N, K, device=d, dtype=torch.bfloat16) * 0.05 for _ in range(n_w)]; s = None
        ref_fn = lambda: [F.linear(x, w) for w in ws]
        nbytes = N * K * 2 * n_w
    t_ref = graph_time(ref_fn)
    ref = ref_fn()[0].float()
    y32s = [torch.empty(M, N, device=d, dtype=torch.float32) for _ in range(n_w)]
    best = None
    for cfg in CFGS:
        try:
            out = run(kind, x, ws[0], s.view(-1) if s is not None else None, y32s[0], cfg).float()
            err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            t = graph_time(lambda: [run(kind, x, w, s.view(-1) if s is not None else None, y, cfg) for w, y in zip(ws, y32s)])
        except Exception as e:
            print(f"   cfg {cfg}: {type(e).__name__} {str(e)[:120]}"); continue
        if best is None or t < best[0]: best = (t, cfg, err)
    t, cfg, err = best
    print(f"{name:20s} {kind} M={M}: stock {t_ref/n_w*1e6:7.1f} us ({nbytes/t_ref/1e9:4.0f} GB/s) | triton {t/n_w*1e6:7.1f} us "
          f"({nbytes/t/1e9:4.0f} GB/s, {100*nbytes/t/1e9/PEAK:3.0f}%) cfg BN,BK,SK,warps={cfg} rel.err {err:.1e}", flush=True)
    del ws; torch.xpu.empty_cache()
