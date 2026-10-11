import time, torch
d = torch.device("xpu"); K, N = 2560, 6144  # a GDN in_proj-sized shard
def t(fn, n=300):
    for _ in range(10): fn()
    torch.xpu.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - s) / n * 1e6
Wb = torch.randn(N, K, device=d, dtype=torch.bfloat16)
W8 = torch.randn(K, N, device=d).to(torch.float8_e4m3fn); S8 = torch.rand(1, N, device=d, dtype=torch.float32)
Wq = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=d); Sq = torch.rand(K // 128, N, device=d, dtype=torch.bfloat16)
Z = torch.tensor([8], dtype=torch.int8, device=d)
for M in (1, 4, 16, 512):
    x = torch.randn(M, K, device=d, dtype=torch.bfloat16)
    b = t(lambda: torch.nn.functional.linear(x, Wb))
    try: f = t(lambda: torch.ops._xpu_C.fp8_gemm_w8a16(x, W8, S8, None))
    except Exception as e: f = float("nan"); print("fp8 err", str(e)[:100])
    i4 = float("nan")
    for wq in (Wq.t(), Wq):
        try: i4 = t(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x, wq, None, Sq, Z, 128, None)); break
        except Exception as e: err = str(e)[:100]
    print(f"M={M:4d}: bf16 {b:7.1f} us ({N*K*2/b/1e3:4.0f} GB/s)  fp8 w8a16 {f:7.1f} us  int4 w4a16 {i4:7.1f} us")
