"""Microbench: the small MoE glue kernels at decode shapes, 48 per XPU graph (one B70, engine stopped).
Times router GEMM, _moe_C.topk_softmax, route counting (remainder + index_add_), alone and chained, to tell real
kernel time from fixed per-kernel costs.   usage: moe_glue_bench.py"""
import time, torch
import torch.nn.functional as F
import vllm._custom_ops as ops

d = torch.device("xpu")
L, E, K, TOPK = 48, 512, 2560, 10


def graph_time(fn, n=50):
    s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s):
            fn()
    g.replay(); torch.xpu.synchronize()
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n): g.replay()
        torch.xpu.synchronize(); ts.append((time.perf_counter() - t) / n)
    return sorted(ts)[2] / L * 1e6  # us per layer


Wg = [torch.randn(E, K, device=d).bfloat16() * 0.02 for _ in range(L)]
counts = torch.zeros(L, E + 1, dtype=torch.int64, device=d)
ones = torch.ones(4096, dtype=torch.int64, device=d)
empty_k = lambda: None
for M in (1, 4, 16):
    x = torch.randn(M, K, device=d).bfloat16()
    logits = [torch.randn(M, E, device=d).bfloat16() for _ in range(L)]
    tw = torch.empty(M, TOPK, dtype=torch.float32, device=d)
    ti = torch.empty(M, TOPK, dtype=torch.int32, device=d)
    tei = torch.empty(M, TOPK, dtype=torch.int32, device=d)

    def gate():
        for w in Wg: F.linear(x, w)

    def topk():
        for lg in logits: ops.topk_softmax(tw, ti, tei, lg, True)

    def count():
        for l in range(L):
            ids = ti.reshape(-1).remainder(E + 1)
            counts[l].index_add_(0, ids, ones[: ids.numel()])

    def index_add_only():
        ids = ti.reshape(-1).remainder(E + 1)
        for l in range(L):
            counts[l].index_add_(0, ids, ones[: ids.numel()])

    def scatter_count():
        for l in range(L):
            ids = ti.reshape(-1).remainder(E + 1).long()
            counts[l].scatter_add_(0, ids, ones[: ids.numel()])

    def fill():
        for l in range(L): tw.fill_(0.0)

    def chain():
        for l in range(L):
            lg = F.linear(x, Wg[l])
            ops.topk_softmax(tw, ti, tei, lg, True)
            ids = ti.reshape(-1).remainder(E + 1)
            counts[l].index_add_(0, ids, ones[: ids.numel()])

    ti.copy_(torch.randint(0, E, (M, TOPK), device=d, dtype=torch.int32))
    print(f"M={M:2d} us per layer: gate GEMM {graph_time(gate):5.1f} | topk_softmax {graph_time(topk):5.1f} | "
          f"remainder+index_add {graph_time(count):5.1f} (index_add alone {graph_time(index_add_only):5.1f}, "
          f"scatter_add {graph_time(scatter_count):5.1f}) | trivial fill {graph_time(fill):4.1f} | "
          f"chain gate+topk+count {graph_time(chain):5.1f}", flush=True)
