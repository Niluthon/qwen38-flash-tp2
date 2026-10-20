"""Card test for libb70_step_trace.so (one B70, engine stopped): marks inside an XPU graph, a known 1 GiB copy
between two marks, the cost per mark, and whether replays lose records.
usage: test_step_trace.py <lib.so>"""
import sys, time, torch

torch.ops.load_library(sys.argv[1])
d = torch.device("xpu")
anchor = torch.zeros(1, device=d)
a = torch.rand(1 << 28, device=d)  # 1 GiB fp32, random (the B70 compresses constant data)
b = torch.empty_like(a)
NM = 500
out = torch.zeros(1 << 20, 2, dtype=torch.int64)


def body():
    torch.ops.b70trace.mark(anchor, 1 << 31)
    b.copy_(a)
    torch.ops.b70trace.mark(anchor, 1)
    for i in range(NM):
        torch.ops.b70trace.mark(anchor, 2 + i)


def drain():
    time.sleep(0.05)
    n = torch.ops.b70trace.drain(anchor, out)
    return out[:n].clone()


s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
with torch.xpu.stream(s):
    body(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g, stream=s):
        body()
torch.xpu.synchronize(); drain()
REP = 10
t = time.perf_counter()
for _ in range(REP):
    g.replay()
torch.xpu.synchronize()
wall = (time.perf_counter() - t) / REP * 1e3
r = drain()
print(f"records {r.shape[0]} (expected {REP * (NM + 2)}), lost {torch.ops.b70trace.lost(anchor)}")
r = r[-(NM + 2):]  # last replay
ts, codes = r[:, 0].double(), r[:, 1]
print(f"codes in order: {bool((codes[1:] - codes[:-1] == 1).all() and codes[0] == (1 << 31) and codes[1] == 1)}")
copy_ms = (ts[1] - ts[0]).item() / 1e6
print(f"1 GiB copy between marks: {copy_ms:.3f} ms = {2 * (1 << 30) / (copy_ms * 1e-3) / 1e9:.0f} GB/s (read+write)")
dm = (ts[2:][1:] - ts[2:][:-1]) / 1e3
print(f"mark to mark inside the graph: median {dm.median():.2f} us, p90 {dm.quantile(0.9):.2f} us, "
      f"max {dm.max():.2f} us; same stamp (batched by the poller): {int((dm == 0).sum())} of {dm.numel()}")
print(f"graph replay wall {wall:.3f} ms vs stamped span {(ts[-1] - ts[0]).item() / 1e6:.3f} ms")
# reference: plain copy timed by events
e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
e0.record(); b.copy_(a); e1.record(); torch.xpu.synchronize()
print(f"same copy by XPU events: {e0.elapsed_time(e1):.3f} ms")
