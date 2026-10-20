"""Spike: how fast is decode MoE when every routed expert lives in VRAM, vs today's hot/cold split?

Runs b70moe.moe_decode (0048/0049) on one B70 for L layers inside one XPU graph, random int4 data.
  HOT=146 COLD_SHARE=0.12  -> today's layout (cold experts in pinned RAM; set B70_MOE_CPU=1 for the 0049 CPU path)
  HOT=255 COLD_SHARE=0     -> all experts resident (one dummy cold expert that is never routed)
usage: allvram_spike.py <lib.so>   env: HOT, COLD_SHARE, L (layers built, default 32), ROWS (default 1,2,4,8,16)
Prints µs per layer and the 48-layer total in ms; the int4 -> int3 byte scaling is applied offline."""
import os, sys, time, torch
from vllm_xpu_kernels.fused_moe_interface import implement_zp
from vllm.model_executor.layers.fused_moe.b70_hotcold import _PinnedArena
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

torch.ops.load_library(sys.argv[1])
HOT = int(os.environ.get("HOT", "146"))
COLD_SHARE = float(os.environ.get("COLD_SHARE", "0.12"))
L = int(os.environ.get("L", "32"))
E_LOCAL, E_GLOBAL, TOPK, N, K, G = 256, 512, 10, 640, 2560, 128


def emap(ids):
    m = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
    for i, g in enumerate(ids):
        m[int(g)] = i
    return m.xpu()


torch.manual_seed(0)
arena = _PinnedArena()
layers = []
for l in range(L):
    w13 = implement_zp(torch.randint(0, 256, (E_LOCAL, 2 * N, K // 2), dtype=torch.uint8))
    w2 = implement_zp(torch.randint(0, 256, (E_LOCAL, K, N // 2), dtype=torch.uint8))
    w13_s = (torch.rand(E_LOCAL, 2 * N, K // G) * 0.002 + 0.001).bfloat16()
    w2_s = (torch.rand(E_LOCAL, K, N // G) * 0.002 + 0.001).bfloat16()
    perm = torch.randperm(E_LOCAL)
    hot_ids, cold_ids = perm[:HOT].sort().values, perm[HOT:].sort().values
    ht = [t[hot_ids].xpu() for t in (w13, w13_s, w2, w2_s)]
    ct = []
    for t in (w13, w13_s, w2, w2_s):
        p = arena.empty(t[cold_ids].shape, t.dtype)
        p.copy_(t[cold_ids])
        ct.append(get_accelerator_view_from_cpu_tensor(p))
    layers.append(dict(hm=emap(hot_ids), cm=emap(cold_ids), ht=ht, ct=ct, hot=hot_ids.tolist(), cold=cold_ids.tolist()))
    del w13, w2
print(f"built L={L} HOT={HOT} COLD_SHARE={COLD_SHARE} B70_MOE_CPU={os.environ.get('B70_MOE_CPU', '0')} "
      f"VRAM {torch.xpu.memory_allocated() / 2**30:.1f} GiB", flush=True)


def routing(rows, lay):
    # half of the 10 picks land on this rank (EP2); of those, COLD_SHARE go to cold experts
    ids = torch.empty(rows, TOPK, dtype=torch.int32)
    for r in range(rows):
        picks = set()
        while len(picks) < TOPK:
            if torch.rand(()) < 0.5:
                pool = lay["cold"] if (COLD_SHARE > 0 and torch.rand(()) < COLD_SHARE) else lay["hot"]
                picks.add(int(pool[int(torch.randint(0, len(pool), ()))]))
            else:
                picks.add(int(torch.randint(E_LOCAL, E_GLOBAL, ())))
        ids[r] = torch.tensor(sorted(picks))
    return ids


def graph_time(fn, n=30):
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn()
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s):
            fn()
    g.replay()
    torch.xpu.synchronize()
    best = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            g.replay()
        torch.xpu.synchronize()
        best.append((time.perf_counter() - t) / n)
    return sorted(best)[1]  # median of 3


for rows in [int(r) for r in os.environ.get("ROWS", "1,2,4,8,16").split(",")]:
    ids = [routing(rows, l).to("xpu") for l in layers]
    x = torch.randn(rows, K, device="xpu", dtype=torch.bfloat16)
    tw = torch.softmax(torch.randn(rows, TOPK, device="xpu"), -1)
    outs = [torch.zeros_like(x) for _ in layers]
    t = graph_time(lambda: [torch.ops.b70moe.moe_decode(o, x, tw, i, l["hm"], l["cm"], *l["ht"], *l["ct"])
                            for l, i, o in zip(layers, ids, outs)])
    uh = sum(len({g for g in i.flatten().tolist() if int(l["hm"][g]) >= 0}) for l, i in zip(layers, ids)) / L
    uc = sum(len({g for g in i.flatten().tolist() if int(l["cm"][g]) >= 0}) for l, i in zip(layers, ids)) / L
    stats = torch.ops.b70moe.cpu_stats(0).tolist() if hasattr(torch.ops.b70moe, "cpu_stats") else []
    print(f"rows {rows:2d}: experts/layer hot {uh:4.1f} cold {uc:4.1f} | {t / L * 1e6:7.1f} us/layer | "
          f"48 layers {t / L * 48 * 1e3:6.2f} ms | cpu served/timeouts {stats}", flush=True)
