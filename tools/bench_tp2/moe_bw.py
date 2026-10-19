"""Hot (VRAM) and cold (pinned host, UVA) int4 MoE at decode sizes on one B70: time per layer as XPU graph replays
over 48 distinct layers, with GB/s of the expert bytes actually touched. Routing: each row picks 10 of 512 experts;
COLD_SHARE of this rank's routes go to cold experts (the rest hot)."""
import sys, time, torch
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, implement_zp
from vllm.model_executor.layers.fused_moe.b70_hotcold import _PinnedArena
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
E_LOCAL, E_GLOBAL, TOPK, N, K, G, HOT, L = 256, 512, 10, 640, 2560, 128, 146, 48
COLD_SHARE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
EXPERT_BYTES = 2 * N * K // 2 + K * N // 2 + (2 * N * K // G + K * N // G) * 2
def moe(w13, w13_s, w2, w2_s, emap, n):
    for w in (w13, w2): w.xpu_fused_moe = True
    m = XpuFusedMoe(w13=w13, w13_scales=w13_s, w13_bias=None, w2=w2, w2_scales=w2_s, w2_bias=None,
                    n_experts_per_token=TOPK, activation="silu", num_experts=n, ep_rank=0, ep_size=2, expert_map=emap)
    m.w13, m.w2 = w13.view(torch.int8), w2.view(torch.int8); return m
def emap(ids):
    m = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
    for i, g in enumerate(ids): m[int(g)] = i
    return m.xpu()
torch.manual_seed(0); arena = _PinnedArena(); layers = []
local = list(range(E_LOCAL))  # this rank owns global 0..255
for l in range(L):
    w13 = implement_zp(torch.randint(0, 256, (E_LOCAL, 2 * N, K // 2), dtype=torch.uint8))
    w2 = implement_zp(torch.randint(0, 256, (E_LOCAL, K, N // 2), dtype=torch.uint8))
    w13_s = (torch.rand(E_LOCAL, 2 * N, K // G) * 0.002 + 0.001).half(); w2_s = (torch.rand(E_LOCAL, K, N // G) * 0.002 + 0.001).half()
    hot_ids, cold_ids = torch.arange(HOT), torch.arange(HOT, E_LOCAL)
    hot = moe(w13[hot_ids].xpu(), w13_s[hot_ids].xpu(), w2[hot_ids].xpu(), w2_s[hot_ids].xpu(), emap(hot_ids), HOT)
    views = []
    for t in (w13, w13_s, w2, w2_s):
        p = arena.empty(t[cold_ids].shape, t.dtype); p.copy_(t[cold_ids]); views.append(get_accelerator_view_from_cpu_tensor(p))
    layers.append((hot, moe(*views, emap(cold_ids), E_LOCAL - HOT)))
    del w13, w2
print(f"built {L} layers, {HOT} hot / {E_LOCAL-HOT} cold, expert {EXPERT_BYTES/1e6:.2f} MB, cold share {COLD_SHARE}", flush=True)
def routing(rows):
    """rows x 10 global ids: half of each row on this rank (0..255), cold with prob COLD_SHARE."""
    ids = torch.empty(rows, TOPK, dtype=torch.int32)
    for r in range(rows):
        picks = set()
        while len(picks) < TOPK:
            if torch.rand(()) < 0.5:
                g = int(torch.randint(HOT, E_LOCAL, ())) if torch.rand(()) < COLD_SHARE else int(torch.randint(0, HOT, ()))
            else:
                g = int(torch.randint(E_LOCAL, E_GLOBAL, ()))
            picks.add(g)
        ids[r] = torch.tensor(sorted(picks))
    return ids
def graph_time(fn, n=20):
    s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s): fn()
    g.replay(); torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): g.replay()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n
for rows in (1, 4, 8, 16):
    ids = routing(rows)
    u_hot = len({int(g) for g in ids.flatten() if g < HOT}); u_cold = len({int(g) for g in ids.flatten() if HOT <= g < E_LOCAL})
    x = torch.randn(rows, K, device="xpu", dtype=torch.bfloat16) * 0.5
    ti = ids.to("xpu"); tw = torch.softmax(torch.randn(rows, TOPK, device="xpu"), -1)
    out = torch.zeros_like(x); cold_out = torch.zeros_like(x)
    th = graph_time(lambda: [h.apply(output=out, hidden_states=x, topk_weights=tw, topk_ids=ti) for h, _ in layers])
    tc = graph_time(lambda: [c.apply(output=cold_out, hidden_states=x, topk_weights=tw, topk_ids=ti) for _, c in layers])
    bh, bc = u_hot * EXPERT_BYTES * L, u_cold * EXPERT_BYTES * L
    print(f"rows {rows:2d}: hot {u_hot:2d} experts/layer {th/L*1e6:6.1f} us/layer ({bh/th/1e9:4.0f} GB/s, {100*bh/th/1e9/608:3.0f}% of 608) | "
          f"cold {u_cold:2d} experts/layer {tc/L*1e6:6.1f} us/layer ({bc/tc/1e9 if u_cold else 0:4.1f} GB/s over PCIe) | "
          f"48 layers: hot {th*1e3:5.2f} ms, cold {tc*1e3:5.2f} ms", flush=True)
