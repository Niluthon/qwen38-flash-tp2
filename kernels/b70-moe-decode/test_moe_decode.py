"""b70moe.moe_decode vs the stock hot+cold XpuFusedMoe path: correctness and graph timing on one B70.
usage: test_moe_decode.py <lib.so> [cold_share]"""
import sys, time, torch
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, implement_zp
from vllm.model_executor.layers.fused_moe.b70_hotcold import _PinnedArena
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
torch.ops.load_library(sys.argv[1])
COLD_SHARE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
E_LOCAL, E_GLOBAL, TOPK, N, K, G, HOT, L = 256, 512, 10, 640, 2560, 128, 146, 48
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
for l in range(L):
    w13 = implement_zp(torch.randint(0, 256, (E_LOCAL, 2 * N, K // 2), dtype=torch.uint8))
    w2 = implement_zp(torch.randint(0, 256, (E_LOCAL, K, N // 2), dtype=torch.uint8))
    w13_s = (torch.rand(E_LOCAL, 2 * N, K // G) * 0.002 + 0.001).bfloat16(); w2_s = (torch.rand(E_LOCAL, K, N // G) * 0.002 + 0.001).bfloat16()
    perm = torch.randperm(E_LOCAL); hot_ids, cold_ids = perm[:HOT].sort().values, perm[HOT:].sort().values
    hot_t = [w13[hot_ids].xpu(), w13_s[hot_ids].xpu(), w2[hot_ids].xpu(), w2_s[hot_ids].xpu()]
    views = []
    for t in (w13, w13_s, w2, w2_s):
        p = arena.empty(t[cold_ids].shape, t.dtype); p.copy_(t[cold_ids]); views.append(get_accelerator_view_from_cpu_tensor(p))
    hm, cm = emap(hot_ids), emap(cold_ids)
    layers.append(dict(hot=moe(*hot_t, hm, HOT), cold=moe(*views, cm, E_LOCAL - HOT), hm=hm, cm=cm, ht=hot_t, ct=views,
                       hot_ids=hot_ids, cold_ids=cold_ids))
    del w13, w2
print("built", flush=True)
def routing(rows, lay):
    hot, cold = lay["hot_ids"].tolist(), lay["cold_ids"].tolist()
    ids = torch.empty(rows, TOPK, dtype=torch.int32)
    for r in range(rows):
        picks = set()
        while len(picks) < TOPK:
            if torch.rand(()) < 0.5:
                pool = cold if torch.rand(()) < COLD_SHARE else hot
                picks.add(int(pool[int(torch.randint(0, len(pool), ()))]))
            else:
                picks.add(int(torch.randint(E_LOCAL, E_GLOBAL, ())))
        ids[r] = torch.tensor(sorted(picks))
    return ids
def stock(lay, x, tw, ti, out, co):
    lay["hot"].apply(output=out, hidden_states=x, topk_weights=tw, topk_ids=ti)
    lay["cold"].apply(output=co, hidden_states=x, topk_weights=tw, topk_ids=ti)
    out.add_(co)
def mine(lay, x, tw, ti, out):
    torch.ops.b70moe.moe_decode(out, x, tw, ti, lay["hm"], lay["cm"], *lay["ht"], *lay["ct"])
def graph_time(fn, n=20):
    s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s): fn()
    g.replay(); torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): g.replay()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n
for rows in [int(r) for r in __import__("os").environ.get("ROWS", "1,2,4,8,12,16").split(",")]:
    lay = layers[0]
    # same routing for every layer (ids are global and every layer has its own hot/cold split -> per-layer ids)
    ids = [routing(rows, l).to("xpu") for l in layers]
    x = torch.randn(rows, K, device="xpu", dtype=torch.bfloat16)
    tw = torch.softmax(torch.randn(rows, TOPK, device="xpu"), -1)
    a = torch.zeros_like(x); c = torch.zeros_like(x); b = torch.zeros_like(x)
    stock(lay, x, tw, ids[0], a, c); mine(lay, x, tw, ids[0], b); torch.xpu.synchronize()
    err = ((a.float() - b.float()).abs().max() / a.float().abs().max()).item()
    rel_l2 = ((a.float() - b.float()).norm() / a.float().norm()).item()
    outs = [torch.zeros_like(x) for _ in layers]; cos = [torch.zeros_like(x) for _ in layers]
    t_stock = graph_time(lambda: [stock(l, x, tw, i, o, cc) for l, i, o, cc in zip(layers, ids, outs, cos)])
    t_mine = graph_time(lambda: [mine(l, x, tw, i, o) for l, i, o in zip(layers, ids, outs)])
    uh = sum(len({int(g) for g in i.flatten().tolist() if int(l["hm"][g]) >= 0}) for l, i in zip(layers, ids)) / L
    uc = sum(len({int(g) for g in i.flatten().tolist() if int(l["cm"][g]) >= 0}) for l, i in zip(layers, ids)) / L
    print(f"rows {rows:2d}: hot {uh:4.1f} cold {uc:3.1f} experts/layer | stock {t_stock/L*1e6:6.1f} us/layer | "
          f"b70moe {t_mine/L*1e6:6.1f} us/layer ({t_stock/t_mine:4.2f}x) | max rel err {err:.2e}, rel L2 {rel_l2:.2e}"
          + (f" | cpu served/timeouts {torch.ops.b70moe.cpu_stats(0).tolist()}" if hasattr(torch.ops.b70moe, "cpu_stats") else ""),
          flush=True)
