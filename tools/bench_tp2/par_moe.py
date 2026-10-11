"""Can hot (VRAM) and cold (UVA) MoE GEMMs overlap on two XPU queues, eagerly and inside an XPU graph?"""
import time, torch
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, implement_zp
from vllm.model_executor.layers.fused_moe.b70_hotcold import _PinnedArena
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
E_LOCAL, E_GLOBAL, TOPK, N, K, G, HOT, L = 256, 512, 10, 640, 2560, 128, 136, 48
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
for l in range(L):  # distinct weights per layer so nothing stays in cache
    w13 = implement_zp(torch.randint(0, 256, (E_LOCAL, 2 * N, K // 2), dtype=torch.uint8))
    w2 = implement_zp(torch.randint(0, 256, (E_LOCAL, K, N // 2), dtype=torch.uint8))
    w13_s = (torch.rand(E_LOCAL, 2 * N, K // G) * 0.002 + 0.001).half(); w2_s = (torch.rand(E_LOCAL, K, N // G) * 0.002 + 0.001).half()
    perm = torch.randperm(E_LOCAL); hot_ids, cold_ids = perm[:HOT].sort().values, perm[HOT:].sort().values
    hot = moe(w13[hot_ids].xpu(), w13_s[hot_ids].xpu(), w2[hot_ids].xpu(), w2_s[hot_ids].xpu(), emap(hot_ids), HOT)
    views = []
    for t in (w13, w13_s, w2, w2_s):
        p = arena.empty(t[cold_ids].shape, t.dtype); p.copy_(t[cold_ids]); views.append(get_accelerator_view_from_cpu_tensor(p))
    layers.append((hot, moe(*views, emap(cold_ids), E_LOCAL - HOT)))
    del w13, w2
print("built", L, "layers; arena GiB", arena.gib, flush=True)
side = torch.xpu.Stream()
def run(x, tw, ti, outs, colds, mode):
    for l, (hot, cold) in enumerate(layers):
        if mode == "hot":   hot.apply(output=outs[l], hidden_states=x, topk_weights=tw, topk_ids=ti)
        elif mode == "cold": cold.apply(output=colds[l], hidden_states=x, topk_weights=tw, topk_ids=ti)
        elif mode == "serial":
            hot.apply(output=outs[l], hidden_states=x, topk_weights=tw, topk_ids=ti)
            cold.apply(output=colds[l], hidden_states=x, topk_weights=tw, topk_ids=ti); outs[l].add_(colds[l])
        else:  # parallel: cold on the side queue
            side.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(side):
                cold.apply(output=colds[l], hidden_states=x, topk_weights=tw, topk_ids=ti)
            hot.apply(output=outs[l], hidden_states=x, topk_weights=tw, topk_ids=ti)
            torch.xpu.current_stream().wait_stream(side); outs[l].add_(colds[l])
def timed(fn, n=10):
    fn(); torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / n * 1e3
for rows in (4, 16):
    x = torch.randn(rows, K, device="xpu", dtype=torch.bfloat16) * 0.5
    ti = torch.stack([torch.randperm(E_GLOBAL)[:TOPK] for _ in range(rows)]).to("xpu", torch.int32)
    tw = torch.softmax(torch.randn(rows, TOPK, device="xpu"), -1)
    res = {}
    for mode in ("hot", "cold", "serial", "parallel"):
        outs = [torch.zeros_like(x) for _ in range(L)]; colds = [torch.zeros_like(x) for _ in range(L)]
        eager = timed(lambda: run(x, tw, ti, outs, colds, mode))
        if mode in ("serial", "parallel"): res[mode] = torch.stack(outs).clone()
        g = torch.xpu.XPUGraph(); s = torch.xpu.Stream()
        s.wait_stream(torch.xpu.current_stream())
        try:
            with torch.xpu.stream(s):
                run(x, tw, ti, outs, colds, mode)  # warm-up on the capture stream
                torch.xpu.synchronize()
                with torch.xpu.graph(g, stream=s):
                    run(x, tw, ti, outs, colds, mode)
            graph = timed(lambda: g.replay())
            if mode in ("serial", "parallel"): res[mode + "_g"] = torch.stack(outs).clone()
            gs = f"{graph:7.2f} ms"
        except Exception as e:
            gs = f"capture FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"rows {rows:2d} {mode:8s}: eager {eager:7.2f} ms   graph {gs}", flush=True)
    eq = all(torch.equal(res["serial"], res[k]) for k in res)
    print(f"rows {rows:2d}: all serial/parallel/graph outputs bit-identical: {eq}", flush=True)
