"""Card test for the B70 MoE glue ops in libb70_moe_decode.so (one B70, engine stopped):
  topk_softmax vs _moe_C.topk_softmax (ids exact, weights in fp32 ulps), random and tie-heavy bf16 logits
  moe_decode_counted vs moe_decode + remainder/index_add_ (same output, identical counts), and timing.
usage: test_glue.py <lib.so>"""
import sys, time, torch
import vllm._custom_ops as ops
from vllm_xpu_kernels.fused_moe_interface import implement_zp
from vllm.model_executor.layers.fused_moe.b70_hotcold import _PinnedArena
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

torch.ops.load_library(sys.argv[1])
d = torch.device("xpu")
E, TOPK = 512, 10
torch.manual_seed(0)


def stock_topk(lg, renorm):
    M = lg.shape[0]
    tw = torch.empty(M, TOPK, dtype=torch.float32, device=d); ti = torch.empty(M, TOPK, dtype=torch.int32, device=d)
    ops.topk_softmax(tw, ti, torch.empty_like(ti), lg, renorm)
    return tw, ti


def mine_topk(lg, renorm):
    M = lg.shape[0]
    tw = torch.empty(M, TOPK, dtype=torch.float32, device=d); ti = torch.empty(M, TOPK, dtype=torch.int32, device=d)
    torch.ops.b70moe.topk_softmax(tw, ti, lg, renorm)
    return tw, ti


bad = 0
for case in ("randn", "router-like", "ties"):
    for renorm in (True, False):
        id_mis = rows_mis = 0; max_ulp = 0; n = 0
        for trial in range(50):
            M = (1, 2, 4, 8, 16)[trial % 5]
            if case == "randn":
                lg = torch.randn(M, E, device=d).bfloat16()
            elif case == "router-like":
                lg = (torch.randn(M, E, device=d) * 2 + torch.randn(1, E, device=d) * 3).bfloat16()
            else:  # coarse values: many exact ties
                lg = (torch.randint(-8, 8, (M, E), device=d).float() * 0.5).bfloat16()
            sw, si = stock_topk(lg, renorm)
            mw, mi = mine_topk(lg, renorm)
            if case == "ties":  # order within equal probabilities is part of the contract: compare as is
                pass
            id_mis += int((si != mi).sum()); rows_mis += int((si != mi).any(1).sum())
            same = si == mi
            if same.any():
                u = (sw.view(torch.int32) - mw.view(torch.int32)).abs()[same]
                max_ulp = max(max_ulp, int(u.max()))
            n += M
        bad += id_mis
        print(f"topk {case:11s} renorm={renorm!s:5s}: {n} rows, id mismatches {id_mis} (rows {rows_mis}), "
              f"weights max {max_ulp} fp32 ulp", flush=True)

# padding rows (VLLM_MOE_SKIP_PADDING): stock writes -1 ids for them
lg = torch.randn(8, E, device=d).bfloat16()
pad = torch.tensor([0, 0, 1, 0, 1, 1, 0, 0], dtype=torch.bool, device=d)
tw1 = torch.empty(8, TOPK, dtype=torch.float32, device=d); ti1 = torch.empty(8, TOPK, dtype=torch.int32, device=d)
ops.topk_softmax(tw1, ti1, torch.empty_like(ti1), lg, True, is_padding=pad)
tw2 = torch.empty_like(tw1); ti2 = torch.empty_like(ti1)
torch.ops.b70moe.topk_softmax(tw2, ti2, lg, True, pad)
ok_pad = bool((ti1 == ti2).all()); bad += not ok_pad
print(f"topk padding rows: ids identical {ok_pad} (stock pad-row ids {ti1[2].tolist()[:3]}...)", flush=True)

# degenerate rows: NaN logits (dummy runs), all equal, -inf entries
lg = torch.randn(6, E, device=d).bfloat16()
lg[0] = float("nan"); lg[1] = 0.0; lg[2, ::2] = float("-inf"); lg[3] = float("-inf"); lg[4, 7] = float("nan")
lg[5, 100] = float("inf")
tw1 = torch.empty(6, TOPK, dtype=torch.float32, device=d); ti1 = torch.empty(6, TOPK, dtype=torch.int32, device=d)
ops.topk_softmax(tw1, ti1, torch.empty_like(ti1), lg, True)
tw2 = torch.empty_like(tw1); ti2 = torch.empty_like(ti1)
torch.ops.b70moe.topk_softmax(tw2, ti2, lg, True, None)
ok_deg = bool((ti1 == ti2).all()) and bool((((tw1 - tw2).abs() < 1e-6) | (tw1.isnan() & tw2.isnan())).all())
in_range = bool(((ti2 >= 0) & (ti2 < E)).all()); bad += (not ok_deg) + (not in_range)
print(f"topk degenerate rows: ids identical {bool((ti1 == ti2).all())}, weights close {ok_deg}, ids in range {in_range}; "
      f"stock ids {ti1[:, :3].tolist()}, b70 ids {ti2[:, :3].tolist()}", flush=True)

# ---- moe_decode_counted ----
E_LOCAL, N, K, G, HOT = 256, 640, 2560, 128, 146
w13 = implement_zp(torch.randint(0, 256, (E_LOCAL, 2 * N, K // 2), dtype=torch.uint8))
w2 = implement_zp(torch.randint(0, 256, (E_LOCAL, K, N // 2), dtype=torch.uint8))
s13 = (torch.rand(E_LOCAL, 2 * N, K // G) * 0.002 + 0.001).bfloat16(); s2 = (torch.rand(E_LOCAL, K, N // G) * 0.002 + 0.001).bfloat16()
perm = torch.randperm(E_LOCAL); hot_ids, cold_ids = perm[:HOT].sort().values, perm[HOT:].sort().values
arena = _PinnedArena()


def emap(ids):
    m = torch.full((E,), -1, dtype=torch.int32)
    for i, g in enumerate(ids): m[int(g)] = i
    return m.to(d)


ht = [t[hot_ids].to(d) for t in (w13, s13, w2, s2)]
ct = []
for t in (w13, s13, w2, s2):
    p = arena.empty(t[cold_ids].shape, t.dtype); p.copy_(t[cold_ids]); ct.append(get_accelerator_view_from_cpu_tensor(p))
hm, cm = emap(hot_ids), emap(cold_ids)
for M in (1, 4, 16):
    x = torch.randn(M, K, device=d).bfloat16()
    lg = torch.randn(M, E, device=d).bfloat16()
    tw, ti = stock_topk(lg, True)
    ti[0, -1] = -1  # a padding slot: counted in the last bucket by remainder
    a = torch.empty_like(x); b = torch.empty_like(x)
    c_ref = torch.zeros(E + 1, dtype=torch.int64, device=d); c_new = torch.zeros_like(c_ref)
    torch.ops.b70moe.moe_decode(a, x, tw, ti, hm, cm, *ht, *ct)
    ids = ti.reshape(-1).remainder(E + 1)
    c_ref.index_add_(0, ids, torch.ones(ids.numel(), dtype=torch.int64, device=d))
    torch.ops.b70moe.moe_decode_counted(b, x, tw, ti, hm, cm, *ht, *ct, c_new)
    torch.xpu.synchronize()
    ok_c = bool((c_ref == c_new).all()); ok_o = bool((a == b).all())
    bad += (not ok_c) + (not ok_o)
    print(f"moe_decode_counted M={M:2d}: output identical {ok_o}, counts identical {ok_c} (total {int(c_new.sum())})",
          flush=True)


def graph_time(fn, L=48, n=50):
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
    return sorted(ts)[2] / L * 1e6


for M in (1, 4, 16):
    lgs = [torch.randn(M, E, device=d).bfloat16() for _ in range(48)]
    tw = torch.empty(M, TOPK, dtype=torch.float32, device=d); ti = torch.empty(M, TOPK, dtype=torch.int32, device=d)
    tei = torch.empty_like(ti)
    t_s = graph_time(lambda: [ops.topk_softmax(tw, ti, tei, lg, True) for lg in lgs])
    t_m = graph_time(lambda: [torch.ops.b70moe.topk_softmax(tw, ti, lg, True) for lg in lgs])
    x = torch.randn(M, K, device=d).bfloat16(); o = torch.empty_like(x)
    ti.copy_(stock_topk(lgs[0], True)[1]); tw.copy_(stock_topk(lgs[0], True)[0])
    cnt = torch.zeros(48, E + 1, dtype=torch.int64, device=d); ones = torch.ones(M * TOPK, dtype=torch.int64, device=d)

    def old():
        for l in range(48):
            torch.ops.b70moe.moe_decode(o, x, tw, ti, hm, cm, *ht, *ct)
            cnt[l].index_add_(0, ti.reshape(-1).remainder(E + 1), ones)

    def new():
        for l in range(48):
            torch.ops.b70moe.moe_decode_counted(o, x, tw, ti, hm, cm, *ht, *ct, cnt[l])
    t_o, t_n = graph_time(old), graph_time(new)
    print(f"M={M:2d} us per layer: topk_softmax stock {t_s:5.1f} -> b70 {t_m:5.1f} | decode op + count: "
          f"{t_o:6.1f} -> {t_n:6.1f} (same layer weights each call)", flush=True)
print("RESULT", "PASS" if bad == 0 else f"FAIL ({bad})")
