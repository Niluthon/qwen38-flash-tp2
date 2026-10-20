"""b70hc (fused HC down / up+mix) vs the stock decode path: correctness on real HC weights and XPU-graph timing.
One B70, engine stopped.   usage: test_hc.py <libb70_hc.so> [layer]
Stock = oneDNN F.linear for both projections + the model's Triton silu / gate-mix kernels (ops/hc.py)."""
import json, os, sys, time, torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm, hc_gate_mix, hc_silu

torch.ops.load_library(sys.argv[1])
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CK = "/models/qwen38-w4a16"
d = torch.device("xpu")
HC, D, R, NPAD = 4, 2560, 320, 336

idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]


def get(name):
    with safe_open(f"{CK}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


def site(layer, kind):
    p = f"model.language_model.layers.{layer}.{kind}_hyper_connection."
    down, inj = get(p + "input_mix_weight_down.weight"), get(p + "block_inject_weight.weight")
    wd = torch.cat([down, inj, torch.zeros(NPAD - R - HC, down.shape[1], dtype=down.dtype)]).to(d, torch.bfloat16)
    return wd.contiguous(), get(p + "input_mix_weight_up.weight").to(d, torch.bfloat16).contiguous(), \
        get(p + "hc_norm.weight").to(d, torch.bfloat16)


def stock(xn, wd, wu):
    dn = F.linear(xn, wd)
    lora, inj, _ = dn.split([R, HC, NPAD - R - HC], dim=-1)
    return dn, hc_gate_mix(xn, F.linear(hc_silu(lora, HC), wu), HC)


def fused(xn, wd, wu):
    dn, act = torch.ops.b70hc.down(xn, wd, R + HC, R, HC)
    return dn, torch.ops.b70hc.up_mix(act, wu, xn, HC)


def ulps(a, b):
    ai, bi = a.contiguous().view(torch.int16).int(), b.contiguous().view(torch.int16).int()
    return (ai - bi).abs()


torch.manual_seed(0)
print(f"real HC weights, layer {LAYER}", flush=True)
for kind in ("attn", "mlp"):
    wd, wu, nw = site(LAYER, kind)
    for M in (1, 2, 4, 8, 12, 16):
        res = (torch.randn(M, HC * D, device=d) * 0.5).bfloat16()
        blk = torch.randn(M, D, device=d).bfloat16()
        inj = torch.randn(M, HC, device=d).bfloat16()
        _, xn = hc_combine_norm(res, blk, inj, nw, 1e-6, HC)
        dn_s, out_s = stock(xn, wd, wu)
        dn_f, out_f = fused(xn, wd, wu)
        u = ulps(dn_s[:, :R + HC], dn_f[:, :R + HC])
        act_s = hc_silu(dn_s[:, :R], HC)
        _, act_f = torch.ops.b70hc.down(xn, wd, R + HC, R, HC)
        ua = ulps(act_s, act_f)
        out_x = torch.ops.b70hc.up_mix(act_s.contiguous(), wu, xn, HC)  # up_mix alone, on the stock activations
        rel = ((out_s.float() - out_f.float()).norm() / out_s.float().norm()).item()
        rel_up = ((out_s.float() - out_x.float()).norm() / out_s.float().norm()).item()
        print(f"{kind} M={M:2d}: down max {int(u.max())} ulp, equal {100 * (u == 0).float().mean():.1f} %, "
              f"<=1 ulp {100 * (u <= 1).float().mean():.1f} %, silu act max {int(ua.max())} ulp | out rel L2 {rel:.2e} (up_mix alone {rel_up:.2e}), "
              f"out equal {100 * (out_s == out_f).float().mean():.1f} %, max ulp {int(ulps(out_s, out_f).max())}",
              flush=True)

# ---- timing: 96 sites (48 layers x 2), distinct random weights, one XPU graph ----
NS = 96
W = [(torch.randn(NPAD, HC * D, device=d) * 0.01).bfloat16() for _ in range(NS)]
U = [(torch.randn(HC * D, R, device=d) * 0.05).bfloat16() for _ in range(NS)]


def graph_time(fn, n=20):
    s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        fn(); torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=s):
            fn()
    g.replay(); torch.xpu.synchronize()
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n): g.replay()
        torch.xpu.synchronize(); ts.append((time.perf_counter() - t) / n)
    return sorted(ts)[1]


nbytes = (NPAD * HC * D + HC * D * R) * 2
print(f"\ntiming, {NS} sites, B70_HC_DOWN_RPW={os.environ.get('B70_HC_DOWN_RPW', '2')}", flush=True)
for M in (1, 4, 8, 16):
    xn = torch.randn(M, HC * D, device=d).bfloat16()
    ts = graph_time(lambda: [stock(xn, w, u) for w, u in zip(W, U)])
    tf = graph_time(lambda: [fused(xn, w, u) for w, u in zip(W, U)])
    td = graph_time(lambda: [torch.ops.b70hc.down(xn, w, R + HC, R, HC) for w in W])
    lora = torch.randn(M, NPAD, device=d).bfloat16()
    act = torch.randn(M, R, device=d).bfloat16()
    gate = torch.randn(M, HC * D, device=d).bfloat16()
    t_sd = graph_time(lambda: [F.linear(xn, w) for w in W])
    t_ss = graph_time(lambda: [hc_silu(lora[:, :R], HC) for _ in W])
    t_su = graph_time(lambda: [F.linear(act, u) for u in U])
    t_sm = graph_time(lambda: [hc_gate_mix(xn, gate, HC) for _ in W])
    t_um = graph_time(lambda: [torch.ops.b70hc.up_mix(act, u, xn, HC) for u in U])

    def hybrid():
        for w, u in zip(W, U):
            dn, a = torch.ops.b70hc.down(xn, w, R + HC, R, HC)
            hc_gate_mix(xn, F.linear(a, u), HC)
    th = graph_time(hybrid)
    print(f"      stock parts us/site: down GEMM {t_sd / NS * 1e6:5.1f}, silu {t_ss / NS * 1e6:4.1f}, up GEMM "
          f"{t_su / NS * 1e6:5.1f}, gate-mix {t_sm / NS * 1e6:4.1f} | b70 up_mix {t_um / NS * 1e6:5.1f} | "
          f"hybrid (b70 down+silu, oneDNN up, Triton mix) {th / NS * 1e6:5.1f} us/site ({ts / th:4.2f}x)", flush=True)
    print(f"M={M:2d}: stock {ts / NS * 1e6:6.1f} us/site ({ts * 1e3:5.2f} ms per 96) | fused {tf / NS * 1e6:6.1f} us/site "
          f"({tf * 1e3:5.2f} ms, {ts / tf:4.2f}x, {nbytes * NS / tf / 1e9:4.0f} GB/s) | down alone {td / NS * 1e6:5.1f} us",
          flush=True)
