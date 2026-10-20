"""Spike check: b70w3.moe_decode_w3 vs a torch fp32 reference on a real layer of Paiton's W3A8 container.
usage: test_w3_spike.py <libb70_w3_spike.so> <layers/L24.safetensors> [layer index, default 24]
Rank 0 of EP2 = global experts 0..255. Random bf16 activations, random top-10 routing over all 512 experts."""
import sys

import torch
from safetensors import safe_open

torch.ops.load_library(sys.argv[1])
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 24
P = f"model.language_model.layers.{LAYER}.mlp.experts."
E_LOCAL, K, NS, SL, TOPK = 256, 2560, 5, 128, 10
dev = "xpu"

with safe_open(sys.argv[2], "pt") as f:
    t = {n: f.get_slice(P + n)[0:E_LOCAL] for n in ("gu_codes", "gu_scales", "gu_exp2", "dn_codes", "dn_scales",
                                                       "dn_exp2")}
w = {n: v.contiguous().to(dev) for n, v in t.items()}
print({n: (tuple(v.shape), str(v.dtype)) for n, v in w.items()})
u8 = lambda v: v.view(torch.uint8)

LEV = torch.tensor([-8., -5., -3., -1., 1., 3., 5., 8.], device=dev)
H = torch.ones(1, 1, device=dev)
while H.shape[0] < 128:
    H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H = H / 128 ** 0.5


def unpack3(codes, n):  # int32 [..., words] -> levels [..., n]
    b = codes.contiguous().view(torch.uint8)
    bits = (b.unsqueeze(-1) >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
    bits = bits.reshape(*b.shape[:-1], -1)[..., : 3 * n].reshape(*b.shape[:-1], n, 3).int()
    return LEV[bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]]


def q8(v, amax_dims):
    sc = torch.clamp(v.abs().amax(dim=amax_dims, keepdim=True) / 448.0, min=1.0 / (448 * 512))
    return (v / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float(), sc


def expert_ref(e, x):  # x [m, K] bf16 -> y [m, K] fp32 (before bf16 rounding)
    gu = unpack3(w["gu_codes"][e], K) * w["gu_scales"][e].float().repeat_interleave(64, -1) \
        * 2.0 ** float(w["gu_exp2"][e])                                       # [5, 2, 128, K]
    dn = unpack3(w["dn_codes"][e], SL) * w["dn_scales"][e].float().repeat_interleave(64, -1) \
        * 2.0 ** float(w["dn_exp2"][e])                                       # [5, K, 128]
    xr = (x.float().view(-1, K // 128, 128) @ H).view(-1, K)
    xq, xs = q8(xr, (1,))
    y = torch.zeros(x.shape[0], K, device=dev)
    for s in range(NS):
        g = (xq @ gu[s, 0].t()) * xs
        u = (xq @ gu[s, 1].t()) * xs
        a = torch.nn.functional.silu(g) * u                                   # [m, 128]
        aq, asc = q8(a @ H, (1,))
        y += (aq @ dn[s].t()) * asc
    return y


torch.manual_seed(0)
mp = torch.full((512,), -1, dtype=torch.int32)
mp[:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32)
mp = mp.to(dev)
ok = True
for M in (1, 2, 4, 7, 16):
    x = (torch.randn(M, K) * 0.5).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(512)[:TOPK] for _ in range(M)]).int().to(dev)
    tw = torch.softmax(torch.randn(M, TOPK), -1).to(dev)
    out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    torch.ops.b70w3.moe_decode_w3(out, x, tw, ids, mp, w["gu_codes"], u8(w["gu_scales"]), w["gu_exp2"],
                                  w["dn_codes"], u8(w["dn_scales"]), w["dn_exp2"])
    ref = torch.zeros(M, K, device=dev)
    for m in range(M):
        for j in range(TOPK):
            g = int(ids[m, j])
            if g < E_LOCAL:
                ref[m] += expert_ref(g, x[m:m + 1])[0].bfloat16().float() * tw[m, j]
    d = (out.float() - ref)
    rel = d.norm() / ref.norm()
    mx = d.abs().max() / ref.abs().max()
    good = rel < 2e-2
    ok &= bool(good)
    print(f"M={M:2d}: local routes {int((ids < E_LOCAL).sum())}, rel L2 err {rel:.2e}, max err / max {mx:.2e}, "
          f"|ref| {ref.norm():.3e} {'OK' if good else 'FAIL'}", flush=True)
print("PASS" if ok else "FAIL")
