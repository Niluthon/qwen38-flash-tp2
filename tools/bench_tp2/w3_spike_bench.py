"""Spike: decode MoE time with every routed expert resident in VRAM as Paiton's 3-bit codebook experts (b70w3 op,
kernels/b70-w3-spike), for L layers in one XPU graph, random data (never constant: the B70 compresses it).
usage: w3_spike_bench.py <libb70_w3_spike.so>   env: L (default 48), ROWS (default 1,2,4,8,16)
Routing as allvram_spike.py: half of the 10 picks land on this rank (EP2), all of them resident."""
import os
import sys
import time

import torch

torch.ops.load_library(sys.argv[1])
L = int(os.environ.get("L", "48"))
E_LOCAL, E_GLOBAL, TOPK, K = 256, 512, 10, 2560
dev = "xpu"
torch.manual_seed(0)


def rand_i32(*shape):
    return torch.randint(-2**31, 2**31 - 1, shape, dtype=torch.int32, device=dev)


def rand_e4m3(*shape):  # valid, non-constant E4M3 bytes around 2^-3 .. 2^0, random sign
    b = torch.randint(0x30, 0x40, shape, dtype=torch.uint8, device=dev)
    return b | (torch.randint(0, 2, shape, dtype=torch.uint8, device=dev) << 7)


layers = []
for _ in range(L):
    layers.append(dict(
        gu=rand_i32(E_LOCAL, 5, 2, 128, 240), gus=rand_e4m3(E_LOCAL, 5, 2, 128, 40),
        gue=torch.full((E_LOCAL,), -7, dtype=torch.int8, device=dev),
        dn=rand_i32(E_LOCAL, 5, K, 12), dns=rand_e4m3(E_LOCAL, 5, K, 2),
        dne=torch.full((E_LOCAL,), -7, dtype=torch.int8, device=dev)))
mp = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
mp[:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32)
mp = mp.to(dev)
print(f"built L={L}: VRAM {torch.xpu.memory_allocated() / 2**30:.2f} GiB of experts", flush=True)


def routing(rows):
    ids = torch.empty(rows, TOPK, dtype=torch.int32)
    for r in range(rows):
        picks = set()
        while len(picks) < TOPK:
            lo, hi = (0, E_LOCAL) if torch.rand(()) < 0.5 else (E_LOCAL, E_GLOBAL)
            picks.add(int(torch.randint(lo, hi, ())))
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
    return sorted(best)[1]


for rows in [int(r) for r in os.environ.get("ROWS", "1,2,4,8,16").split(",")]:
    ids = [routing(rows).to(dev) for _ in layers]
    x = torch.randn(rows, K, device=dev, dtype=torch.bfloat16)
    tw = torch.softmax(torch.randn(rows, TOPK, device=dev), -1)
    outs = [torch.empty_like(x) for _ in layers]
    t = graph_time(lambda: [torch.ops.b70w3.moe_decode_w3(o, x, tw, i, mp, l["gu"], l["gus"], l["gue"], l["dn"],
                                                          l["dns"], l["dne"]) for l, i, o in zip(layers, ids, outs)])
    ue = sum(len({g for g in i.flatten().tolist() if g < E_LOCAL}) for i in ids) / L
    gbs = ue * 1.92e6 / (t / L) / 1e9
    print(f"rows {rows:2d}: experts/layer {ue:4.1f} | {t / L * 1e6:7.1f} us/layer | 48 layers {t / L * 48 * 1e3:6.2f} ms "
          f"| expert bytes {gbs:5.0f} GB/s", flush=True)
