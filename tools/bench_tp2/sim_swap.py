"""Offline replay of hot-set policies: sim_swap.py "<dir>/r0-*.json" [rankings.json]. Snapshots = copies of the
B70_MOE_ROUTE_COUNT=1 dump taken every period; reports the decode cold share for static, oracle and swap policies."""
import json, glob, sys, numpy as np
R = json.load(open(sys.argv[2] if len(sys.argv) > 2 else '/models/qwen38-w4a16/runtime/static_hot_cache_rankings.json'))
snaps = sorted(glob.glob(sys.argv[1]))
C = np.array([[json.load(open(f))[str(L)] for L in range(48)] for f in snaps], dtype=np.int64)  # [T,48,2,512]
D = np.diff(C, axis=0); D = D[D.sum(axis=(1, 2, 3)) > 0]
P, Dc = D[:, :, 0, :], D[:, :, 1, :]  # prefill, decode per window
H = 136; print(f"windows {len(D)}  prefill routes {P.sum()}  decode routes {Dc.sum()}")
prior = np.zeros((48, 512))
for L in range(48):
    auth = [int(g) for g in R.get(str(L), [])]
    for i, g in enumerate(auth): prior[L, g] = 1 - i / len(auth)
def hot_mask(score):
    m = np.zeros((48, 512), bool)
    for r in (0, 1):
        sl = slice(256 * r, 256 * r + 256)
        np.put_along_axis(m[:, sl], np.argsort(-score[:, sl], axis=1)[:, :H], True, axis=1)
    return m
static = hot_mask(prior)
cs = lambda W, m: (W * ~m).sum() / W.sum()
print(f"static checkpoint ranking           decode cold {cs(Dc, static):.3f}")
print(f"oracle per window                   decode cold {sum((w*~hot_mask(w+1e-3*prior)).sum() for w in Dc)/Dc.sum():.3f}")
def sim(signal, decay, budget, pw=0.05):
    m = static.copy(); ema = np.zeros((48, 512)); cold = 0; swaps = 0
    for t in range(len(D)):
        cold += (Dc[t] * ~m).sum()
        ema = decay * ema + signal[t]
        score = ema / max(ema.max(), 1) + pw * prior
        want = hot_mask(score)
        for L in range(48):
            for r in (0, 1):
                sl = np.arange(256 * r, 256 * r + 256)
                ins = [g for g in sl[np.argsort(-score[L, sl])] if want[L, g] and not m[L, g]][:budget]
                outs = [g for g in sl[np.argsort(score[L, sl])] if m[L, g] and not want[L, g]][:len(ins)]
                for a, b in zip(ins, outs):
                    if score[L, a] > 1.1 * score[L, b]:
                        m[L, a], m[L, b] = True, False; swaps += 1
    return cold / Dc.sum(), swaps / len(D) / 2
for name, sig in [("prefill counts", P), ("decode counts", Dc), ("prefill+decode", P + Dc)]:
    for decay in (0.8, 0.95):
        for budget in (4, 16):
            c, s = sim(sig, decay, budget)
            print(f"{name:15s} decay {decay:.2f} budget {budget:2d}: decode cold {c:.3f}  swaps/5s/rank {s:4.0f} ({s*4.9:4.0f} MB)")
