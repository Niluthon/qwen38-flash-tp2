"""Fixed-pool replay: cold-route hit rate by pool size, lag and fill policy. pool_reuse.py <trace-rankRANK.jsonl>
(B70_MOE_ROUTE_TRACE output; RANK is replaced by 0 and 1)."""
import json, sys, collections
for rank in (0, 1):
    lines = open(fsys.argv[1].replace("RANK", str(rank))).read().splitlines()
    cold = {L: set(v) for L, v in json.loads(lines[0])["cold"].items()}
    steps = [json.loads(l) for l in lines[1:]]
    layers = sorted(cold, key=int)
    need = [{L: set(s[L]) & cold[L] for L in layers} for s in steps]
    avg_need = sum(len(n[L]) for n in need for L in layers) / (len(need) * len(layers))
    print(f"rank {rank}: {len(steps)} steps, cold experts needed per layer per step: {avg_need:.2f}")
    for P in (2, 4, 8):
        for lag in (1, 2):
            for policy in ("last", "freq8", "freq32"):
                hit = tot = 0
                for t in range(40, len(need)):
                    for L in layers:
                        n = need[t][L]
                        if not n: continue
                        src = t - lag
                        if policy == "last":
                            cand = collections.Counter({e: 1 for e in need[src][L]})
                            for k in range(src - 1, src - 4, -1):
                                for e in need[k][L]: cand[e] += 0.5 ** (src - k)
                        else:
                            W = int(policy[4:]); cand = collections.Counter()
                            for k in range(src - W + 1, src + 1):
                                for e in need[k][L]: cand[e] += 1
                        pool = {e for e, _ in cand.most_common(P)}
                        hit += len(n & pool); tot += len(n)
                if P == 4 or policy == "freq8":
                    print(f"  P={P} lag={lag} {policy:6s}: cold-route hit rate {hit/tot:.1%}")
