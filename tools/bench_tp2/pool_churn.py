"""Fixed-pool replay: per-step DMA volume vs UVA reads avoided. pool_churn.py <trace-rank0.jsonl>"""
import json, collections, sys
lines = open(sys.argv[1]).read().splitlines()
cold = {L: set(v) for L, v in json.loads(lines[0])["cold"].items()}
need = [{L: set(json.loads(l)[L]) & cold[L] for L in cold} for l in lines[1:]]
for P in (4, 8):
    pools = {L: set() for L in cold}; hits = loads = misses = 0
    for t in range(40, len(need)):
        for L in cold:
            cand = collections.Counter()
            for k in range(t - 2 - 7, t - 1):
                for e in need[k][L]: cand[e] += 1
            new = {e for e, _ in cand.most_common(P)}
            loads += len(new - pools[L]); pools[L] = new
            hits += len(need[t][L] & new); misses += len(need[t][L] - new)
    n = len(need) - 40
    MB = 2.46
    print(f"P={P}: per step  hits {hits/n:6.1f} experts ({hits/n*MB:5.0f} MB of UVA reads avoided)  "
          f"DMA loads {loads/n:6.1f} experts ({loads/n*MB:5.0f} MB)  remaining UVA {misses/n*MB:5.0f} MB")
