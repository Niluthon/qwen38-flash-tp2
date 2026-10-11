"""Paired comparison: compare.py <tagA> <tagB> — MMLU accuracy + discordant pairs (exact McNemar), HumanEval pass@1."""
import json, math, sys
a, b = sys.argv[1:3]
A = {x["i"]: x["pred"] == x["gold"] for x in json.load(open(f"/work/eval/{a}-mmlu.json"))}
B = {x["i"]: x["pred"] == x["gold"] for x in json.load(open(f"/work/eval/{b}-mmlu.json"))}
k = sorted(set(A) & set(B)); n01 = sum(A[i] and not B[i] for i in k); n10 = sum(B[i] and not A[i] for i in k)
n = n01 + n10; p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
print(f"MMLU ({len(k)} paired): {a} {sum(A[i] for i in k)/len(k):.4f}  {b} {sum(B[i] for i in k)/len(k):.4f}  "
      f"only-{a}-right {n01}, only-{b}-right {n10}, McNemar exact p={p:.3f}")
HA = json.load(open(f"/work/eval/{a}-humaneval-results.json")); HB = json.load(open(f"/work/eval/{b}-humaneval-results.json"))
d1 = [t for t in HA if HA[t] and not HB[t]]; d2 = [t for t in HB if HB[t] and not HA[t]]
print(f"HumanEval: {a} {sum(HA.values())}/164  {b} {sum(HB.values())}/164  only-{a}: {d1}  only-{b}: {d2}")
