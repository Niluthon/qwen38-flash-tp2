"""Report for B70 step trace dumps (vllm/model_executor/b70_step_trace.py): where a decode step's time goes.

usage: step_trace_report.py <dump dir> [--tokens 4,16] [--rank 0,1] [--top 40]

A step = one target forward with T tokens (T = decode batch x (1 + MTP depth)) plus everything up to the next
target forward: the target logits, the draft passes and the unmarked time between them (sampler, rejection,
input preparation, host scheduling). Inside a forward each interval between two consecutive marks goes to the
innermost open module ("self time"); layer numbers are folded so the 48 layers add up. All times in ms per step.
"""
import argparse
import collections
import glob
import os
import re
import statistics

import torch

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--tokens", default="")
ap.add_argument("--rank", default="0,1")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--skip", type=int, default=0, help="drop the first N dump files per rank (warm-up)")
a = ap.parse_args()

LAYER = re.compile(r"(?:language_model\.)?(?:model\.)?layers\.(\d+)")


def fold(label: str, layer_types: dict) -> str:
    def rep(m):
        return "L[" + layer_types.get(int(m.group(1)), "?") + "]"
    s = LAYER.sub(rep, label)
    return s.replace("language_model.", "")


def load(rank: int):
    files = sorted(glob.glob(os.path.join(a.dir, f"rank{rank}-*.pt")))[a.skip:]
    rows = []
    meta = None
    cur = None
    for f in files:
        d = torch.load(f)
        meta = d
        inv = [{v: k for k, v in lab.items()} for lab in d["labels"]]
        for t, code in d["recs"].tolist():
            mid, val = (code >> 28) & 7, code & ((1 << 28) - 1)
            if code >> 31:
                cur = dict(seq=len(rows), model=d["names"][mid], mid=mid, tokens=val, t0=t, ev=[])
                rows.append(cur)
            elif cur is not None and cur["mid"] == mid:
                cur["ev"].append((t, inv[mid].get(val, f"slot{val}")))
    if meta is not None:
        meta = dict(meta, ticks_per_ns=1.0)  # host-stamped: ns
    return rows, meta


def layer_types(rows):
    lt = {}
    for row in rows:
        if row["model"] != "target":
            continue
        for _, lab in row["ev"]:
            m = LAYER.search(lab)
            if m:
                n = int(m.group(1))
                if ".linear_attn" in lab:
                    lt[n] = "lin"
                elif ".self_attn" in lab:
                    lt[n] = "attn"
    return lt


def self_times(row, lt, tpn):
    """innermost-open-module attribution of the intervals between consecutive marks of one row"""
    out = collections.defaultdict(float)
    stack = []
    prev = row["t0"]
    for t, lab in row["ev"]:
        key = stack[-1] if stack else "(top level)"
        out[fold(key, lt)] += (t - prev) / tpn / 1e6
        prev = t
        if lab.endswith(":s"):
            stack.append(lab[:-2])
        elif lab.endswith(":e"):
            name = lab[:-2]
            if name in stack:
                while stack and stack.pop() != name:
                    pass
    return out


def category(k: str) -> str:
    """coarse groups for the rollup (order matters)"""
    if "/allreduce" in k:
        return "all-reduces (TP2, incl. waiting for the other card)"
    if "apply_decode_op" in k:
        return "MoE routed experts: decode op (0048/0049)"
    if ".mlp.experts" in k:
        return "MoE glue: routing top-k, route counting, modular-kernel wrapper"
    if ".mlp.gate" in k or "shared_expert" in k or k.endswith(".mlp"):
        return "MoE router GEMM + shared expert"
    if "hyper_connection" in k or re.fullmatch(r"L\[\w+\]", k):
        return "hyper-connections (HC mix/combine, replicated)"
    if ".linear_attn" in k:
        return "linear attention (GDN): projections, conv, recurrence"
    if ".self_attn" in k:
        return "full attention: qkv/o projections, rope, attention"
    if ".ple" in k:
        return "PLE"
    if "logits" in k:
        return "logits (lm_head)"
    return "other (embedding, final mixer, glue)"


for rank in [int(x) for x in a.rank.split(",")]:
    rows, meta = load(rank)
    if not rows:
        print(f"rank {rank}: no rows")
        continue
    tpn = meta["ticks_per_ns"]
    lt = layer_types(rows)
    targets = [i for i, r in enumerate(rows) if r["model"] == "target"]
    want = {int(x) for x in a.tokens.split(",") if x} or None
    groups = collections.defaultdict(list)
    for k, i in enumerate(targets[:-1]):
        r = rows[i]
        if want is not None and r["tokens"] not in want:
            continue
        nxt = targets[k + 1]
        between = rows[i + 1:nxt]
        if any(b["model"] != "draft" for b in between):
            continue
        groups[r["tokens"]].append((r, between, rows[nxt]))
    print(f"\n######## rank {rank}: {len(rows)} rows, {meta['ticks_per_ns']:.4f} ticks/ns, "
          f"layer types: {collections.Counter(lt.values())}")
    for T, steps in sorted(groups.items()):
        if len(steps) < 5:
            continue
        if T > 64:
            continue  # prefill
        per = collections.defaultdict(list)
        comp_t = collections.defaultdict(float)
        comp_d = collections.defaultdict(float)
        ndraft = []
        for r, drafts, nxt in steps:
            ms = lambda x: x / tpn / 1e6
            step = ms(nxt["t0"] - r["t0"])
            ev = r["ev"]
            fend = next((t for t, l in ev if l == "forward.end"), ev[-1][0] if ev else r["t0"])
            tend = ev[-1][0] if ev else r["t0"]
            per["step period"].append(step)
            per["target forward"].append(ms(fend - r["t0"]))
            per["target after forward (logits)"].append(ms(tend - fend))
            d_in, d_gap, last = 0.0, 0.0, tend
            for dr in drafts:
                d_gap += ms(dr["t0"] - last)
                dend = dr["ev"][-1][0] if dr["ev"] else dr["t0"]
                d_in += ms(dend - dr["t0"])
                last = dend
            per["draft passes (marked)"].append(d_in)
            per["unmarked between target and drafts"].append(d_gap)
            per["unmarked after drafts, before next target"].append(ms(nxt["t0"] - last))
            ndraft.append(len(drafts))
            for k2, v in self_times(r, lt, tpn).items():
                comp_t[k2] += v
            for dr in drafts:
                for k2, v in self_times(dr, {}, tpn).items():
                    comp_d[k2] += v
        n = len(steps)
        print(f"\n=== T = {T} tokens per target forward: {n} steps, drafts per step {statistics.median(ndraft)} ===")
        print(f"{'step part':45s} {'median':>8s} {'mean':>8s}  (ms per step)")
        for k2, v in per.items():
            print(f"{k2:45s} {statistics.median(v):8.2f} {statistics.mean(v):8.2f}")
        for title, comp in (("target forward + logits", comp_t), ("draft passes", comp_d)):
            tot = sum(comp.values()) / n
            print(f"\n-- {title}: self time per module, ms per step (total {tot:.2f}) --")
            ar = sum(v for k2, v in comp.items() if "/allreduce" in k2) / n
            print(f"   all-reduces together: {ar:.2f} ms ({100 * ar / max(tot, 1e-9):.0f} %)")
            roll = collections.defaultdict(float)
            for k2, v in comp.items():
                roll[category(k2)] += v / n
            print("   by category:")
            for k2, v in sorted(roll.items(), key=lambda kv: -kv[1]):
                print(f"     {v:7.3f}  {100 * v / max(tot, 1e-9):5.1f} %  {k2}")
            for k2, v in sorted(comp.items(), key=lambda kv: -kv[1])[: a.top]:
                print(f"   {v / n:7.3f}  {100 * v / n / max(tot, 1e-9):5.1f} %  {k2}")
