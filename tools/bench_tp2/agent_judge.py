"""Summarize agent dumps: cap hits, repeat-loop rate, length. agent_judge.py a.json [b.json ...]"""
import json, re, sys
def loop(t):  # a line or 20+-char span repeated many times = degenerate
    lines = [l.strip() for l in t.splitlines() if len(l.strip()) > 20]
    if lines and max(lines.count(l) for l in set(lines)) >= 6: return True
    return any(re.search(r"\w", m.group(1)) for m in re.finditer(r"(.{1,8})\1{15,}", t, re.S))
for f in sys.argv[1:]:
    d = json.load(open(f)); n = len(d)
    cap = sum(x["finish"] == "length" for x in d); lp = sum(loop(x["text"]) for x in d)
    print(f"{f.split('/')[-1]:28s} responses {n}  hit 500-token cap {cap} ({cap/n:.0%})  loops {lp}  mean chars {sum(len(x['text']) for x in d)/n:.0f}")
