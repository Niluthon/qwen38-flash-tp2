"""Known issue 6 check: N bursts of 4 simultaneous requests (temp 0.7, 200 tokens); flags degenerate outputs
(NaN garbage: long token repeats, replacement chars, very low character diversity) and empty outputs."""
import json, re, sys, threading, urllib.request
BASE = "http://localhost:8000"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
P = ["Write a Python class for a bank account with deposit, withdraw and tests.",
     "Explain in detail how HTTPS establishes a secure connection.",
     "Напиши рассказ о космонавте, который вернулся домой через сто лет.",
     "Refactor this: def f(a,b):\n  if a>b: return a-b\n  else: return b-a\nand explain why."]
def bad(t):
    if not t or len(t) < 40: return "short"
    if "�" in t: return "replacement-char"
    # a unit with a letter or digit repeated 16+ times (whitespace/rules in ASCII diagrams are fine)
    if any(re.search(r"\w", m.group(1)) for m in re.finditer(r"(.{1,8})\1{15,}", t, re.S)): return "repeat"
    if len(set(t)) < 12: return "low-diversity"
    return None
res = []
def go(p, k):
    q = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": p}], "max_tokens": 200,
         "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(q).encode(),
                      {"Content-Type": "application/json"}), timeout=600))
        t = r["choices"][0]["message"]["content"] or ""; b = bad(t); res.append((k, b, t[:80]))
        if b: open(f"flagged-{k[0]}-{k[1]}.txt", "w").write(t)
    except Exception as e:
        res.append((k, f"error {e}", ""))
for b in range(N):
    th = [threading.Thread(target=go, args=(p, (b, i))) for i, p in enumerate(P)]
    [t.start() for t in th]; [t.join() for t in th]
badl = [r for r in res if r[1]]
print(f"bursts {N}, requests {len(res)}, bad {len(badl)}")
for r in badl[:8]: print("  ", r)
