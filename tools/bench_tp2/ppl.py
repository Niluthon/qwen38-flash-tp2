"""Perplexity on fixed texts via prompt logprobs (temperature irrelevant). ppl.py <tag> -> /work/exp3/ppl-<tag>.json
Texts: vLLM source files (code), lumnus docs (English technical), a fixed Russian text. ~2K-token windows."""
import glob, json, math, os, sys, urllib.request
BASE = "http://localhost:8000"; tag = sys.argv[1]
texts = []
for f in [f for f in sorted(glob.glob("/work/vllm-b70/vllm/v1/core/*.py")) if os.path.getsize(f) > 8000][:6]: texts.append(("code", open(f).read()[:7000]))
for f in sorted(glob.glob("/lumnus/docs/*.md"))[:5]: texts.append(("docs", open(f).read()[:7000]))
ru = open(os.environ.get("PPL_RU_TEXT", "ru.txt")).read() if os.path.exists(os.environ.get("PPL_RU_TEXT", "ru.txt")) else ""
for i in range(0, min(len(ru), 21000), 7000): texts.append(("ru", ru[i:i + 7000]))
res = {}
for kind, t in texts:
    q = {"model": "qwen38-flash-next", "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", json.dumps(q).encode(),
                  {"Content-Type": "application/json"}), timeout=900))
    lp = [list(d.values())[0]["logprob"] for d in r["choices"][0]["prompt_logprobs"][1:] if d]
    res.setdefault(kind, []).append((sum(lp), len(lp)))
out = {}
for kind, v in res.items():
    s = sum(a for a, _ in v); n = sum(b for _, b in v); out[kind] = {"nll_per_tok": -s / n, "ppl": math.exp(-s / n), "tokens": n}
    print(f"  {kind:5s}: ppl {math.exp(-s/n):7.4f}  ({n} tokens)")
json.dump(out, open(f"ppl-{tag}.json", "w"), indent=1)
