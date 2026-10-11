"""Reproduce known issue 13 (docs/known-issues.md): one chat request with a long synthetic prompt.

    python repro.py <prompt_lines> <thinking 0|1> [max_tokens] [temperature]

1,050 lines are ~17K tokens (more than one 4,096-token prefill chunk). As the FIRST request after an engine start
(WARMUP=0) it stops the MTP engine; after the warm-up it must print OK. B70_URL overrides http://localhost:8000.
"""
import json, os, random, sys, time, urllib.request

BASE = os.environ.get("B70_URL", "http://localhost:8000")
lines, thinking = int(sys.argv[1]), sys.argv[2] == "1"
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 400
random.seed(lines)
words = "river stone garden lamp window market silver cloud harbor letter mountain candle forest bridge winter".split()
doc = "\n".join(" ".join(random.choice(words) for _ in range(14)) + "." for _ in range(lines))
q = {"model": "qwen38-flash-next", "max_tokens": max_tokens, **({"temperature": float(sys.argv[4])} if len(sys.argv) > 4 else {}),
     "messages": [{"role": "user", "content": doc + "\n\nHow many lines are above? Answer briefly."}],
     "chat_template_kwargs": {"enable_thinking": thinking}}
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(q).encode(),
        {"Content-Type": "application/json"}), timeout=900))
    print(f"OK lines={lines} thinking={thinking} usage={r['usage']['prompt_tokens']}+{r['usage']['completion_tokens']} {time.time()-t:.0f}s")
except Exception as e:
    print(f"FAILED lines={lines} thinking={thinking}: {e}")
