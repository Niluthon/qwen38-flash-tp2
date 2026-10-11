"""Deterministic decode bench: fixed real prompts, temperature 0, thinking off. Reports tok/s, acceptance length
(from the engine's spec counters) and ms per engine step, for 1 stream (each prompt in turn) and 4 streams."""
import json, sys, time, threading, urllib.request
BASE = "http://localhost:8000"
P = ["Explain how a refrigerator moves heat from inside to outside, step by step, for a curious teenager.",
     "Write a Python function that parses an ISO-8601 date string without using datetime, with tests.",
     "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
     "Compare TCP and UDP for a game developer: latency, reliability, ordering, and when to use each."]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
def metric(name):
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))
def gen(p, out, i):
    q = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": p}], "max_tokens": N,
         "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(q).encode(),
                  {"Content-Type": "application/json"}), timeout=900))
    out[i] = r["usage"]["completion_tokens"]
def cell(prompts):
    d0, a0 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
    out = [0] * len(prompts); th = [threading.Thread(target=gen, args=(p, out, i)) for i, p in enumerate(prompts)]
    t = time.time(); [x.start() for x in th]; [x.join() for x in th]; el = time.time() - t
    d, a = metric("vllm:spec_decode_num_drafts_total") - d0, metric("vllm:spec_decode_num_accepted_tokens_total") - a0
    tok = sum(out); steps = d / len(prompts) if d else tok
    return tok, el, (a + d) / d if d else 1.0, el / steps * 1e3
for p in P:
    tok, el, acc, ms = cell([p]); print(f"  1 stream : {tok/el:6.1f} tok/s  accept {acc:.2f}  {ms:5.1f} ms/step  ({p[:30]})", flush=True)
tok, el, acc, ms = cell(P); print(f"  4 streams: {tok/el:6.1f} tok/s total  accept {acc:.2f}  {ms:5.1f} ms/step", flush=True)
