"""MTP acceptance and decode speed on real chat prompts (server default sampling), from the engine's counters."""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BASE  # noqa: E402
from bench2 import external_posts, wait_idle  # noqa: E402

PROMPTS = [
    "Explain how a refrigerator moves heat from inside to outside, step by step, for a curious teenager.",
    "Write a Python function that parses an ISO-8601 date string without using datetime, with tests.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
]


def counters() -> dict:
    text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {"drafts": 0.0, "accepted": 0.0, "pos": {}}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] += float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] += float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos = line.split('position="')[1].split('"')[0]
            out["pos"][pos] = out["pos"].get(pos, 0.0) + float(line.split()[-1])
    return out


if __name__ == "__main__":
    waited = wait_idle()
    before_posts, c0 = external_posts(), counters()
    tokens, seconds = 0, 0.0
    for p in PROMPTS:
        q = {"model": "qwen38-flash-next", "max_tokens": 400,
             "messages": [{"role": "user", "content": p}],
             "chat_template_kwargs": {"enable_thinking": False}}
        t = time.perf_counter()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            BASE + "/v1/chat/completions", json.dumps(q).encode(),
            {"Content-Type": "application/json"}), timeout=900))
        dt = time.perf_counter() - t
        n = r["usage"]["completion_tokens"]
        tokens, seconds = tokens + n, seconds + dt
        print(f"  {n:4d} tokens in {dt:5.1f}s = {n / dt:5.1f} tok/s  ({p[:50]}...)", flush=True)
    c1 = counters()
    d = c1["drafts"] - c0["drafts"]
    acc = c1["accepted"] - c0["accepted"]
    per = [(c1["pos"][k] - c0["pos"].get(k, 0)) / d for k in sorted(c1["pos"])] if d else []
    others = external_posts() - before_posts
    print(f"overall {tokens / seconds:.1f} tok/s (incl. prefill); acceptance length {1 + acc / d:.2f} tokens/step; "
          f"per draft position {' / '.join(f'{x:.2f}' for x in per)}; waited {waited:.0f}s for idle"
          + (f"  !! {others} request(s) from other clients" if others else ""))
