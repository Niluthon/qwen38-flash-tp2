"""Streaming benchmark: TTFT / prefill rate and decode rate per stream (tools/bench_tp2, docs/two-cards.md).

    python bench.py [<streams:words:gen> ...]     e.g. 1:20:256 4:20:256 1:11000:64

Words of synthetic text, not tokens (~1.07 tokens per word). B70_URL overrides http://localhost:8000.
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("B70_URL", "http://localhost:8000")

URL = BASE + "/v1/completions"
WORDS = ("the quick brown fox jumps over a lazy dog while seven wizards quietly brew potent "
         "elixirs beneath ancient oak trees and distant thunder rolls across silver hills ").split()


def prompt_of(n_words: int, salt: int) -> str:
    body = " ".join(WORDS[(i * 7 + salt) % len(WORDS)] for i in range(n_words))
    return f"[{salt}] Summarise the following text in one sentence.\n\n{body}\n\nSummary:"


def one(prompt: str, max_tokens: int, out: list) -> None:
    req = urllib.request.Request(URL, data=json.dumps({
        "model": "qwen38-flash-next", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            if msg.get("usage"):
                usage = msg["usage"]
            if first is None and msg.get("choices") and msg["choices"][0].get("text"):
                first = time.perf_counter()
    end = time.perf_counter()
    out.append({"ttft": first - t0, "decode_tps": (usage["completion_tokens"] - 1) / (end - first),
                "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]})


def run(streams: int, n_words: int, max_tokens: int, salt: int) -> None:
    out: list = []
    th = [threading.Thread(target=one, args=(prompt_of(n_words, salt + i), max_tokens, out))
          for i in range(streams)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    p = out[0]["prompt_tokens"]
    per = sum(r["decode_tps"] for r in out) / len(out)
    ttft = max(r["ttft"] for r in out)
    print(f"streams={streams:2d} prompt={p:6d} gen={max_tokens}: ttft {ttft:6.2f}s "
          f"(prefill {streams * p / ttft:7.0f} tok/s)  decode {per:6.1f} tok/s/stream, "
          f"{per * streams:6.1f} total  wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    salt = int(time.time()) % 100000  # fresh prompts: no prefix-cache hits across runs
    for spec in sys.argv[1:] or ["1:20:256", "1:20:256", "4:20:256", "1:2800:64", "1:11000:64"]:
        s, w, g = (int(x) for x in spec.split(":"))
        run(s, w, g, salt)
        salt += 100
