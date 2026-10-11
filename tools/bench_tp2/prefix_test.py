"""Prefix-cache test: the same long chat prompt twice, then the same + a new turn. Prints TTFT and cached tokens."""
import json, time, urllib.request, sys
BASE = "http://localhost:8000"
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
salt = int(time.time())
doc = f"Document {salt}. " + " ".join(words[(i * 7 + salt) % len(words)] + str(i % 97) for i in range(n))
def ask(msgs):
    q = {"model": "qwen38-flash-next", "messages": msgs, "max_tokens": 8, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); ttft = None; usage = None
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(q).encode(),
                               {"Content-Type": "application/json"}), timeout=900)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d = json.loads(line[5:])
        if ttft is None and d.get("choices") and d["choices"][0]["delta"].get("content"): ttft = time.time() - t0
        if d.get("usage"): usage = d["usage"]
    pt = usage["prompt_tokens"]; cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return ttft, pt, cached
sys_msg = {"role": "system", "content": "You are a careful assistant. " + doc}
m1 = [sys_msg, {"role": "user", "content": "How many words are in the document? Answer briefly."}]
for label, msgs in [("first", m1), ("same again", m1),
                    ("new turn", m1 + [{"role": "assistant", "content": "About many."},
                                       {"role": "user", "content": "And which word comes first?"}])]:
    ttft, pt, cached = ask(msgs)
    print(f"  prefix {label:10s}: prompt {pt} tok, cached {cached}, TTFT {ttft:.2f} s", flush=True)
