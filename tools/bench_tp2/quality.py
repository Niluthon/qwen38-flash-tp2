"""Quality spot-check: needle retrieval at several depths/lengths + short factual/math answers (temperature 0).
Writes answers to /work/quality-<tag>.json for a side-by-side diff between configurations."""
import json, sys, random, urllib.request
BASE = "http://localhost:8000"; tag = sys.argv[1]
def chat(msgs, n=64):
    q = {"model": "qwen38-flash-next", "messages": msgs, "max_tokens": n, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(q).encode(),
                  {"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"]
rnd = random.Random(7); topics = "river mountain city forest harbor desert valley island bridge tower".split()
def filler(n): return " ".join(f"The {rnd.choice(topics)} number {rnd.randint(1,9999)} was quiet that day." for _ in range(n))
res = {"needle": [], "facts": []}; ok = 0; tot = 0
for sents, depth in [(2500, .1), (2500, .5), (2500, .9), (5000, .3), (5000, .7), (7500, .5)]:
    code = f"{rnd.randint(100000, 999999)}"; parts = filler(sents).split(". ")
    k = int(len(parts) * depth); parts.insert(k, f"The secret access code for vault Kestrel is {code}")
    a, pt = chat([{"role": "user", "content": ". ".join(parts) + "\n\nWhat is the secret access code for vault Kestrel? Reply with the number only."}], 16)
    hit = code in a; ok += hit; tot += 1; res["needle"].append([pt, depth, code, a, hit])
    print(f"  needle {pt:6d} tok depth {depth:.1f}: {'OK ' if hit else 'MISS'} {a.strip()[:30]!r}", flush=True)
for qn in ["What is 17*23? Number only.", "Capital of Australia? One word.", "Who wrote 'War and Peace'? Name only.",
           "Сколько будет 144/12? Только число.", "What is the boiling point of water at sea level in Celsius? Number only.",
           "Write a haiku about autumn.", "List the first 8 prime numbers, comma separated."]:
    a, _ = chat([{"role": "user", "content": qn}], 48); res["facts"].append([qn, a]); print(f"  {qn[:40]:40s} -> {a.strip()[:60]!r}", flush=True)
json.dump(res, open(f"/work/quality-{tag}.json", "w"), ensure_ascii=False, indent=1); print(f"  needles {ok}/{tot}")
