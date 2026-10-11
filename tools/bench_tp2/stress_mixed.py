"""Mixed load: 2 streams decoding 600 tokens while 3 long prompts (8K-20K tokens) prefill; checks all succeed."""
import json, threading, time, urllib.request
BASE = "http://localhost:8000"; res = []
def go(msg, n, tag):
    q = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": msg}], "max_tokens": n,
         "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(q).encode(),
                      {"Content-Type": "application/json"}), timeout=900))
        res.append((tag, "ok", r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"], round(time.time() - t, 1)))
    except Exception as e:
        res.append((tag, f"FAIL {e}", 0, 0, 0))
w = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor".split()
th = [threading.Thread(target=go, args=("Write a long essay about the history of computing.", 600, "decode1")),
      threading.Thread(target=go, args=("Объясни подробно теорию относительности.", 600, "decode2"))]
for i, n in enumerate([6000, 10000, 15000]):
    th.append(threading.Thread(target=go, args=(" ".join(w[(j*3+i) % len(w)] for j in range(n)) + "\nSummarize.", 64, f"long{n}")))
for t in th: t.start(); time.sleep(0.5)
for t in th: t.join()
for r in sorted(res): print("  ", r)
