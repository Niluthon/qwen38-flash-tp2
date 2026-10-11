"""Quality eval: MMLU (N sampled, zero-shot letter) + HumanEval (164, temp 0). run_eval.py <tag> [N]
Writes /work/eval/<tag>-mmlu.json (per-question answers) and /work/eval/<tag>-humaneval.json (completions)."""
import json, random, re, sys, threading, queue, urllib.request
import pandas as pd
BASE = "http://localhost:8000"; tag = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
def chat(content, n):
    q = {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": content}], "max_tokens": n,
         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(q).encode(),
                  {"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"] or ""
def pool(items, fn, width=4):
    out = {}; q = queue.Queue(); [q.put(x) for x in items]
    def w():
        while True:
            try: i, x = q.get_nowait()
            except queue.Empty: return
            out[i] = fn(x)
    th = [threading.Thread(target=w) for _ in range(width)]; [t.start() for t in th]; [t.join() for t in th]
    return out
mm = pd.read_parquet("/work/datasets/cais_mmlu/all/test-00000-of-00001.parquet")
idx = sorted(random.Random(1234).sample(range(len(mm)), N))
def mmlu_q(i):
    r = mm.iloc[i]; ch = list(r["choices"])
    p = (f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n{r['question']}\n" +
         "".join(f"{'ABCD'[k]}. {c}\n" for k, c in enumerate(ch)) + "\nAnswer with the letter only.")
    a = chat(p, 4); m = re.search(r"[ABCD]", a)
    return {"i": int(i), "pred": m.group(0) if m else None, "gold": "ABCD"[int(r["answer"])]}
res = pool(list(enumerate(idx)), lambda x: mmlu_q(x))
acc = sum(r["pred"] == r["gold"] for r in res.values()) / len(res)
json.dump(list(res.values()), open(f"/work/eval/{tag}-mmlu.json", "w"))
print(f"MMLU {tag}: {acc:.4f} on {len(res)} questions", flush=True)
he = pd.read_parquet("/work/datasets/openai_openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
def he_q(i):
    r = he.iloc[i]
    a = chat("Complete the following Python function. Return the complete function (with imports) in one ```python code block, no explanations.\n\n```python\n" + r["prompt"] + "```", 768)
    m = re.search(r"```(?:python)?\n(.*?)```", a, re.S)
    return {"task_id": r["task_id"], "code": m.group(1) if m else a, "test": r["test"], "entry_point": r["entry_point"]}
hres = pool(list(enumerate(range(len(he)))), lambda x: he_q(x))
json.dump([hres[k] for k in sorted(hres)], open(f"/work/eval/{tag}-humaneval.json", "w"))
print(f"HumanEval {tag}: {len(hres)} completions written", flush=True)
