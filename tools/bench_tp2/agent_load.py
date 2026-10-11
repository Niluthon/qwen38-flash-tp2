"""Agentic-coding load: several concurrent sessions, each with a tool-use system prompt, real source files as
context and 5 follow-up turns (implement, test, fix, refactor, explain). Model output is fed back each turn."""
import json, os, random, sys, threading, time, urllib.request
BASE = "http://localhost:8000"; rnd = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 6; PAR = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SRC = os.environ.get("AGENT_CORPUS", "/work/agent-corpus")  # a frozen copy: edits to the live tree must not change the sessions
files = [os.path.join(d, f) for d, _, fs in os.walk(SRC + "/vllm") for f in fs if f.endswith(".py")]
files += [os.path.join(d, f) for d, _, fs in os.walk(SRC + "/csrc") for f in fs if f.endswith((".cpp", ".cu", ".h"))]
files = sorted(f for f in files if 3000 < os.path.getsize(f) < 20000)
SYSTEM = ("You are an autonomous coding agent working in a git repository. You can call tools: read_file(path), "
          "write_file(path, content), run(cmd), search(pattern). Think about the change, then answer with the code "
          "edits and the commands to run. Keep answers focused and complete.\n" + "\n".join(
          f"Tool {n}: {json.dumps({'name': n, 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}}})}"
          for n in ("read_file", "write_file", "run", "search")))
TURNS = ["Read this file and implement a small improvement: add input validation with clear error messages where it is missing. Show the full diff.",
         "Write pytest unit tests for the functions you changed, including edge cases.",
         "One test fails with a TypeError on None input. Find the cause and fix it; show the corrected code.",
         "Refactor the longest function into smaller helpers without changing behavior. Show the new code.",
         "Write a concise commit message and a short explanation of the changes for a reviewer."]
lock = threading.Lock(); stats = []; dumps = []
def post(msgs, n):
    q = {"model": "qwen38-flash-next", "messages": msgs, "max_tokens": n, "temperature": float(os.environ.get("AGENT_TEMP", "0.7")), "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions",
        json.dumps(q).encode(), {"Content-Type": "application/json"}), timeout=1800))
    c = r["choices"][0]
    if os.environ.get("AGENT_DUMP"):
        with lock: dumps.append({"finish": c.get("finish_reason"), "text": c["message"]["content"] or ""})
    return c["message"]["content"] or "", r["usage"], time.time() - t
def session(i):
    f = rnd.choice(files); code = open(f, errors="replace").read()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"File {f[len(SRC)+1:]}:\n```\n{code}\n```\n{TURNS[0]}"}]
    for t in range(len(TURNS)):
        if t: msgs.append({"role": "user", "content": TURNS[t]})
        out, u, el = post(msgs, 500); msgs.append({"role": "assistant", "content": out})
        with lock: stats.append((u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, u["completion_tokens"], el))
q = list(range(SESSIONS))
def worker():
    while True:
        with lock:
            if not q: return
            i = q.pop()
        session(i)
t0 = time.time(); th = [threading.Thread(target=worker) for _ in range(PAR)]; [x.start() for x in th]; [x.join() for x in th]
pt = sum(s[0] for s in stats); ct = sum(s[1] for s in stats); gt = sum(s[2] for s in stats)
print(f"agent load: {len(stats)} requests, prompt {pt} tok ({ct} cached = {ct/max(pt,1):.0%}), generated {gt} tok, "
      f"{time.time()-t0:.0f} s, gen {gt/(time.time()-t0):.1f} tok/s overall")
if os.environ.get("AGENT_DUMP"):
    json.dump(dumps, open(os.environ["AGENT_DUMP"], "w"), ensure_ascii=False)
