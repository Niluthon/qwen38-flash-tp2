"""Run HumanEval tests (inside a --network none container). he_exec.py <file.json>"""
import json, subprocess, sys, tempfile, os
d = json.load(open(sys.argv[1])); ok = 0; res = {}
for x in d:
    src = x["code"] + "\n\n" + x["test"] + f"\n\ncheck({x['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src)
    try: p = subprocess.run([sys.executable, f.name], capture_output=True, timeout=15); good = p.returncode == 0
    except subprocess.TimeoutExpired: good = False
    os.unlink(f.name); res[x["task_id"]] = good; ok += good
json.dump(res, open(sys.argv[1].replace(".json", "-results.json"), "w"))
print(f"{os.path.basename(sys.argv[1])}: pass@1 {ok}/{len(d)} = {ok/len(d):.4f}")
