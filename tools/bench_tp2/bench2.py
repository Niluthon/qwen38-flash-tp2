"""Benchmark cells that start only when the engine is idle; a cell is flagged if other clients sent requests meanwhile.

    python bench2.py <streams:words:gen> ...      (cells as in bench.py)

B70_ENGINE_LOG: the engine log restart-tp2.sh writes (default /work/engine.log, the work volume in the container);
other clients are counted from its access lines. A cell that waited 900 s without an idle engine runs anyway
and prints "waited -1s": discard it.
"""
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

LOG = os.environ.get("B70_ENGINE_LOG", "/work/engine.log")


def gauge(name: str) -> float:
    text = urllib.request.urlopen(bench.BASE + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))


def external_posts() -> int:
    with open(LOG, errors="replace") as f:
        return sum(1 for l in f if "POST /v1/" in l and "127.0.0.1" not in l)


def wait_idle(quiet_s: float = 5.0, limit_s: float = 900.0) -> float:
    t0 = time.time()
    quiet_since = None
    while time.time() - t0 < limit_s:
        busy = gauge("vllm:num_requests_running") + gauge("vllm:num_requests_waiting")
        if busy == 0:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= quiet_s:
                return time.time() - t0
        else:
            quiet_since = None
        time.sleep(1)
    return -1


if __name__ == "__main__":
    salt = int(time.time()) % 100000
    for spec in sys.argv[1:]:
        s, w, g = (int(x) for x in spec.split(":"))
        waited = wait_idle()
        before = external_posts()
        bench.run(s, w, g, salt)
        salt += 100
        others = external_posts() - before
        flag = f"  !! {others} request(s) from other clients during this cell" if others else ""
        print(f"    (waited {waited:.0f}s for idle){flag}", flush=True)
