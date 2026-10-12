#!/usr/bin/env python3
"""verify-tree.py <root>... — every file the series touches must carry the "final" sha256 in image/verify-overlay.sh.

<root> is a directory that holds `vllm/` (a source tree, or an image's site-packages). Checked: the wu1ff layer
(PY_FILES) and the b70 series result (B70_FILES, which overrides PY_FILES where both name a file). Binaries are not
checked here (verify-overlay.sh ABS_FILES does that for the 4-card image). Exit 0 = all match. Standard library only.
"""
import hashlib, pathlib, re, sys

vo = (pathlib.Path(__file__).with_name("verify-overlay.sh")).read_text()


def block(name):
    body = re.search(name + r"='\n(.*?)'", vo, re.S).group(1)
    return [l.split() for l in body.splitlines() if l.strip()]


want = {rel: sha for sha, rel in block("PY_FILES")}                 # wu1ff layer
want.update({rel: sha for sha, _tag, rel in block("B70_FILES")})     # the series result overrides
rc = 0
for root in map(pathlib.Path, sys.argv[1:] or ["."]):
    bad = 0
    for rel, sha in sorted(want.items()):
        p = root / rel
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
        if got != sha:
            bad += 1
            print(f"DIFFERENT {p}: want {sha} got {got}", file=sys.stderr)
    print(f"hash check {root}: {len(want) - bad} of {len(want)} files carry the verify-overlay.sh 'final' sha256")
    rc |= bool(bad)
sys.exit(rc)
