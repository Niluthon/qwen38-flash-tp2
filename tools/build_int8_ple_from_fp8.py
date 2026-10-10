#!/usr/bin/env python3
"""Build the INT8 per-row PLE table (patch 0008/0013 format) from a checkpoint's FP8 PLE shards.

For checkpoints that ship the PLE n-gram table as FP8 E4M3 with one global scale (RadixArk's table, as in
albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE): 128 tensors `...ple_embedding.ngram_embedding.shard_<i>.weight`
[2500012, 160] F8_E4M3 + `...ngram_embedding.weight_scale` [1] BF16, spread over several .safetensors files.

Same output format as tools/build_int8_ple.py (`lumnus-ple-int8-rowscale/v1`, data at a 4096-byte boundary,
table U8 [320001536, 164] then the three I64 layout tensors), so 0008 and 0013 serve it unchanged. Each row is
dequantised exactly (fp8 value x scale in float32) and re-quantised as in build_int8_ple.quantize():
s = absmax/127, q = round_half_even(x/s) clamped to +-127.

Modes (CKPT = the checkpoint dir, OUT_DIR = where the table goes):
  build   stream every shard, write OUT.part with a running sha256, fdatasync, rename, write OUT.ok.
          Refuses to overwrite an existing OUT.
  verify  header/layout checks, and on a stratified sample (4096 strata x 64 rows, seed 20260928):
          rel-L2 error of the INT8 rows against the FP8 rows they came from, plus byte-exact
          re-quantisation of every sampled row. Prints JSON.
"""
import hashlib, json, os, re, struct, sys, time
import numpy as np
import torch

CKPT = os.environ.get("CKPT", "/models/qwen38-w4a16")
OUT_DIR = os.environ.get("OUT_DIR", "/models/qwen38-w4a16-int8ple")
NAME = "ple_ngram_int8_rowscale.safetensors"
ROWS, D, W = 320_001_536, 160, 164
CHUNK = int(os.environ.get("CHUNK_ROWS", str(1 << 20)))
FORMAT = "lumnus-ple-int8-rowscale/v1"
LAYOUT_KEYS = ["ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers"]
PFX = "model.language_model.layers.1.ple.ple_embedding."
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:8.1f}s]", *a, flush=True)


def tensor_index():
    """name -> (file path, dtype, shape, absolute byte offset) for every PLE embedding tensor."""
    wm = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted({v for k, v in wm.items() if k.startswith(PFX)}):
        path = os.path.join(CKPT, f)
        with open(path, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            hdr = json.loads(h.read(n))
        for k, v in hdr.items():
            if k.startswith(PFX):
                out[k] = (path, v["dtype"], v["shape"], 8 + n + v["data_offsets"][0])
    return out


def shards(idx):
    sh = {}
    for k, v in idx.items():
        m = re.search(r"ngram_embedding\.shard_(\d+)\.weight$", k)
        if m:
            sh[int(m.group(1))] = v
    order = sorted(sh)
    assert order == list(range(len(order))), "shard ids must be 0..n-1"
    for i in order:
        assert sh[i][1] == "F8_E4M3" and sh[i][2][1] == D, sh[i]
    assert sum(sh[i][2][0] for i in order) == ROWS, sum(sh[i][2][0] for i in order)
    return [sh[i] for i in order]


def global_scale(idx) -> float:
    path, dtype, shape, off = idx[PFX + "ngram_embedding.weight_scale"]
    assert dtype == "BF16" and shape == [1], (dtype, shape)
    with open(path, "rb") as h:
        h.seek(off)
        raw = h.read(2)
    s = float(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()[0])
    assert np.isfinite(s) and s > 0, s
    return s


def layout_tensors(idx):
    out = {}
    for k in LAYOUT_KEYS:
        path, dtype, shape, off = idx[PFX + k]
        assert dtype == "I64", (k, dtype)
        n = int(np.prod(shape))
        with open(path, "rb") as h:
            h.seek(off)
            out[k] = torch.from_numpy(np.frombuffer(h.read(n * 8), dtype="<i8").copy()).reshape(shape)
    assert out["ngram_heads_offsets"].numel() == 16 and out["layer_multipliers"].numel() == 3
    return out


def fp8_to_f32(codes: np.ndarray, scale: float) -> torch.Tensor:
    return torch.from_numpy(codes).view(torch.float8_e4m3fn).to(torch.float32) * scale


def quantize(x: torch.Tensor) -> torch.Tensor:
    """[n,160] float32 -> [n,164] uint8 packed rows (build_int8_ple.quantize, from float32)."""
    s = x.abs().amax(1, keepdim=True) / 127.0
    q = (x / s.clamp_min(1e-30)).round().clamp_(-127, 127).to(torch.int8)
    out = torch.empty(x.shape[0], W, dtype=torch.uint8)
    out[:, :D] = q.view(torch.uint8)
    out[:, D:] = s.contiguous().view(torch.uint8).reshape(-1, 4)  # host little-endian
    return out


def header_bytes(layout, scale):
    meta = {
        "format": FORMAT,
        "row_layout": "160 x int8 q | float32 LE s; s=absmax/127, q=round_half_even(x/s) clamp +-127; x~=q*s",
        "source_file": f"{os.path.basename(os.path.normpath(CKPT))}: FP8 E4M3 ngram_embedding shards, global scale {scale!r}",
        "recipe": "github.com/Lumnus/b70-flash-next tools/build_int8_ple_from_fp8.py",
    }
    hdr = {"__metadata__": meta, "table": {"dtype": "U8", "shape": [ROWS, W], "data_offsets": [0, ROWS * W]}}
    pos = ROWS * W
    for k in LAYOUT_KEYS:
        n = layout[k].numel() * 8
        hdr[k] = {"dtype": "I64", "shape": list(layout[k].shape), "data_offsets": [pos, pos + n]}
        pos += n
    js = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
    js += b" " * ((-(8 + len(js))) % 4096)
    return len(js).to_bytes(8, "little") + js, pos


def build():
    out_path = os.path.join(OUT_DIR, NAME)
    if os.path.exists(out_path):
        sys.exit(f"refuse: {out_path} exists")
    os.makedirs(OUT_DIR, exist_ok=True)
    idx = tensor_index()
    parts, scale, layout = shards(idx), global_scale(idx), layout_tensors(idx)
    log(f"{len(parts)} FP8 shards, global scale {scale!r}, layout", {k: v.tolist() for k, v in layout.items()})
    hdr, data_len = header_bytes(layout, scale)
    h = hashlib.sha256()
    part = out_path + ".part"
    fout = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fout, hdr); h.update(hdr)
    written, synced, done = len(hdr), 0, 0
    for si, (path, _, shape, off) in enumerate(parts):
        fin = os.open(path, os.O_RDONLY)
        for r0 in range(0, shape[0], CHUNK):
            n = min(CHUNK, shape[0] - r0)
            a = off + r0 * D
            buf = os.pread(fin, n * D, a)
            assert len(buf) == n * D
            os.posix_fadvise(fin, a, n * D, os.POSIX_FADV_DONTNEED)
            packed = quantize(fp8_to_f32(np.frombuffer(buf, dtype=np.uint8).reshape(n, D), scale))
            b = packed.numpy().tobytes()
            mv = memoryview(b)
            while mv:
                k = os.write(fout, mv); mv = mv[k:]
            h.update(b); written += len(b); done += n
            if written - synced > (4 << 30):
                os.fdatasync(fout); os.posix_fadvise(fout, 0, written, os.POSIX_FADV_DONTNEED); synced = written
        os.close(fin)
        if si % 8 == 7:
            log(f"shard {si + 1}/{len(parts)}  rows {done}/{ROWS} ({100 * done / ROWS:.1f}%)")
    for k in LAYOUT_KEYS:
        b = layout[k].numpy().astype("<i8").tobytes()
        os.write(fout, b); h.update(b); written += len(b)
    os.fdatasync(fout); os.posix_fadvise(fout, 0, written, os.POSIX_FADV_DONTNEED)
    os.close(fout)
    assert written == len(hdr) + data_len, (written, len(hdr), data_len)
    sha = h.hexdigest()
    os.rename(part, out_path)
    with open(out_path + ".ok", "w") as f:
        f.write(f"{sha}  {NAME}\n")
    log(f"DONE {out_path} {written} B sha256 {sha}")


def verify(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    ds = 8 + n
    assert ds % 4096 == 0, ds
    assert hdr["__metadata__"]["format"] == FORMAT
    assert hdr["table"]["shape"] == [ROWS, W]
    idx = tensor_index()
    parts, scale = shards(idx), global_scale(idx)
    rng = np.random.default_rng(20260928)
    edges = np.linspace(0, ROWS, 4096 + 1).astype(np.int64)
    rows = np.unique(np.concatenate([rng.integers(edges[i], edges[i + 1], 64) for i in range(4096)]))
    per = parts[0][2][0]
    fp8 = np.empty((rows.size, D), np.uint8)
    got = np.empty((rows.size, W), np.uint8)
    with open(path, "rb") as fo:
        for j, r in enumerate(rows):
            p, _, _, off = parts[r // per]
            with open(p, "rb") as fi:
                fi.seek(off + (r % per) * D); fp8[j] = np.frombuffer(fi.read(D), np.uint8)
            fo.seek(ds + int(r) * W); got[j] = np.frombuffer(fo.read(W), np.uint8)
    x = fp8_to_f32(fp8, scale)
    want = quantize(x).numpy()
    q = torch.from_numpy(got[:, :D].copy()).view(torch.int8).float()
    s = torch.from_numpy(got[:, D:].copy()).view(torch.float32)
    deq = q * s
    num = (deq - x).norm(dim=1)
    den = x.norm(dim=1)
    rel = (num / den.clamp_min(1e-30))[den > 0]
    res = {
        "rows_sampled": int(rows.size),
        "byte_exact_requant": bool((got == want).all()),
        "scale_finite_nonneg": bool(torch.isfinite(s).all() and (s >= 0).all()),
        "rel_l2_vs_fp8": {"mean": float(rel.mean()), "p50": float(rel.median()),
                          "p99": float(rel.quantile(0.99)), "max": float(rel.max())},
        "zero_rows": int((den == 0).sum()),
    }
    print(json.dumps(res, indent=1))
    assert res["byte_exact_requant"] and res["scale_finite_nonneg"]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "build":
        build()
    elif mode == "verify":
        verify(sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT_DIR, NAME))
    else:
        sys.exit(__doc__)
