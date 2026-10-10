#!/usr/bin/env python3
"""fp8ple_snapshot.py — serve snapshot of an Intel-AutoRound checkpoint whose PLE table is FP8 (albucino hybrid).

albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE @ ef554143 is Intel's AutoRound target tensors (the same quantization block
as engines/serve-config-intel-autoround.json) plus RadixArk's FP8 PLE table in `model-plefp8-*.safetensors`. The
series serves the INT8 table built from those shards (tools/build_int8_ple_from_fp8.py), so the snapshot:

  1. symlinks every file of the checkpoint except config.json and the index (relative links: keep both on one volume);
  2. writes an index without the PLE table tensors (`ngram_embedding.shard_<i>.weight` and the table's
     `ngram_embedding.weight_scale`) and without `self_attn.indexer.*` (unused on the dense-QSA config);
  3. writes config.json = the given serve config, after checking that the checkpoint's config differs from it only in
     the indexer_* keys and `ple_embedding_dtype` (the quantization block must be identical).

    fp8ple_snapshot.py <checkpoint dir> <engines/serve-config-intel-autoround.json> <new snapshot dir>
    fp8ple_snapshot.py --keep-quant <runtime/mtp-int4-g32> <engines/serve-config-intel-autoround.json> <new dir>
        the same for the checkpoint's MTP draft (a separate compressed-tensors model, `speculative_config.model`):
        config.json = the serve config with the draft's own quantization block.

Nothing in the checkpoint is modified. Standard library only.
"""
import json
import os
import re
import sys

DROP = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.(shard_\d+\.weight|weight_scale)$|\.self_attn\.indexer\.")
SKIP = {"config.json", "model.safetensors.index.json"}
ALLOWED_DIFF = re.compile(r"^text_config\.(indexer_\w+|ple_embedding_dtype)$")


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "quantization_config":
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def main(src: str, serve_config: str, dst: str, keep_quant: bool = False) -> int:
    src = os.path.abspath(src)
    if os.path.exists(dst):
        print(f"{dst} exists", file=sys.stderr)
        return 1
    with open(serve_config) as f:
        serve = json.load(f)
    with open(os.path.join(src, "config.json")) as f:
        ckpt = json.load(f)
    if keep_quant:
        serve["quantization_config"] = ckpt["quantization_config"]
    fs, fc = flat(serve), flat(ckpt)
    diff = sorted(k for k in set(fs) | set(fc) if fs.get(k, "<absent>") != fc.get(k, "<absent>"))
    bad = [k for k in diff if not ALLOWED_DIFF.match(k)]
    if bad:
        print(f"serve config and checkpoint config differ beyond indexer/PLE dtype keys: {bad}", file=sys.stderr)
        return 1
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wm = index["weight_map"]
    kept = {k: v for k, v in wm.items() if not DROP.search(k)}
    missing = sorted(n for n in set(kept.values()) if not os.path.isfile(os.path.join(src, n)))
    if missing:
        print(f"checkpoint files missing: {missing}", file=sys.stderr)
        return 1
    index["weight_map"] = kept
    os.makedirs(dst)
    for name in sorted(os.listdir(src)):
        if name in SKIP or name.startswith(".") or name.endswith(".bak"):
            continue
        os.symlink(os.path.relpath(os.path.join(src, name), dst), os.path.join(dst, name))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(serve, f, indent=2)
        f.write("\n")
    print(f"{dst}: {len(kept)} of {len(wm)} tensors indexed ({len(wm) - len(kept)} dropped); "
          f"config = {os.path.basename(serve_config)}; config diffs accepted: {diff}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    keep = argv[:1] == ["--keep-quant"]
    if keep:
        argv = argv[1:]
    if len(argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*argv, keep_quant=keep))
