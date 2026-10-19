# Check B70 0048 draft-head FP8: quantization + fp8_gemm_w8a16 match a dequantized reference, top-1 agreement with
# the BF16 head on the real lm_head shard, and time per pass.  Run in the image with PYTHONPATH=/work/dev-tree.
import os
import sys

import torch
import torch.nn.functional as F
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C)
from safetensors import safe_open

sys.path.insert(0, os.environ.get("DEV_TREE", "/r/engine-data/dev-tree"))
os.environ["B70_DRAFT_HEAD_FP8"] = "1"
from vllm.models.qwen4_exp.nvidia import mtp  # noqa: E402

d = "xpu"
snap = os.environ.get("SNAP", "/m")
idx = __import__("json").load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{snap}/{idx['lm_head.weight']}", "pt") as f:
    full = f.get_tensor("lm_head.weight")
W = full[: full.shape[0] // 2].to(d, torch.bfloat16).contiguous()  # rank-0 vocab shard
del full


class Head:
    weight = W
    tp_size = 1
    shard_indices = None


class Fake:
    lm_head = Head()


fake = Fake()
mtp.Qwen4ExpMTP._b70_prepare_draft_head(fake)
h8 = fake._b70_head8
x = torch.randn(8, W.shape[1], device=d, dtype=torch.bfloat16) * 1.5
ref = F.linear(x, W).float()
ref8 = (x.float() @ (h8.w8.float() * h8.scale))
got = h8.apply(None, x).float()
print("kernel vs dequant ref: rel L2 %.2e" % ((got - ref8).norm() / ref8.norm()).item())
print("fp8 vs bf16 head:      rel L2 %.2e" % ((got - ref).norm() / ref.norm()).item())
print("top-1 agreement (random x): %.3f" % (got.argmax(-1) == ref.argmax(-1)).float().mean().item())


def gt(fn, n=50):
    g = torch.xpu.XPUGraph()
    fn()
    with torch.xpu.graph(g):
        fn()
    torch.xpu.synchronize()
    a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    torch.xpu.synchronize()
    return a.elapsed_time(b) / n * 1000


for m in (1, 4):
    xm = x[:m]
    tb, t8 = gt(lambda: F.linear(xm, W)), gt(lambda: h8.apply(None, xm))
    print(f"M={m}: bf16 head {tb:7.1f} us ({W.numel()*2/tb/1e3:5.0f} GB/s) | fp8 head {t8:7.1f} us "
          f"({h8.w8.numel()/t8/1e3:5.0f} GB/s)")
