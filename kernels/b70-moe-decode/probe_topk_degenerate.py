"""What does _moe_C.topk_softmax write for degenerate rows? (one B70)"""
import torch
import vllm._custom_ops as ops
d = torch.device("xpu")
E, K = 512, 10
lg = torch.randn(6, E, device=d).bfloat16()
lg[0] = float("nan"); lg[1] = float("-inf"); lg[2, 7] = float("nan"); lg[3, 100] = float("inf")
lg[4] = 0.0; lg[5, ::2] = float("-inf")
for renorm in (True, False):
    tw = torch.empty(6, K, dtype=torch.float32, device=d); ti = torch.empty(6, K, dtype=torch.int32, device=d)
    ops.topk_softmax(tw, ti, torch.empty_like(ti), lg, renorm)
    torch.xpu.synchronize()
    for r, name in enumerate(["all nan", "all -inf", "one nan", "one +inf", "all zero", "half -inf"]):
        print(f"renorm={renorm} {name:9s}: ids {ti[r, :5].tolist()} w {[round(x, 4) for x in tw[r, :5].tolist()]}")
