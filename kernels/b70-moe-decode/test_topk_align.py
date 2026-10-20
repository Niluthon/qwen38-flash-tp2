"""Does b70moe.topk_softmax fault on a misaligned logits view? (one B70)   usage: test_topk_align.py <lib.so> <offset>"""
import sys, torch
torch.ops.load_library(sys.argv[1])
d = torch.device("xpu")
off = int(sys.argv[2])
base = torch.randn(4 * 512 + 64, device=d).bfloat16()
lg = base[off: off + 4 * 512].view(4, 512)
print("data_ptr % 16 =", lg.data_ptr() % 16, flush=True)
tw = torch.empty(4, 10, dtype=torch.float32, device=d); ti = torch.empty(4, 10, dtype=torch.int32, device=d)
torch.ops.b70moe.topk_softmax(tw, ti, lg, True, None)
torch.xpu.synchronize()
ref = torch.topk(torch.softmax(lg.float(), -1), 10, -1)
print("ok, ids match torch.topk:", bool((ref.indices.int() == ti).all()), flush=True)
