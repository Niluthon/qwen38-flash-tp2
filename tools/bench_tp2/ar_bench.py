import os, time, torch, torch.distributed as dist
dist.init_process_group("xccl"); r = dist.get_rank(); torch.xpu.set_device(r); dev = torch.device("xpu", r)
for numel in [2560, 4 * 2560, 16 * 2560, 4096 * 2560]:
    x = torch.ones(numel, dtype=torch.bfloat16, device=dev)
    for _ in range(20): dist.all_reduce(x)
    torch.xpu.synchronize(); n = 200 if numel < 1e6 else 20
    t = time.perf_counter()
    for _ in range(n): dist.all_reduce(x)
    torch.xpu.synchronize(); us = (time.perf_counter() - t) / n * 1e6
    if r == 0: print(f"all_reduce {numel*2/1024:9.0f} KiB: {us:8.1f} us", flush=True)
dist.destroy_process_group()
