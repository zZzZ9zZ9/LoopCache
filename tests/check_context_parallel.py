import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from loopcache.context_parallel import ContextParallel


def reference(query, keys, values):
    return F.scaled_dot_product_attention(
        query.float(), keys.float(), values.float(), enable_gqa=True
    )


def assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.001)


def check(device):
    torch.manual_seed(7)
    batch, heads, kv_heads, dim, length = 2, 4, 2, 64, 8195
    cp = ContextParallel()
    if cp.world < 2:
        raise ValueError("Use torchrun with at least two CUDA devices")
    cache = cp.make_cache(length + 32)

    def sample(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16)

    q = sample(batch, heads, length, dim)
    k = sample(batch, kv_heads, length, dim)
    v = sample(batch, kv_heads, length, dim)
    for start, end in [(0, 13), (13, 4093), (4093, length)]:
        block = cp.prefill(
            q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], cache, 0, start
        )
        for pos in [start, end - 1]:
            target = reference(q[:, :, pos : pos + 1], k[:, :, : pos + 1], v[:, :, : pos + 1])
            assert_close(block[:, :, pos - start : pos - start + 1], target)

    for shard, source in [(cache.keys[0], k), (cache.values[0], v)]:
        local = source[:, :, cp.rank :: cp.world]
        assert shard.shape[2] == (cache.capacity + cp.world - 1) // cp.world
        torch.testing.assert_close(shard[:, :, : local.shape[2]], local, rtol=0, atol=0)

    q = sample(batch, heads, 5, dim)
    tail_k = sample(batch, kv_heads, 5, dim)
    tail_v = sample(batch, kv_heads, 5, dim)
    offsets = torch.tensor([length, length - 2], device=device)[:, None]
    positions = offsets + torch.tensor([0, 1, 1, 2, 3], device=device)
    write = torch.tensor([True, True, False, False, False], device=device).expand(batch, -1)
    ends = offsets + torch.tensor([0, 1, 0, 0, 0], device=device)
    ends[:, -1] = -1
    mask = torch.zeros(batch, 5, 5, device=device, dtype=torch.bool)
    mask[:, 2, 2] = True
    mask[:, 3, 2:4] = True
    args = (q, tail_k, tail_v, cache, 0, positions, write, ends, mask, 1024, "two_term")
    cp.attention(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = cp.attention(*args)
    for _ in range(2):
        tail_v.normal_()
        graph.replay()
        for row, prefix in enumerate((length, length - 2)):
            saved_k = torch.cat([k[row, :, :prefix], tail_k[row, :, :2]], dim=1)
            saved_v = torch.cat([v[row, :, :prefix], tail_v[row, :, :2]], dim=1)
            for lane in range(4):
                end = ends[row, lane].item() + 1
                keys = torch.cat([saved_k[:, :end], tail_k[row, :, mask[row, lane]]], dim=1)
                values = torch.cat([saved_v[:, :end], tail_v[row, :, mask[row, lane]]], dim=1)
                target = reference(q[row : row + 1, :, lane : lane + 1], keys[None], values[None])
                assert_close(result[row : row + 1, :, lane : lane + 1], target)
        assert not result[:, :, -1].count_nonzero().item()


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        check(device)
        if dist.get_rank() == 0:
            print(f"Context parallel checks passed on {dist.get_world_size()} ranks.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
