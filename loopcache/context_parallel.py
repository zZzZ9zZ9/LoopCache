import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .batching.shared_prefix import _partial


@triton.jit
def _write_local(
    K,
    V,
    KEY,
    VALUE,
    POS,
    VALID,
    KS: tl.constexpr,
    VS: tl.constexpr,
    CS: tl.constexpr,
    PS: tl.constexpr,
    MS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pos = tl.load(POS + batch * PS[0] + row * PS[1])
    valid = tl.load(VALID + batch * MS[0] + row * MS[1])
    if valid & (pos % WORLD == RANK):
        dim = tl.arange(0, BD)
        k = tl.load(K + batch * KS[0] + head * KS[1] + row * KS[2] + dim * KS[3], dim < D, 0)
        v = tl.load(V + batch * VS[0] + head * VS[1] + row * VS[2] + dim * VS[3], dim < D, 0)
        offset = batch * CS[0] + head * CS[1] + (pos // WORLD) * CS[2] + dim * CS[3]
        tl.store(KEY + offset, k, dim < D)
        tl.store(VALUE + offset, v, dim < D)


@triton.jit
def _local_statistics(
    PARTS, PEAK, MOMENTS, D: tl.constexpr, S: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr
):
    row = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, BD)
    offset = (row * S + s) * (D + 2)
    maxima = tl.load(PARTS + offset + D, s < S, float("-inf"))
    sums = tl.load(PARTS + offset + D + 1, s < S, 0.0)
    peak = tl.max(maxima, 0)
    scales = tl.where(sums > 0, tl.exp(maxima - peak), 0.0)
    acc = tl.load(PARTS + offset[:, None] + d[None, :], (s[:, None] < S) & (d[None, :] < D), 0.0)
    tl.store(PEAK + row, peak)
    tl.store(MOMENTS + row * (D + 1) + d, tl.sum(acc * scales[:, None], 0), d < D)
    tl.store(MOMENTS + row * (D + 1) + D, tl.sum(sums * scales, 0))


class ContextCache:
    def __init__(self, capacity, world):
        self.capacity = capacity
        self.local_capacity = triton.cdiv(capacity, world)
        self.keys, self.values, self.lengths = {}, {}, {}
        self.position = 0

    def allocate(self, slot, sample):
        if slot not in self.keys:
            b, h, _, d = sample.shape
            self.keys[slot] = sample.new_empty((b, h, self.local_capacity, d))
            self.values[slot] = torch.empty_like(self.keys[slot])
        return self.keys[slot], self.values[slot]


class ContextParallel:
    def __init__(self, group=None):
        if not dist.is_initialized() or dist.get_backend(group) != "nccl":
            raise RuntimeError("Context parallelism requires a torchrun NCCL process group")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)

    def make_cache(self, capacity):
        return ContextCache(capacity, self.world)

    def _attend(
        self, q, key, value, ends, tail_key, tail_value, mask, split_size, probability_mode
    ):
        b, h, n, d = q.shape
        capacity = key.shape[2]
        splits = triton.cdiv(capacity, split_size) + 1
        local_ends = torch.div(ends - self.rank, self.world, rounding_mode="floor")
        tail = tail_key.shape[2] if self.rank == 0 else 0
        parts = torch.empty((b * n, h, splits, d + 2), device=q.device, dtype=torch.float32)
        _partial[(triton.cdiv(n, 16), b * h, splits)](
            q,
            key,
            value,
            local_ends,
            tail_key,
            tail_value,
            mask,
            parts,
            q.stride(0),
            key.stride(0),
            value.stride(0),
            tail_key.stride(0),
            tail_value.stride(0),
            mask.stride(0),
            local_ends.stride(0),
            *q.stride()[1:],
            *key.stride()[1:],
            *value.stride()[1:],
            *tail_key.stride()[1:],
            *tail_value.stride()[1:],
            *mask.stride()[1:],
            local_ends.stride(1),
            n,
            h,
            h // key.shape[1],
            d,
            tail,
            16,
            64,
            triton.next_power_of_2(d),
            splits,
            split_size,
            capacity,
            d**-0.5,
            probability_mode == "two_term",
            num_warps=4,
        )
        peaks = torch.empty((b, n, h, 1), device=q.device, dtype=torch.float32)
        moments = torch.empty((b, n, h, d + 1), device=q.device, dtype=torch.float32)
        _local_statistics[(b * n * h,)](
            parts,
            peaks,
            moments,
            d,
            splits,
            triton.next_power_of_2(d),
            triton.next_power_of_2(splits),
        )
        global_peaks = peaks.clone()
        dist.all_reduce(global_peaks, op=dist.ReduceOp.MAX, group=self.group)
        scale = torch.where(torch.isfinite(peaks), torch.exp(peaks - global_peaks), 0.0)
        moments.mul_(scale)
        dist.all_reduce(moments, group=self.group)
        out = moments[..., :d] / moments[..., d:].clamp_min(1e-30)
        return out.to(q.dtype).transpose(1, 2)

    def prefill(self, q, k, v, cache, slot, start):
        b, _, n, _ = k.shape
        limit = start + n
        if start < 0 or limit > cache.capacity:
            raise ValueError("Prefill exceeds context cache capacity")
        key, value = cache.allocate(slot, k)
        offset = (self.rank - start) % self.world
        if offset < n:
            first = (start + offset) // self.world
            count = triton.cdiv(n - offset, self.world)
            key[:, :, first : first + count].copy_(k[:, :, offset :: self.world])
            value[:, :, first : first + count].copy_(v[:, :, offset :: self.world])
        cache.lengths[slot] = limit
        local_positions = triton.cdiv(limit, self.world)
        bound = min(cache.local_capacity, max(1024, triton.next_power_of_2(local_positions)))
        ends = torch.arange(start, limit, device=q.device).expand(b, n)
        mask = torch.empty((b, n, 0), dtype=torch.bool, device=q.device)
        return self._attend(
            q,
            key[:, :, :bound],
            value[:, :, :bound],
            ends,
            k[:, :, :0],
            v[:, :, :0],
            mask,
            1024,
            "two_term",
        )

    def attention(
        self, q, k, v, cache, slot, positions, write, ends, mask, split_size, probability_mode
    ):
        key, value = cache.keys[slot], cache.values[slot]
        b, h, n, d = k.shape
        _write_local[(n, h, b)](
            k,
            v,
            key,
            value,
            positions,
            write,
            k.stride(),
            v.stride(),
            key.stride(),
            positions.stride(),
            write.stride(),
            self.rank,
            self.world,
            d,
            triton.next_power_of_2(d),
        )
        return self._attend(q, key, value, ends, k, v, mask, split_size, probability_mode)
