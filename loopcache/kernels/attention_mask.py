import torch
import triton
import triton.language as tl


@triton.jit
def _build(P, R, W, B, A, E, T, N: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    good = ix < N * N
    i = ix // N
    j = ix % N
    pi = tl.load(P + i, good, other=0)
    pj = tl.load(P + j, good, other=0)
    ri = tl.load(R + i, good, other=False)
    rj = tl.load(R + j, good, other=False)
    wi = tl.load(W + i, good, other=False)
    aj = tl.load(A + j, good, other=False)
    b = tl.load(B)
    end = tl.where(ri, tl.where(wi, pi, tl.minimum(pi - 1, b - 1)), -1)
    tail = ri & ~wi & ((pj < pi) & (pj > end) & aj | (i == j) & rj)
    tl.store(E + i, end, good & (j == 0))
    tl.store(T + ix, tail, good)


class Builder:
    def __init__(self):
        self.buffers = {}

    def __call__(self, positions, read_valid, write_valid, base_end, primary):
        if positions.ndim != 1 or positions.dtype != torch.int64 or (not positions.is_cuda):
            raise ValueError("positions must be a CUDA int64 vector")
        n = positions.numel()
        if not 1 <= n <= 512:
            raise ValueError("metadata row count must be 1..512")
        if any((x.shape != (n,) or x.dtype != torch.bool for x in (read_valid, write_valid, primary))):
            raise ValueError("read/write/primary must be matching bool vectors")
        if base_end.ndim != 0 or base_end.dtype != torch.int64:
            raise ValueError("base_end must be an int64 scalar tensor")
        if any(
            (
                x.device != positions.device or not x.is_contiguous()
                for x in (positions, read_valid, write_valid, base_end, primary)
            )
        ):
            raise ValueError("all metadata must be contiguous on the same CUDA device")
        key = (positions.device, n)
        if key not in self.buffers:
            self.buffers[key] = (
                torch.empty_like(positions),
                torch.empty((n, n), device=positions.device, dtype=torch.bool),
            )
        ends, tail = self.buffers[key]
        _build[triton.cdiv(n * n, 256),](
            positions, read_valid, write_valid, base_end, primary, ends, tail, n, 256, num_warps=4
        )
        return (ends.view(n), tail.view(n, n))
