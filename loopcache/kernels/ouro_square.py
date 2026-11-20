import torch
import triton
import triton.language as tl


@triton.jit
def _cast_square(X, Output, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    square = x * x
    tl.store(Output + i, square, i < N)


def square_fp32(x):
    if not x.is_contiguous():
        return x.float().square()
    out = torch.empty_like(x, dtype=torch.float32)
    _cast_square[triton.cdiv(x.numel(), 256),](
        x, out, x.numel(), 256, num_warps=4, enable_fp_fusion=False
    )
    return out
