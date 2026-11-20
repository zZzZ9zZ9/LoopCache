from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class SelectionBuffers:
    indices: torch.Tensor
    selected_valid: torch.Tensor
    observed: torch.Tensor
    mandatory: torch.Tensor
    eligible: torch.Tensor
    motion: torch.Tensor
    row_map: torch.Tensor
    proposals: torch.Tensor
    advantage: torch.Tensor


def allocate_selection(lanes, width):
    W = lanes.numel()
    dev = lanes.device
    return SelectionBuffers(
        torch.empty(width, device=dev, dtype=torch.long),
        torch.empty(width, device=dev, dtype=torch.bool),
        torch.empty(W, device=dev, dtype=torch.bool),
        torch.empty(W, device=dev, dtype=torch.bool),
        torch.empty(W, device=dev, dtype=torch.bool),
        torch.empty(W, device=dev, dtype=torch.float32),
        torch.empty(W, device=dev, dtype=torch.long),
        torch.empty(W, device=dev, dtype=torch.long),
        torch.empty(W, device=dev, dtype=torch.float32),
    )


@triton.jit
def _motion(
    X,
    A,
    ND,
    V,
    M,
    Must,
    E,
    W: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BH)
    x = tl.load(X + row * H + h, h < H, 0).to(tl.float32)
    a = tl.load(A + row * H + h, h < H, 0).to(tl.float32)
    delta = x - a
    motion = tl.sum(delta * delta, 0) / tl.maximum(tl.sum(a * a, 0), 1e-12)
    motion = tl.where(motion == motion, tl.minimum(motion, 1e30), 0.0)
    valid = tl.load(V + row)
    depth = tl.load(ND + row)
    child = tl.load(V + row + 1, row < W - 1, False)
    eligible = valid & (depth >= D)
    must = eligible & ((depth == D) | (depth == R) | ~child)
    tl.store(M + row, motion)
    tl.store(Must + row, must)
    tl.store(E + row, eligible)


@triton.jit
def _select(
    M, Must, E, Idx, SV, Observed, Map, Stats, Batches, W: tl.constexpr, K: tl.constexpr, B: tl.constexpr
):
    i = tl.arange(0, B)
    mask = i < W
    motion = tl.load(M + i, mask, 0.0)
    must = tl.load(Must + i, mask, False)
    eligible = tl.load(E + i, mask, False)
    score = tl.where(must, float("inf"), tl.where(eligible, motion, tl.where(mask, -1.0, float("-inf"))))
    better = (score[None, :] > score[:, None]) | (score[None, :] == score[:, None]) & (
        i[None, :] < i[:, None]
    )
    rank = tl.sum(better.to(tl.int32), 1)
    selected = mask & (rank < K)
    compact = tl.cumsum(selected.to(tl.int32), 0) - 1
    observed = selected & eligible
    tl.store(Idx + compact, i.to(tl.int64), selected)
    tl.store(SV + compact, eligible, selected)
    tl.store(Observed + i, observed, mask)
    tl.store(Map + i, tl.where(selected, compact, 0).to(tl.int64), mask)
    n = tl.sum(observed.to(tl.int64), 0)
    mandatory = tl.sum(must.to(tl.int64), 0)
    optional = tl.sum((observed & ~must).to(tl.int64), 0)
    j = tl.arange(0, 8)
    old = tl.load(Stats + j, j < 7, 0)
    value = tl.where(
        j == 0,
        old + 1,
        tl.where(
            j == 1,
            old + n,
            tl.where(
                j == 2,
                old + K,
                tl.where(
                    j == 3,
                    old + mandatory,
                    tl.where(
                        j == 4,
                        old + optional,
                        tl.where(j == 5, tl.maximum(old, mandatory), old + (mandatory > K).to(tl.int64)),
                    ),
                ),
            ),
        ),
    )
    tl.store(Stats + j, value, j < 7)
    tl.store(Batches, tl.load(Batches) + (n > 0).to(tl.int64))


@triton.jit
def _finish(P, A, Map, Observed, T, OP, OA, X, Anchor, W: tl.constexpr, H: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    h = tile * BH + tl.arange(0, BH)
    observed = tl.load(Observed + row)
    slot = tl.load(Map + row)
    if tile == 0:
        proposal = tl.load(P + slot, observed, 0)
        held = tl.load(T + (row + 1) % W)
        advantage = tl.load(A + slot, observed, 0.0)
        tl.store(OP + row, tl.where(observed, proposal, held))
        tl.store(OA + row, advantage)
    hidden = tl.load(X + row * H + h, (h < H) & observed, 0)
    tl.store(Anchor + row * H + h, hidden, (h < H) & observed)


def select_gpu(
    recurrent, anchor, nextdepth, valid, *, first_depth, final_depth, buffers, stats, batches
):
    W = valid.numel()
    H = recurrent.shape[-1]
    K = buffers.indices.numel()
    _motion[W,](
        recurrent,
        anchor,
        nextdepth,
        valid,
        buffers.motion,
        buffers.mandatory,
        buffers.eligible,
        W,
        H,
        first_depth,
        final_depth,
        triton.next_power_of_2(H),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _select[1,](
        buffers.motion,
        buffers.mandatory,
        buffers.eligible,
        buffers.indices,
        buffers.selected_valid,
        buffers.observed,
        buffers.row_map,
        stats,
        batches,
        W,
        K,
        triton.next_power_of_2(W),
        num_warps=4,
    )
    return (buffers.indices, buffers.selected_valid, buffers.observed)


def finish_gpu(small_proposals, small_advantage, tokens, recurrent, anchor, buffers):
    W = tokens.numel()
    H = recurrent.shape[-1]
    _finish[W, triton.cdiv(H, 256)](
        small_proposals,
        small_advantage,
        buffers.row_map,
        buffers.observed,
        tokens,
        buffers.proposals,
        buffers.advantage,
        recurrent,
        anchor,
        W,
        H,
        256,
    )
    return (buffers.proposals, buffers.advantage)
