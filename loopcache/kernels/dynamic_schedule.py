from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class ScheduleBuffers:
    tokens: torch.Tensor
    depths: torch.Tensor
    valid: torch.Tensor
    candidates: torch.Tensor
    streaks: torch.Tensor
    hidden: torch.Tensor
    injected: torch.Tensor
    anchor: torch.Tensor | None
    plan: torch.Tensor


def allocate_schedule_buffers(tokens, hidden, *, with_anchor=False):
    return ScheduleBuffers(
        torch.empty_like(tokens),
        torch.empty_like(tokens),
        torch.empty_like(tokens, dtype=torch.bool),
        torch.empty_like(tokens),
        torch.empty_like(tokens),
        torch.empty_like(hidden),
        torch.empty_like(hidden),
        torch.empty_like(hidden) if with_anchor else None,
        torch.empty(tokens.numel() + 2, device=tokens.device, dtype=torch.long),
    )


@triton.jit
def _metadata(
    P,
    A,
    ND,
    V,
    T,
    C,
    S,
    Count,
    Output,
    Finished,
    Stats,
    Observed,
    EOS,
    OT,
    OD,
    OV,
    OC,
    OS,
    Plan,
    W: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BUDGET: tl.constexpr,
    PROMPT: tl.constexpr,
    CAPACITY: tl.constexpr,
    POLICY: tl.constexpr,
    MARGIN: tl.constexpr,
    PATIENCE: tl.constexpr,
    PENALTY: tl.constexpr,
    FIXED: tl.constexpr,
    NEOS: tl.constexpr,
    BEOS: tl.constexpr,
    OBSERVED_VECTOR: tl.constexpr,
):
    i = tl.arange(0, B)
    mask = i < W
    count = tl.load(Count)
    oldfinished = tl.load(Finished)
    if OBSERVED_VECTOR:
        observed = tl.load(Observed + i, mask, False)
    else:
        observed = tl.load(Observed)
    valid = tl.load(V + i, mask, False)
    depth = tl.load(ND + i, mask, 0)
    tokens = tl.load(T + i, mask, 0)
    child = tl.load(T + (i + 1) % W, mask, 0)
    childvalid = tl.load(V + (i + 1) % W, mask, False) & (i < W - 1)
    proposal = tl.load(P + i, mask, 0)
    if not FIXED:
        proposal = tl.where(observed, proposal, child)
    candidate = tl.load(C + i, mask, -1)
    oldstreak = tl.load(S + i, mask, 0)
    streak = tl.where(observed, tl.where(candidate == proposal, oldstreak + 1, 1), oldstreak)
    advantage = tl.load(A + i, mask, 0.0)
    work = tl.where(valid, depth, 0)
    spent = tl.sum(work, 0) - tl.cumsum(work, 0)
    if POLICY == 0:
        permit = tl.full((B,), True, tl.int1)
    elif POLICY == 1:
        permit = advantage >= MARGIN
    elif POLICY == 2:
        permit = (streak >= PATIENCE) & (advantage >= MARGIN)
    else:
        permit = advantage >= MARGIN + PENALTY * spent.to(tl.float32)
    if FIXED:
        due = valid & ((depth == D) | (depth == R))
    else:
        due = valid & (depth >= D)
    final = tl.load(V) & (tl.load(ND) == R)
    different = child != proposal
    if FIXED:
        event = due & (~childvalid | different & (depth == R))
    else:
        event = due & (~childvalid | different & (permit | (depth == R)))
    event = event & (i < W - 1) & (PROMPT + count + i < CAPACITY)
    cut = tl.min(tl.where(event, i, W), 0)
    pick = tl.sum(tl.where(i == tl.minimum(cut, W - 1), proposal, 0), 0)
    create = (i == cut + 1) & (cut < W)
    newvalid = valid & (i <= cut) | create
    newtokens = tl.where(create, pick, tokens)
    newdepth = tl.where(create, 0, depth)
    source = (i + final.to(tl.int32)) % W
    tl.store(OT + i, tl.gather(newtokens, source, 0), mask)
    tl.store(OD + i, tl.gather(newdepth, source, 0), mask)
    tl.store(OV + i, tl.gather(newvalid, source, 0) & (~final | (i < W - 1)), mask)
    if FIXED:
        tl.store(OC + i, candidate, mask)
        tl.store(OS + i, oldstreak, mask)
    else:
        newcandidate = tl.where(create, -1, tl.where(observed, proposal, candidate))
        newstreak = tl.where(create, 0, streak)
        tl.store(OC + i, tl.gather(newcandidate, source, 0), mask)
        tl.store(OS + i, tl.gather(newstreak, source, 0), mask)
    reset = (source == cut + 1) & (cut < W)
    tl.store(Plan + i, source + reset.to(tl.int32) * W, mask)
    tl.store(Plan + W, cut)
    tl.store(Plan + W + 1, final.to(tl.int64))
    first = tl.sum(tl.where(i == 0, proposal, 0), 0)
    tl.store(Output + tl.where(final, count, BUDGET), first)
    newcount = count + final.to(tl.int64)
    tl.store(Count, newcount)
    ei = tl.arange(0, BEOS)
    eos = tl.load(EOS + ei, ei < NEOS, -1)
    committed_eos = tl.sum(((eos == first) & (ei < NEOS)).to(tl.int32), 0) > 0
    tl.store(Finished, oldfinished | final & committed_eos | (newcount >= BUDGET))
    cut_child = tl.sum(((i == tl.minimum(cut, W - 1)) & childvalid).to(tl.int32), 0) > 0
    switch = (cut < W) & cut_child & ~(final & (cut == 0))
    if FIXED:
        switch = tl.full((), False, tl.int1)
    checked = final & tl.load(V + 1) if W > 1 else tl.full((), False, tl.int1)
    accepted = checked & (tl.load(T + 1) == first) if W > 1 else tl.full((), False, tl.int1)
    live = tl.sum(valid.to(tl.int64), 0)
    reads = tl.sum(due.to(tl.int64), 0)
    if not FIXED:
        reads = tl.sum((due & observed).to(tl.int64), 0)
    discarded = tl.sum((valid & (i > cut)).to(tl.int64), 0)
    si = tl.arange(0, 8)
    delta = tl.where(
        si == 0,
        (live > 0).to(tl.int64),
        tl.where(
            si == 1,
            live,
            tl.where(
                si == 2,
                reads,
                tl.where(
                    si == 3,
                    switch.to(tl.int64),
                    tl.where(
                        si == 4,
                        checked.to(tl.int64),
                        tl.where(si == 5, accepted.to(tl.int64), discarded),
                    ),
                ),
            ),
        ),
    )
    tl.store(Stats + si, tl.load(Stats + si, si < 7, 0) + delta, si < 7)


@triton.jit
def _reorder_hidden(
    X,
    Injected,
    Anchor,
    OX,
    OI,
    OA,
    Plan,
    W: tl.constexpr,
    H: tl.constexpr,
    HAS_ANCHOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    plan = tl.load(Plan + row)
    source = plan % W
    reset = plan >= W
    idx = source * H + h
    outidx = row * H + h
    x = tl.load(X + idx, h < H, 0)
    injected = tl.load(Injected + idx, h < H, 0)
    tl.store(OX + outidx, tl.where(reset, 0, x), h < H)
    tl.store(OI + outidx, tl.where(reset, 0, injected), h < H)
    if HAS_ANCHOR:
        anchor = tl.load(Anchor + idx, h < H, 0)
        tl.store(OA + outidx, tl.where(reset, 0, anchor), h < H)


@triton.jit
def _commit(
    IT,
    ID,
    IV,
    IC,
    IS,
    IX,
    II,
    IA,
    OT,
    OD,
    OV,
    OC,
    OS,
    OX,
    OI,
    OA,
    W: tl.constexpr,
    H: tl.constexpr,
    HAS_ANCHOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hv = idx < W * H
    tl.store(OX + idx, tl.load(IX + idx, hv, 0), hv)
    tl.store(OI + idx, tl.load(II + idx, hv, 0), hv)
    if HAS_ANCHOR:
        tl.store(OA + idx, tl.load(IA + idx, hv, 0), hv)
    mv = idx < W
    tl.store(OT + idx, tl.load(IT + idx, mv, 0), mv)
    tl.store(OD + idx, tl.load(ID + idx, mv, 0), mv)
    tl.store(OV + idx, tl.load(IV + idx, mv, 0), mv)
    tl.store(OC + idx, tl.load(IC + idx, mv, 0), mv)
    tl.store(OS + idx, tl.load(IS + idx, mv, 0), mv)


def fused_schedule(
    *,
    proposals,
    advantage,
    nextdepth,
    valid,
    tokens,
    candidates,
    streaks,
    recurrent,
    injected,
    count,
    output,
    finished,
    stats,
    observed,
    eos,
    out,
    budget,
    prompt_length,
    capacity,
    draft_depth,
    total_depth,
    policy="always",
    margin=0.0,
    patience=2,
    work_penalty=0.0,
    last_readout_hidden=None,
    fixed_depth=False,
    step_size=1,
):
    policies = {"always": 0, "margin": 1, "patience": 2, "cost": 3}
    if policy not in policies:
        raise ValueError(policy)
    if step_size < 1:
        raise ValueError("step_size must be positive")
    if fixed_depth and (step_size != draft_depth or total_depth % draft_depth):
        raise ValueError("fixed-depth scheduler requires step_size=d and R%d==0")
    W = tokens.numel()
    H = recurrent.shape[-1]
    if observed.numel() not in (1, W):
        raise ValueError("observed must be scalar or [W]")
    if recurrent.shape != (1, W, H) or not recurrent.is_contiguous() or (not injected.is_contiguous()):
        raise ValueError("expected contiguous [1,W,H] hidden state")
    has_anchor = last_readout_hidden is not None
    if has_anchor != (out.anchor is not None):
        raise ValueError("anchor allocation mismatch")
    _metadata[1,](
        proposals,
        advantage,
        nextdepth,
        valid,
        tokens,
        candidates,
        streaks,
        count,
        output,
        finished,
        stats,
        observed,
        eos,
        out.tokens,
        out.depths,
        out.valid,
        out.candidates,
        out.streaks,
        out.plan,
        W,
        triton.next_power_of_2(W),
        draft_depth,
        total_depth,
        budget,
        prompt_length,
        capacity,
        policies[policy],
        margin,
        patience,
        work_penalty,
        fixed_depth,
        eos.numel(),
        triton.next_power_of_2(max(1, eos.numel())),
        observed.numel() != 1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _reorder_hidden[W, triton.cdiv(H, 256)](
        recurrent,
        injected,
        last_readout_hidden if has_anchor else recurrent,
        out.hidden,
        out.injected,
        out.anchor if has_anchor else out.hidden,
        out.plan,
        W,
        H,
        has_anchor,
        256,
    )
    return out


def commit_schedule(
    out, *, tokens, depths, valid, candidates, streaks, hidden, injected, last_readout_hidden=None
):
    W = tokens.numel()
    H = hidden.shape[-1]
    has_anchor = last_readout_hidden is not None
    if has_anchor != (out.anchor is not None):
        raise ValueError("anchor allocation mismatch")
    _commit[triton.cdiv(W * H, 256),](
        out.tokens,
        out.depths,
        out.valid,
        out.candidates,
        out.streaks,
        out.hidden,
        out.injected,
        out.anchor if has_anchor else out.hidden,
        tokens,
        depths,
        valid,
        candidates,
        streaks,
        hidden,
        injected,
        last_readout_hidden if has_anchor else hidden,
        W,
        H,
        has_anchor,
        256,
    )
