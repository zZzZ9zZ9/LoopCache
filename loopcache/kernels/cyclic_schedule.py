from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class CyclicScheduleBuffers:
    lanes: torch.Tensor
    spec: torch.Tensor
    kv: torch.Tensor
    injected: torch.Tensor
    plan: torch.Tensor
    invalidate_from: torch.Tensor


def allocate(engine):
    width = engine.W
    return CyclicScheduleBuffers(
        torch.empty((8, width), device=engine.tokens.device, dtype=torch.long),
        torch.empty_like(engine.spec_hidden),
        torch.empty_like(engine.kv_hidden),
        torch.empty_like(engine.injected),
        torch.empty(width + 2, device=engine.tokens.device, dtype=torch.long),
        torch.empty((), device=engine.tokens.device, dtype=torch.long),
    )


@triton.jit
def _metadata(
    Chosen,
    Adv,
    Observed,
    KP,
    SD,
    Valid,
    KVValid,
    SpecValid,
    KD,
    SDue,
    New,
    Tokens,
    Merged,
    Candidate,
    Streak,
    Count,
    Output,
    Finished,
    Stats,
    EOS,
    Next,
    Plan,
    Invalidate,
    PHASE,
    R,
    PROMPT,
    CAPACITY,
    BUDGET,
    MARGIN,
    PENALTY,
    W: tl.constexpr,
    B: tl.constexpr,
    NEOS: tl.constexpr,
    BEOS: tl.constexpr,
):
    i = tl.arange(0, B)
    mask = i < W
    count = tl.load(Count)
    old_finished = tl.load(Finished)
    valid = tl.load(Valid + i, mask, False)
    kv = tl.load(KVValid + i, mask, False)
    spec = tl.load(SpecValid + i, mask, False)
    kd = tl.load(KD + i, mask, False)
    sdue = tl.load(SDue + i, mask, False)
    new = tl.load(New + i, mask, False)
    merged = tl.load(Merged + i, mask, False)
    sd = tl.load(SD + i, mask, 0)
    tokens = tl.load(Tokens + i, mask, 0)
    child = tl.load(Tokens + (i + 1) % W, mask, 0)
    childvalid = tl.load(Valid + (i + 1) % W, mask, False) & (i < W - 1)
    chosen = tl.load(Chosen + i, mask, 0)
    adv = tl.load(Adv + i, mask, 0.0)
    observed = tl.load(Observed + i, mask, False)
    candidate = tl.load(Candidate + i, mask, -1)
    streak = tl.load(Streak + i, mask, 0)
    streak = tl.where(observed, tl.where(candidate == chosen, streak + 1, 1), streak)
    work = kv.to(tl.int64) * (PHASE + 1) + tl.where(merged, 0, sd * valid.to(tl.int64))
    spent = tl.sum(work, 0) - tl.cumsum(work, 0)
    permit = adv >= MARGIN + PENALTY * spent.to(tl.float32)
    kp = tl.load(KP + i, mask, 0)
    final = kv & (PHASE + 1 == R)
    chosen = tl.where(final, kp, chosen)
    event = valid & observed & (~childvalid | (chosen != child) & (permit | final))
    event = event & (i < W - 1) & (PROMPT + count + i < CAPACITY)
    cut = tl.min(tl.where(event, i, W), 0)
    tl.store(Invalidate, tl.where(cut < W, PROMPT + count + cut, CAPACITY))
    create = (i == cut + 1) & (cut < W)
    keep = valid & (i <= cut)
    pick = tl.sum(tl.where(i == tl.minimum(cut, W - 1), chosen, 0), 0)
    nt = tl.where(create, pick, tokens)
    nv = keep | create
    nf = create
    nsd = tl.where(create, 0, sd)
    nkv = kv & keep
    nm = merged & keep
    nc = tl.where(create, -1, tl.where(observed, chosen, candidate))
    nstreak = tl.where(create, 0, streak)
    commits = tl.full((), 0, tl.int64)
    root = tl.full((B,), False, tl.int1)
    source = i
    if PHASE + 1 == R:
        mismatch = final & (~childvalid | (child != kp))
        first_bad = tl.min(tl.where(mismatch, i, W), 0)
        commits = tl.minimum(tl.sum(final.to(tl.int64), 0), first_bad + 1)
        commits = tl.minimum(commits, BUDGET - count)
        eos_commit = tl.full((), False, tl.int1)
        if NEOS:
            ei = tl.arange(0, BEOS)
            eos = tl.load(EOS + ei, ei < NEOS, -1)
            is_eos = tl.sum(((kp[:, None] == eos[None, :]) & (ei[None, :] < NEOS)).to(tl.int32), 1) > 0
            first_eos = tl.min(tl.where((i < commits) & is_eos, i, W), 0)
            commits = tl.minimum(commits, first_eos + 1)
            eos_commit = first_eos < W
        write = mask & (i < commits)
        tl.store(Output + count + i, kp, write)
        if commits < W:
            tl.store(Output + BUDGET, 0)
        source = tl.minimum(i + commits, W - 1).to(tl.int32)
        inrange = i + commits < W
        nt = tl.gather(nt, source, 0)
        nv = tl.gather(nv, source, 0) & inrange
        nf = tl.gather(nf, source, 0) & inrange
        nsd = tl.gather(nsd, source, 0)
        nkv = tl.full((B,), False, tl.int1)
        nm = tl.gather(nm, source, 0) & inrange
        nc = tl.gather(nc, source, 0)
        nstreak = tl.gather(nstreak, source, 0)
        root_valid = tl.sum(((i == 0) & nv).to(tl.int32), 0) > 0
        need_root = (commits > 0) & ~root_valid
        root = (i == 0) & need_root
        bonus_ix = tl.minimum(tl.maximum(commits - 1, 0), W - 1)
        bonus = tl.sum(tl.where(i == bonus_ix, kp, 0), 0)
        nt = tl.where(root, bonus, nt)
        nv = nv | root
        nf = nf | root
        nsd = tl.where(root, 0, nsd)
        nm = nm & ~root
        nc = tl.where(root, -1, nc)
        nstreak = tl.where(root, 0, nstreak)
        tl.store(Count, count + commits)
        tl.store(Finished, old_finished | eos_commit | (count + commits >= BUDGET))
    tl.store(Next + 0 * W + i, nt, mask)
    tl.store(Next + 1 * W + i, nv.to(tl.int64), mask)
    tl.store(Next + 2 * W + i, nf.to(tl.int64), mask)
    tl.store(Next + 3 * W + i, nsd, mask)
    tl.store(Next + 4 * W + i, nkv.to(tl.int64), mask)
    tl.store(Next + 5 * W + i, nm.to(tl.int64), mask)
    tl.store(Next + 6 * W + i, nc, mask)
    tl.store(Next + 7 * W + i, nstreak, mask)
    reset = tl.gather(create, source.to(tl.int32), 0) | root
    tl.store(Plan + i, source + reset.to(tl.int64) * W, mask)
    tl.store(Plan + W, cut)
    tl.store(Plan + W + 1, commits)
    checked = tl.sum((final & childvalid & (i < commits)).to(tl.int64), 0)
    accepted = tl.sum((final & childvalid & (i < commits) & (child == kp)).to(tl.int64), 0)
    selected = mask & (i == tl.minimum(cut, W - 1))
    switched = (
        (cut < W)
        & (tl.sum((selected & childvalid).to(tl.int32), 0) > 0)
        & ~(tl.sum((selected & final).to(tl.int32), 0) > 0)
    )
    live = tl.sum(valid.to(tl.int64), 0)
    kvn = tl.sum(kv.to(tl.int64), 0)
    sn = tl.sum(spec.to(tl.int64), 0)
    merged_count = tl.sum((valid & merged).to(tl.int64), 0)
    reads = tl.sum(kd.to(tl.int64), 0) + tl.sum(sdue.to(tl.int64), 0)
    discarded = tl.sum((valid & (i > cut)).to(tl.int64), 0)
    creates = tl.sum(new.to(tl.int64), 0)
    cut_spent = tl.sum(tl.where((cut < W) & selected, spent, 0), 0)
    si = tl.arange(0, 16)
    delta = tl.where(
        si == 0,
        (live > 0).to(tl.int64),
        tl.where(
            si == 1,
            kvn + sn,
            tl.where(
                si == 2,
                kvn,
                tl.where(
                    si == 3,
                    sn,
                    tl.where(
                        si == 4,
                        merged_count,
                        tl.where(
                            si == 5,
                            reads,
                            tl.where(
                                si == 6,
                                switched.to(tl.int64),
                                tl.where(
                                    si == 7,
                                    discarded,
                                    tl.where(
                                        si == 8,
                                        checked,
                                        tl.where(
                                            si == 9,
                                            accepted,
                                            tl.where(
                                                si == 10,
                                                commits,
                                                tl.where(
                                                    si == 11,
                                                    creates,
                                                    tl.where(
                                                        si == 12,
                                                        ((live > 0) & (PHASE == 0)).to(tl.int64),
                                                        tl.where(si == 15, cut_spent, 0),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    tl.store(Stats + si, tl.load(Stats + si) + delta)


@triton.jit
def _reorder(
    Recurrent,
    Injected,
    Plan,
    SpecOut,
    KVOut,
    InjectedOut,
    W: tl.constexpr,
    H: tl.constexpr,
    RN: tl.constexpr,
    RD: tl.constexpr,
    IN: tl.constexpr,
    ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    code = tl.load(Plan + row)
    source = code % W
    zero = code >= W
    s = tl.load(Recurrent + (source + W) * RN + h * RD, h < H, 0)
    k = tl.load(Recurrent + source * RN + h * RD, h < H, 0)
    inj = tl.load(Injected + source * IN + h * ID, h < H, 0)
    tl.store(SpecOut + row * H + h, tl.where(zero, 0, s), h < H)
    tl.store(KVOut + row * H + h, tl.where(zero, 0, k), h < H)
    tl.store(InjectedOut + row * H + h, tl.where(zero, 0, inj), h < H)


@triton.jit
def _commit(
    Next,
    SpecIn,
    KVIn,
    InjectedIn,
    Tokens,
    Valid,
    Fresh,
    Depth,
    KVActive,
    Merged,
    Candidates,
    Streaks,
    Spec,
    KV,
    Injected,
    W: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hv = ix < W * H
    mv = ix < W
    tl.store(Spec + ix, tl.load(SpecIn + ix, hv, 0), hv)
    tl.store(KV + ix, tl.load(KVIn + ix, hv, 0), hv)
    tl.store(Injected + ix, tl.load(InjectedIn + ix, hv, 0), hv)
    tl.store(Tokens + ix, tl.load(Next + 0 * W + ix, mv, 0), mv)
    tl.store(Valid + ix, tl.load(Next + 1 * W + ix, mv, 0), mv)
    tl.store(Fresh + ix, tl.load(Next + 2 * W + ix, mv, 0), mv)
    tl.store(Depth + ix, tl.load(Next + 3 * W + ix, mv, 0), mv)
    tl.store(KVActive + ix, tl.load(Next + 4 * W + ix, mv, 0), mv)
    tl.store(Merged + ix, tl.load(Next + 5 * W + ix, mv, 0), mv)
    tl.store(Candidates + ix, tl.load(Next + 6 * W + ix, mv, 0), mv)
    tl.store(Streaks + ix, tl.load(Next + 7 * W + ix, mv, 0), mv)


def transition(
    e, *, chosen, adv, observed, kp, sd, valid, kvvalid, specvalid, kd, sdue, new, recurrent, phase
):
    out = e.cyclic_schedule_buffers
    W = e.W
    H = e.spec_hidden.shape[-1]
    _metadata[1,](
        chosen,
        adv,
        observed,
        kp,
        sd,
        valid,
        kvvalid,
        specvalid,
        kd,
        sdue,
        new,
        e.tokens,
        e.merged,
        e.candidates,
        e.streaks,
        e.count,
        e.output,
        e.finished,
        e.stats,
        e.eos,
        out.lanes,
        out.plan,
        out.invalidate_from,
        phase,
        e.R,
        e.prompt_length,
        e.capacity,
        e.budget,
        e.margin,
        0.0 if e.policy == "confidence" else e.work_penalty,
        W,
        triton.next_power_of_2(W),
        e.eos.numel(),
        triton.next_power_of_2(max(1, e.eos.numel())),
        num_warps=4,
        enable_fp_fusion=False,
    )
    if e.draft_pool is not None:
        e.draft_pool.invalidate_from(out.invalidate_from)
    _reorder[W, triton.cdiv(H, 256)](
        recurrent,
        e.injected,
        out.plan,
        out.spec,
        out.kv,
        out.injected,
        W,
        H,
        *recurrent.stride()[1:],
        *e.injected.stride()[1:],
        256,
        num_warps=4,
    )
    _commit[triton.cdiv(W * H, 256),](
        out.lanes,
        out.spec,
        out.kv,
        out.injected,
        e.tokens,
        e.valid,
        e.fresh,
        e.spec_depth,
        e.kv_active,
        e.merged,
        e.candidates,
        e.streaks,
        e.spec_hidden,
        e.kv_hidden,
        e.injected,
        W,
        H,
        256,
        num_warps=4,
    )
