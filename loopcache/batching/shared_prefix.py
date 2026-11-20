import torch
import triton
import triton.language as tl
from ..kernels.indexed_attention import _merge


@triton.jit
def _partial(
    Q,
    K,
    V,
    ENDS,
    TK,
    TV,
    MASK,
    OUT,
    QB: tl.constexpr,
    KB: tl.constexpr,
    VB: tl.constexpr,
    TB: tl.constexpr,
    UB: tl.constexpr,
    MB: tl.constexpr,
    EB: tl.constexpr,
    QH: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    TH: tl.constexpr,
    TT: tl.constexpr,
    TD: tl.constexpr,
    UH: tl.constexpr,
    UT: tl.constexpr,
    UD: tl.constexpr,
    MN: tl.constexpr,
    MT: tl.constexpr,
    EN: tl.constexpr,
    N: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,
    L: tl.constexpr,
    SCALE: tl.constexpr,
    TWOTERM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    h = bh % HQ
    b = bh // HQ
    sp = tl.program_id(2)
    kh = h // G
    Q += b * QB
    K += b * KB
    V += b * VB
    TK += b * TB
    TV += b * UB
    MASK += b * MB
    ENDS += b * EB
    ds = tl.arange(0, BD)
    ends = tl.load(ENDS + rows * EN, rows < N, other=-1)
    q = tl.load(
        Q + h * QH + rows[:, None] * QN + ds[None, :] * QD,
        (rows[:, None] < N) & (ds[None, :] < D),
        other=0,
    )
    m = tl.full((BM,), float("-inf"), tl.float32)
    den = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, BD), 0.0, tl.float32)
    if sp < S - 1:
        begin = sp * CHUNK
        end = tl.minimum(tl.minimum((sp + 1) * CHUNK, tl.max(ends, axis=0) + 1), L)
        for start in range(begin, end, BN):
            ts = start + tl.arange(0, BN)
            k = tl.load(
                K + kh * KH + ds[:, None] * KD + ts[None, :] * KT,
                (ds[:, None] < D) & (ts[None, :] < end),
                other=0,
            )
            score = tl.dot(q, k).to(tl.float32) * SCALE
            allow = (
                (rows[:, None] < N)
                & (ts[None, :] < end)
                & (ts[None, :] <= ends[:, None])
            )
            score = tl.where(allow, score, float("-inf"))
            new_m = tl.maximum(m, tl.max(score, axis=1))
            safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
            prob = tl.exp(score - safe_m[:, None])
            alpha = tl.exp(m - safe_m)
            v = tl.load(
                V + kh * VH + ts[:, None] * VT + ds[None, :] * VD,
                (ts[:, None] < end) & (ds[None, :] < D),
                other=0,
            )
            hi = prob.to(q.dtype)
            acc = tl.dot(hi, v, acc * alpha[:, None])
            if TWOTERM:
                lo = (prob - hi.to(tl.float32)).to(q.dtype)
                acc = tl.dot(lo, v, acc)
            den = den * alpha + tl.sum(prob, axis=1)
            m = new_m
    else:
        for tail_start in range(0, T, BN):
            ts = tail_start + tl.arange(0, BN)
            k = tl.load(
                TK + kh * TH + ds[:, None] * TD + ts[None, :] * TT,
                (ds[:, None] < D) & (ts[None, :] < T),
                other=0,
            )
            allowed = tl.load(
                MASK + rows[:, None] * MN + ts[None, :] * MT,
                (rows[:, None] < N) & (ts[None, :] < T),
                other=False,
            )
            score = tl.dot(q, k).to(tl.float32) * SCALE
            score = tl.where(allowed, score, float("-inf"))
            new_m = tl.maximum(m, tl.max(score, axis=1))
            safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
            prob = tl.exp(score - safe_m[:, None])
            alpha = tl.exp(m - safe_m)
            v = tl.load(
                TV + kh * UH + ts[:, None] * UT + ds[None, :] * UD,
                (ts[:, None] < T) & (ds[None, :] < D),
                other=0,
            )
            hi = prob.to(q.dtype)
            acc = tl.dot(hi, v, acc * alpha[:, None])
            if TWOTERM:
                lo = (prob - hi.to(tl.float32)).to(q.dtype)
                acc = tl.dot(lo, v, acc)
            den = den * alpha + tl.sum(prob, axis=1)
            m = new_m
    base = (((b * N + rows) * HQ + h) * S + sp) * (D + 2)
    tl.store(
        OUT + base[:, None] + ds[None, :], acc, (rows[:, None] < N) & (ds[None, :] < D)
    )
    tl.store(OUT + base + D, m, rows < N)
    tl.store(OUT + base + D + 1, den, rows < N)


def shared_prefix_batch(
    q,
    key,
    value,
    ends,
    tail_key,
    tail_value,
    mask,
    split_size=1024,
    probability_mode="single",
):
    b, h, n, d = q.shape
    t = tail_key.shape[2]
    capacity = key.shape[2]
    splits = triton.cdiv(capacity, split_size) + 1
    tmp = torch.empty((b * n, h, splits, d + 2), device=q.device, dtype=torch.float32)
    out = torch.empty((1, h, b * n, d), device=q.device, dtype=q.dtype)
    _partial[triton.cdiv(n, 16), b * h, splits](
        q,
        key,
        value,
        ends,
        tail_key,
        tail_value,
        mask,
        tmp,
        q.stride(0),
        key.stride(0),
        value.stride(0),
        tail_key.stride(0),
        tail_value.stride(0),
        mask.stride(0),
        ends.stride(0),
        *q.stride()[1:],
        *key.stride()[1:],
        *value.stride()[1:],
        *tail_key.stride()[1:],
        *tail_value.stride()[1:],
        *mask.stride()[1:],
        ends.stride(1),
        n,
        h,
        h // key.shape[1],
        d,
        t,
        16,
        64,
        triton.next_power_of_2(d),
        splits,
        split_size,
        capacity,
        d ** (-0.5),
        probability_mode == "two_term",
        num_warps=4,
    )
    _merge[b * n, h](
        tmp,
        out,
        *out.stride()[1:],
        h,
        d,
        triton.next_power_of_2(d),
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return out.reshape(h, b, n, d).permute(1, 0, 2, 3)
