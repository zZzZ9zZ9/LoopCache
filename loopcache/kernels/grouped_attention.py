import torch
import triton
import triton.language as tl
from loopcache.kernels.indexed_attention import _merge
from loopcache.kernels.shared_attention import shared_prefix_attention as _original


@triton.jit
def _gqa_partial(
    Q,
    K,
    V,
    ENDS,
    TK,
    TV,
    MASK,
    OUT,
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
    QROWS: tl.constexpr,
    HEADS: tl.constexpr,
    HTILES: tl.constexpr,
    CHUNK: tl.constexpr,
    L: tl.constexpr,
    SCALE: tl.constexpr,
    TWOTERM: tl.constexpr,
):
    tile = tl.program_id(0)
    flat = tl.arange(0, BM)
    rows = tile // HTILES * QROWS + flat // HEADS
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    local_h = tile % HTILES * HEADS + flat % HEADS
    h = kh * G + local_h
    valid = (rows < N) & (local_h < G)
    ds = tl.arange(0, BD)
    ends = tl.load(ENDS + rows * EN, valid, other=-1)
    q = tl.load(
        Q + h[:, None] * QH + rows[:, None] * QN + ds[None, :] * QD,
        valid[:, None] & (ds[None, :] < D),
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
            allow = valid[:, None] & (ts[None, :] < end) & (ts[None, :] <= ends[:, None])
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
                valid[:, None] & (ts[None, :] < T),
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
    base = ((rows * HQ + h) * S + sp) * (D + 2)
    tl.store(OUT + base[:, None] + ds[None, :], acc, valid[:, None] & (ds[None, :] < D))
    tl.store(OUT + base + D, m, valid)
    tl.store(OUT + base + D + 1, den, valid)


def shared_prefix_attention_gqa(
    q,
    key,
    value,
    prefix_ends,
    tail_key,
    tail_value,
    tail_mask,
    *,
    max_length=None,
    out=None,
    workspace=None,
    split_size=2048,
    query_block=16,
    key_block=64,
    probability_mode="single",
    head_block=None,
):
    if q.ndim != 4 or q.shape[0] != 1 or (not q.is_cuda):
        raise ValueError("CUDA batch-one Q is required")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Shared tensor-core attention requires BF16 or FP16")
    _, hq, n, d = q.shape
    if (
        key.ndim != 4
        or key.shape[0] != 1
        or key.shape[-1] != d
        or (key.shape[1] < 1)
        or hq % key.shape[1]
    ):
        raise ValueError("Invalid shared GQA KV")
    g = hq // key.shape[1]
    if g == 1:
        return _original(
            q,
            key,
            value,
            prefix_ends,
            tail_key,
            tail_value,
            tail_mask,
            max_length=max_length,
            out=out,
            workspace=workspace,
            split_size=split_size,
            query_block=16 if query_block < 16 else query_block,
            key_block=key_block,
            probability_mode=probability_mode,
        )
    if g not in (2, 4, 8, 16):
        raise ValueError("GQA group size must be 2, 4, 8 or 16")
    heads = min(g, 8) if head_block is None else head_block
    if heads not in (1, 2, 4, 8) or heads > g:
        raise ValueError("Invalid grouped-head tile")
    if query_block not in (2, 4, 8, 16, 32) or not 16 <= query_block * heads <= 128:
        raise ValueError("Combined query/head tile M must be between 16 and 128")
    if key.shape != value.shape or tail_key.shape != tail_value.shape:
        raise ValueError("K/V shapes differ")
    if tail_key.ndim != 4 or tail_key.shape[:2] != key.shape[:2] or tail_key.shape[-1] != d:
        raise ValueError("Invalid tail shape")
    t = tail_key.shape[2]
    if prefix_ends.dtype not in (torch.int32, torch.int64):
        raise ValueError("prefix_ends must be integer positions")
    if prefix_ends.shape != (n,) or tail_mask.shape != (n, t) or tail_mask.dtype != torch.bool:
        raise ValueError("Invalid row metadata")
    if any((x.device != q.device for x in (key, value, prefix_ends, tail_key, tail_value, tail_mask))):
        raise ValueError("All tensors must share a device")
    if any((x.dtype != q.dtype for x in (key, value, tail_key, tail_value))):
        raise ValueError("Q/K/V dtype mismatch")
    if d not in (64, 96, 128) or key_block not in (32, 64):
        raise ValueError("Unsupported tile or head size")
    if (
        split_size < key_block
        or split_size % key_block
        or probability_mode not in ("single", "two_term")
    ):
        raise ValueError("Invalid split/probability mode")
    max_length = key.shape[2] if max_length is None else max_length
    if not 0 <= max_length <= key.shape[2]:
        raise ValueError("Invalid prefix capacity")
    if not n:
        return torch.empty_like(q)
    splits = triton.cdiv(max_length, split_size) + 1
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if workspace is None:
        workspace = torch.empty((n, hq, splits, d + 2), device=q.device, dtype=torch.float32)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("Invalid output buffer")
    if (
        workspace.dtype != torch.float32
        or workspace.device != q.device
        or (not workspace.is_contiguous())
    ):
        raise ValueError("Workspace must be contiguous CUDA FP32")
    if workspace.numel() < n * hq * splits * (d + 2):
        raise ValueError("Workspace too small")
    bm = query_block * heads
    _gqa_partial[triton.cdiv(n, query_block) * triton.cdiv(g, heads), key.shape[1], splits](
        q,
        key,
        value,
        prefix_ends,
        tail_key,
        tail_value,
        tail_mask,
        workspace,
        *q.stride()[1:],
        *key.stride()[1:],
        *value.stride()[1:],
        *tail_key.stride()[1:],
        *tail_value.stride()[1:],
        *tail_mask.stride(),
        prefix_ends.stride(0),
        n,
        hq,
        g,
        d,
        t,
        bm,
        key_block,
        triton.next_power_of_2(d),
        splits,
        query_block,
        heads,
        triton.cdiv(g, heads),
        split_size,
        max_length,
        d ** (-0.5),
        probability_mode == "two_term",
        num_warps=8 if bm >= 128 else 4,
    )
    _merge[n, hq](
        workspace,
        out,
        *out.stride()[1:],
        hq,
        d,
        triton.next_power_of_2(d),
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return out
