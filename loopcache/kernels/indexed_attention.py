import torch
import triton
import triton.language as tl


@triton.jit
def _partial(
    Q,
    KP,
    VP,
    POS,
    SLOT,
    VALID,
    OUT,
    TMP,
    QH: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    OH: tl.constexpr,
    ON: tl.constexpr,
    OD: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    N: tl.constexpr,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    SCALE: tl.constexpr,
    DTYPE: tl.constexpr,
    INDEXED: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    pos = tl.load(POS + row)
    kh = h // GROUP
    ix = row
    if INDEXED:
        valid = tl.load(VALID + row)
        ix = tl.where(valid, tl.load(SLOT + row), 0)
        pos = tl.where(valid, pos, -1)
    kp = tl.load(KP + ix).to(tl.pointer_type(DTYPE))
    vp = tl.load(VP + ix).to(tl.pointer_type(DTYPE))
    ds = tl.arange(0, BD)
    q = tl.load(Q + h * QH + row * QN + ds * QD, ds < D, other=0).to(tl.float32)
    m = tl.full((), float("-inf"), tl.float32)
    den = tl.full((), 0.0, tl.float32)
    acc = tl.full((BD,), 0.0, tl.float32)
    begin = sp * CHUNK
    end = tl.minimum((sp + 1) * CHUNK, pos + 1)
    for start in range(begin, end, BT):
        ts = start + tl.arange(0, BT)
        mask = (ts[:, None] < end) & (ds[None, :] < D)
        k = tl.load(kp + kh * KH + ts[:, None] * KT + ds[None, :] * KD, mask, other=0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1) * SCALE
        score = tl.where(ts < end, score, float("-inf"))
        new_m = tl.maximum(m, tl.max(score, axis=0))
        weight = tl.exp(score - new_m)
        alpha = tl.exp(m - new_m)
        v = tl.load(vp + kh * VH + ts[:, None] * VT + ds[None, :] * VD, mask, other=0).to(tl.float32)
        acc = acc * alpha + tl.sum(weight[:, None] * v, axis=0)
        den = den * alpha + tl.sum(weight, axis=0)
        m = new_m
    if SPLITS == 1:
        out = acc / tl.maximum(den, 1e-30)
        tl.store(OUT + h * OH + row * ON + ds * OD, out, ds < D)
    else:
        base = ((row * HQ + h) * SPLITS + sp) * (D + 2)
        tl.store(TMP + base + ds, acc, ds < D)
        tl.store(TMP + base + D, m)
        tl.store(TMP + base + D + 1, den)


@triton.jit
def _merge(
    TMP,
    OUT,
    OH: tl.constexpr,
    ON: tl.constexpr,
    OD: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    ss = tl.arange(0, BS)
    ds = tl.arange(0, BD)
    bases = ((row * HQ + h) * SPLITS + ss) * (D + 2)
    m = tl.load(TMP + bases + D, ss < SPLITS, other=float("-inf"))
    den = tl.load(TMP + bases + D + 1, ss < SPLITS, other=0.0)
    gmax = tl.max(m, axis=0)
    alpha = tl.where(den > 0, tl.exp(m - gmax), 0.0)
    acc = tl.load(
        TMP + bases[:, None] + ds[None, :], (ss[:, None] < SPLITS) & (ds[None, :] < D), other=0.0
    )
    out = tl.sum(acc * alpha[:, None], axis=0) / tl.maximum(tl.sum(den * alpha, axis=0), 1e-30)
    tl.store(OUT + h * OH + row * ON + ds * OD, out, ds < D)


@triton.jit
def _scatter(
    K,
    V,
    KP,
    VP,
    POS,
    SLOT,
    VALID,
    KH: tl.constexpr,
    KN: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VN: tl.constexpr,
    VD: tl.constexpr,
    CKH: tl.constexpr,
    CKT: tl.constexpr,
    CKD: tl.constexpr,
    CVH: tl.constexpr,
    CVT: tl.constexpr,
    CVD: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    valid = tl.load(VALID + row)
    slot = tl.where(valid, tl.load(SLOT + row), 0)
    pos = tl.load(POS + row)
    ds = tl.arange(0, BD)
    kp = tl.load(KP + slot).to(tl.pointer_type(DTYPE))
    vp = tl.load(VP + slot).to(tl.pointer_type(DTYPE))
    kval = tl.load(K + h * KH + row * KN + ds * KD, (ds < D) & valid, other=0)
    vval = tl.load(V + h * VH + row * VN + ds * VD, (ds < D) & valid, other=0)
    tl.store(kp + h * CKH + pos * CKT + ds * CKD, kval, (ds < D) & valid)
    tl.store(vp + h * CVH + pos * CVT + ds * CVD, vval, (ds < D) & valid)


def scatter_kv_from_tables(
    k, v, key_ptrs, value_ptrs, positions, slots, valid, kv_stride, *, value_stride=None
):
    value_stride = kv_stride if value_stride is None else value_stride
    d = k.shape[-1]
    dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[k.dtype]
    _scatter[k.shape[2], k.shape[1]](
        k,
        v,
        key_ptrs,
        value_ptrs,
        positions,
        slots,
        valid,
        *k.stride()[1:],
        *v.stride()[1:],
        *kv_stride[1:],
        *value_stride[1:],
        d,
        triton.next_power_of_2(d),
        dtype,
        num_warps=4,
    )


def indexed_attention_gpu(
    q,
    key_ptrs,
    value_ptrs,
    positions,
    slots,
    valid,
    kv_shape,
    kv_stride,
    *,
    max_length,
    value_stride=None,
    out=None,
    workspace=None,
    split_size=256,
):
    return indexed_attention_from_tables(
        q,
        key_ptrs,
        value_ptrs,
        positions,
        kv_shape,
        kv_stride,
        max_length=max_length,
        value_stride=value_stride,
        out=out,
        workspace=workspace,
        split_size=split_size,
        slots=slots,
        valid=valid,
    )


def indexed_attention_from_tables(
    q,
    key_ptrs,
    value_ptrs,
    positions,
    kv_shape,
    kv_stride,
    *,
    max_length,
    value_stride=None,
    out=None,
    workspace=None,
    split_size=256,
    slots=None,
    valid=None,
    block_size=32,
):
    if not q.is_cuda or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Expected CUDA Q with shape (1,Hq,N,D)")
    _, hq, n, d = q.shape
    if hq % kv_shape[1] or kv_shape[-1] != d:
        raise ValueError("Invalid GQA shape")
    if max_length < 0 or max_length > kv_shape[2]:
        raise ValueError("max_length exceeds KV capacity")
    if n == 0:
        return torch.empty_like(q)
    if (slots is None) != (valid is None):
        raise ValueError("slots and valid must be provided together")
    if positions.numel() != n:
        raise ValueError("Position row mismatch")
    if slots is not None and (slots.numel() != n or valid.numel() != n):
        raise ValueError("Slot/valid row mismatch")
    if any((t.device != q.device for t in (key_ptrs, value_ptrs, positions))):
        raise ValueError("Pointer metadata must be on query device")
    if key_ptrs.dtype != torch.int64 or value_ptrs.dtype != torch.int64:
        raise ValueError("Pointer metadata must be int64")
    splits = max(1, triton.cdiv(max_length, split_size))
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if workspace is None:
        workspace = torch.empty((n, hq, splits, d + 2), dtype=torch.float32, device=q.device)
    if workspace.numel() < n * hq * splits * (d + 2):
        raise ValueError("Workspace is too small")
    value_stride = kv_stride if value_stride is None else value_stride
    dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[q.dtype]
    _partial[n, hq, splits](
        q,
        key_ptrs,
        value_ptrs,
        positions,
        slots if slots is not None else positions,
        valid if valid is not None else positions,
        out,
        workspace,
        *q.stride()[1:],
        *out.stride()[1:],
        *kv_stride[1:],
        *value_stride[1:],
        n,
        hq,
        hq // kv_shape[1],
        d,
        triton.next_power_of_2(d),
        block_size,
        splits,
        split_size,
        d ** (-0.5),
        dtype,
        slots is not None,
        num_warps=4,
    )
    if splits > 1:
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
