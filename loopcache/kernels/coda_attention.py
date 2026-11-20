import types
import torch
import triton
import triton.language as tl
from loopcache.kernels.indexed_attention import _merge, scatter_kv_from_tables


@triton.jit
def _shared_partial(
    Q,
    K,
    V,
    POS,
    VALID,
    OUT,
    TMP,
    QH: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    OH: tl.constexpr,
    ON: tl.constexpr,
    OD: tl.constexpr,
    N: tl.constexpr,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    MAXL: tl.constexpr,
    SCALE: tl.constexpr,
    TWOTERM: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    rows = block * BM + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    pos = tl.load(POS + rows, rows < N, other=-1)
    valid = tl.load(VALID + rows, rows < N, other=False) & (pos >= 0) & (rows < N)
    maxpos = tl.minimum(tl.max(tl.where(valid, pos, -1), axis=0), MAXL - 1)
    q = tl.load(
        Q + head * QH + rows[:, None] * QN + ds[None, :] * QD,
        valid[:, None] & (ds[None, :] < D),
        other=0,
    )
    kh = head // GROUP
    m = tl.full((BM,), float("-inf"), tl.float32)
    den = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, BD), 0.0, tl.float32)
    begin = split * CHUNK
    end = tl.minimum((split + 1) * CHUNK, maxpos + 1)
    for start in range(begin, end, BN):
        ts = start + tl.arange(0, BN)
        k = tl.load(
            K + kh * KH + ds[:, None] * KD + ts[None, :] * KT,
            (ds[:, None] < D) & (ts[None, :] < end),
            other=0,
        )
        score = tl.dot(q, k).to(tl.float32) * SCALE
        score = tl.where(
            valid[:, None] & (ts[None, :] < end) & (ts[None, :] <= pos[:, None]), score, float("-inf")
        )
        new_m = tl.maximum(m, tl.max(score, axis=1))
        safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
        prob = tl.exp(score - safe_m[:, None])
        alpha = tl.exp(m - safe_m)
        v = tl.load(
            V + kh * VH + ts[:, None] * VT + ds[None, :] * VD,
            (ts[:, None] < end) & (ds[None, :] < D),
            other=0,
        )
        acc = acc * alpha[:, None]
        prob_hi = prob.to(q.dtype)
        acc = tl.dot(prob_hi, v, acc)
        if TWOTERM:
            prob_lo = (prob - prob_hi.to(tl.float32)).to(q.dtype)
            acc = tl.dot(prob_lo, v, acc)
        den = den * alpha + tl.sum(prob, axis=1)
        m = new_m
    if SPLITS == 1:
        result = acc / tl.maximum(den[:, None], 1e-30)
        tl.store(
            OUT + head * OH + rows[:, None] * ON + ds[None, :] * OD,
            result,
            (rows[:, None] < N) & (ds[None, :] < D),
        )
    else:
        base = ((rows * HQ + head) * SPLITS + split) * (D + 2)
        tl.store(TMP + base[:, None] + ds[None, :], acc, (rows[:, None] < N) & (ds[None, :] < D))
        tl.store(TMP + base + D, m, rows < N)
        tl.store(TMP + base + D + 1, den, rows < N)


def shared_attention(
    q,
    key,
    value,
    positions,
    valid,
    *,
    max_length,
    out=None,
    workspace=None,
    query_block=16,
    key_block=64,
    split_size=512,
    probability_mode="two_term",
):
    if probability_mode not in ("single", "two_term"):
        raise ValueError("Invalid probability mode")
    if q.ndim != 4 or q.shape[0] != 1 or (not q.is_cuda):
        raise ValueError("CUDA batch-one Q required")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Tensor-core attention requires BF16/FP16 only")
    if key.dtype != q.dtype or value.dtype != q.dtype or key.shape != value.shape:
        raise ValueError("KV shape/dtype mismatch")
    _, hq, n, d = q.shape
    if key.shape[0] != 1 or key.shape[-1] != d or hq % key.shape[1]:
        raise ValueError("Invalid GQA shape")
    if d not in (64, 128):
        raise ValueError("Validated head dimensions:64/128")
    if not 0 <= max_length <= key.shape[2]:
        raise ValueError("Invalid capacity bound")
    if positions.numel() != n or valid.numel() != n:
        raise ValueError("Row metadata mismatch")
    if any((x.device != q.device for x in (key, value, positions, valid))):
        raise ValueError("Device mismatch")
    if query_block not in (16, 32) or key_block not in (32, 64) or split_size % key_block:
        raise ValueError("Invalid tiling")
    if not n:
        return torch.empty_like(q)
    splits = max(1, triton.cdiv(max_length, split_size))
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if workspace is None:
        workspace = torch.empty((n, hq, splits, d + 2), device=q.device, dtype=torch.float32)
    if workspace.numel() < n * hq * splits * (d + 2):
        raise ValueError("Workspace too small")
    _shared_partial[triton.cdiv(n, query_block), hq, splits](
        q,
        key,
        value,
        positions,
        valid,
        out,
        workspace,
        *q.stride()[1:],
        *key.stride()[1:],
        *value.stride()[1:],
        *out.stride()[1:],
        n,
        hq,
        hq // key.shape[1],
        d,
        query_block,
        key_block,
        triton.next_power_of_2(d),
        splits,
        split_size,
        max_length,
        d ** (-0.5),
        probability_mode == "two_term",
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


class SharedCodaAttention:
    def __init__(
        self,
        adapter,
        *,
        query_block=16,
        key_block=64,
        split_size=512,
        min_length=0,
        probability_mode="two_term",
    ):
        if adapter.family not in ("raven", "huginn"):
            raise ValueError("Coda-family adapter required")
        self.a = adapter
        self.tile = dict(
            query_block=query_block,
            key_block=key_block,
            split_size=split_size,
            probability_mode=probability_mode,
        )
        self.min_length = min_length
        self.active_coda = None

    def __enter__(self):
        from loopcache.dynamic import DeviceEngine

        self.cls = DeviceEngine
        self.original_attention = DeviceEngine.attention
        self.original_native = self.a._native_many
        indices = {id(b): i for i, b in enumerate(self.a.inner.coda)}
        ctx = self

        def native(adapter, block, *args, **kwargs):
            old = ctx.active_coda
            ctx.active_coda = indices.get(id(block))
            try:
                return ctx.original_native(block, *args, **kwargs)
            finally:
                ctx.active_coda = old

        def attention(engine, q, k, v, cache, slots, positions, layout=None):
            idx = ctx.active_coda
            if (
                engine.a is not ctx.a
                or idx is None
                or q.dtype not in (torch.bfloat16, torch.float16)
                or (engine.capacity < ctx.min_length)
            ):
                return ctx.original_attention(engine, q, k, v, cache, slots, positions, layout)
            if cache is not engine.cache:
                raise RuntimeError("Unexpected cache view in device Coda")
            scatter_kv_from_tables(
                k,
                v,
                engine.kptr,
                engine.vptr,
                engine.positions,
                engine.slots,
                engine.write_valid,
                engine.kstride,
                value_stride=engine.vstride,
            )
            return shared_attention(
                q,
                cache.keys[-(idx + 1)],
                cache.values[-(idx + 1)],
                engine.positions,
                engine.read_valid,
                max_length=engine.capacity,
                **ctx.tile,
            )

        self.native = types.MethodType(native, self.a)
        self.attention = attention
        self.a._native_many = self.native
        DeviceEngine.attention = attention
        return self

    def __exit__(self, *exc):
        if self.a._native_many is not self.native or self.cls.attention is not self.attention:
            raise RuntimeError("Concurrent adapter/attention patch")
        self.a._native_many = self.original_native
        self.cls.attention = self.original_attention
