import torch
import triton
import triton.language as tl


@triton.jit
def _complex_rope(
    Q,
    K,
    F,
    OQ,
    OK,
    QN: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KN: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    ON: tl.constexpr,
    OH: tl.constexpr,
    OD: tl.constexpr,
    PN: tl.constexpr,
    PH: tl.constexpr,
    PD: tl.constexpr,
    FN: tl.constexpr,
    FD: tl.constexpr,
    FC: tl.constexpr,
    N: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    d = i % D
    h = i // D % HQ
    n = i // (D * HQ)
    if tl.program_id(1) == 0:
        mask = i < N * HQ * D
        x = tl.load(Q + n * QN + h * QH + d * QD, mask, 0).to(tl.float32)
        y = tl.load(Q + n * QN + h * QH + (d ^ 1) * QD, mask, 0).to(tl.float32)
    else:
        h = i // D % HK
        n = i // (D * HK)
        mask = i < N * HK * D
        x = tl.load(K + n * KN + h * KH + d * KD, mask, 0).to(tl.float32)
        y = tl.load(K + n * KN + h * KH + (d ^ 1) * KD, mask, 0).to(tl.float32)
    c = tl.load(F + n * FN + d // 2 * FD, mask, 0)
    s = tl.load(F + n * FN + d // 2 * FD + FC, mask, 0)
    a = x * c
    b = y * s
    z = tl.where(d % 2 == 0, a - b, a + b)
    if tl.program_id(1) == 0:
        tl.store(OQ + n * ON + h * OH + d * OD, z, mask)
    else:
        tl.store(OK + n * PN + h * PH + d * PD, z, mask)


def complex_rope_pair(q, k, freqs_cis):
    if q.device.type != "cuda" or q.ndim != 4 or k.ndim != 4 or (q.shape[0] != 1) or (k.shape[0] != 1):
        raise ValueError("CUDA batch1 Q/K required")
    if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1] or q.shape[-1] % 2:
        raise ValueError("Q/K mismatch")
    if (
        q.dtype != k.dtype
        or q.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or freqs_cis.dtype != torch.float32
    ):
        raise ValueError("dtype mismatch")
    if q.device != k.device or q.device != freqs_cis.device:
        raise ValueError("rotary device mismatch")
    _, n, hq, d = q.shape
    hk = k.shape[2]
    if tuple(freqs_cis.shape) != (1, n, 1, d // 2, 2):
        raise ValueError("frequency shape mismatch")
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    _complex_rope[triton.cdiv(n * max(hq, hk) * d, 256), 2](
        q,
        k,
        freqs_cis,
        oq,
        ok,
        *q.stride()[1:],
        *k.stride()[1:],
        *oq.stride()[1:],
        *ok.stride()[1:],
        freqs_cis.stride(1),
        freqs_cis.stride(3),
        freqs_cis.stride(4),
        n,
        hq,
        hk,
        d,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (oq, ok)


@triton.jit
def _mixed_split_rope(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    QH: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KH: tl.constexpr,
    KN: tl.constexpr,
    KD: tl.constexpr,
    OH: tl.constexpr,
    ON: tl.constexpr,
    OD: tl.constexpr,
    PH: tl.constexpr,
    PN: tl.constexpr,
    PD: tl.constexpr,
    CN: tl.constexpr,
    CD: tl.constexpr,
    SN: tl.constexpr,
    SD: tl.constexpr,
    N: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    ROUND_PRODUCT: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    d = i % D
    n = i // D % N
    h = i // (N * D)
    mate = tl.where(d < D // 2, d + D // 2, d - D // 2)
    if tl.program_id(1) == 0:
        mask = i < HQ * N * D
        x = tl.load(Q + h * QH + n * QN + d * QD, mask, 0).to(tl.float32)
        y = tl.load(Q + h * QH + n * QN + mate * QD, mask, 0).to(tl.float32)
    else:
        mask = i < HK * N * D
        x = tl.load(K + h * KH + n * KN + d * KD, mask, 0).to(tl.float32)
        y = tl.load(K + h * KH + n * KN + mate * KD, mask, 0).to(tl.float32)
    c = tl.load(C + n * CN + d * CD, mask, 0).to(tl.float32)
    s = tl.load(S + n * SN + d * SD, mask, 0).to(tl.float32)
    y = tl.where(d < D // 2, -y, y)
    a = x * c
    b = y * s
    if ROUND_PRODUCT:
        a = a.to(Q.dtype.element_ty).to(tl.float32)
        b = b.to(Q.dtype.element_ty).to(tl.float32)
    z = a + b
    if tl.program_id(1) == 0:
        tl.store(OQ + h * OH + n * ON + d * OD, z, mask)
    else:
        tl.store(OK + h * PH + n * PN + d * PD, z, mask)


def mixed_split_rope_pair(q, k, cos, sin):
    if (
        q.device.type != "cuda"
        or q.ndim != 4
        or q.shape[0] != 1
        or (k.shape[0] != 1)
        or (q.shape[2:] != k.shape[2:])
        or q.shape[-1] % 2
    ):
        raise ValueError("Q/K shape mismatch")
    if (
        q.dtype != k.dtype
        or cos.dtype != sin.dtype
        or q.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or (cos.dtype not in (q.dtype, torch.float32))
    ):
        raise ValueError("unsupported promoted RoPE dtype")
    if q.device != k.device or q.device != cos.device or q.device != sin.device:
        raise ValueError("rotary device mismatch")
    _, hq, n, d = q.shape
    hk = k.shape[1]
    if cos.shape != (1, n, d) or sin.shape != cos.shape:
        raise ValueError("cos/sin shape mismatch")
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    _mixed_split_rope[triton.cdiv(max(hq, hk) * n * d, 256), 2](
        q,
        k,
        cos,
        sin,
        oq,
        ok,
        *q.stride()[1:],
        *k.stride()[1:],
        *oq.stride()[1:],
        *ok.stride()[1:],
        *cos.stride()[1:],
        *sin.stride()[1:],
        n,
        hq,
        hk,
        d,
        cos.dtype == q.dtype and q.dtype != torch.float32,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (oq, ok)
