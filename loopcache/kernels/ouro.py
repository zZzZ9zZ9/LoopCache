import contextlib
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_pair(
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
    OQH: tl.constexpr,
    OQN: tl.constexpr,
    OQD: tl.constexpr,
    OKH: tl.constexpr,
    OKN: tl.constexpr,
    OKD: tl.constexpr,
    CN: tl.constexpr,
    CD: tl.constexpr,
    SN: tl.constexpr,
    SD: tl.constexpr,
    N: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    ix = tl.program_id(0) * B + tl.arange(0, B)
    h = ix // (N * D)
    n = ix // D % N
    d = ix % D
    mate = tl.where(d < D // 2, d + D // 2, d - D // 2)
    if tl.program_id(1) == 0:
        mask = ix < HQ * N * D
        x = tl.load(Q + h * QH + n * QN + d * QD, mask, other=0).to(tl.float32)
        y = tl.load(Q + h * QH + n * QN + mate * QD, mask, other=0).to(tl.float32)
    else:
        mask = ix < HK * N * D
        x = tl.load(K + h * KH + n * KN + d * KD, mask, other=0).to(tl.float32)
        y = tl.load(K + h * KH + n * KN + mate * KD, mask, other=0).to(tl.float32)
    c = tl.load(C + n * CN + d * CD, mask, other=0).to(tl.float32)
    s = tl.load(S + n * SN + d * SD, mask, other=0).to(tl.float32)
    y = tl.where(d < D // 2, -y, y)
    a = (x * c).to(Q.dtype.element_ty).to(tl.float32)
    b = (y * s).to(Q.dtype.element_ty).to(tl.float32)
    out = (a + b).to(Q.dtype.element_ty)
    if tl.program_id(1) == 0:
        tl.store(OQ + h * OQH + n * OQN + d * OQD, out, mask)
    else:
        tl.store(OK + h * OKH + n * OKN + d * OKD, out, mask)


def rope_pair(q, k, cos, sin):
    if q.device.type != "cuda" or q.ndim != 4 or q.shape[0] != 1 or (k.shape[0] != 1):
        raise ValueError("Expected CUDA batch-one Q/K")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported dtype")
    if not q.dtype == k.dtype == cos.dtype == sin.dtype:
        raise ValueError("RoPE dtypes must match")
    if q.shape[2:] != k.shape[2:] or q.shape[-1] % 2:
        raise ValueError("Q/K shape mismatch")
    if cos.shape != (1, q.shape[2], q.shape[3]) or sin.shape != cos.shape:
        raise ValueError("Expected cosine/sine shape (1,N,D)")
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    _, hq, n, d = q.shape
    hk = k.shape[1]
    _rope_pair[triton.cdiv(max(hq, hk) * n * d, 256), 2](
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
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (oq, ok)


@triton.jit
def _rms_tail(
    X,
    V,
    W,
    R,
    Output,
    XS: tl.constexpr,
    OS: tl.constexpr,
    RS: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    RESIDUAL: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < H
    x = tl.load(X + row * XS + col, mask, other=0).to(tl.float32)
    variance = tl.load(V + row)
    inverse = tl.rsqrt(variance + EPS)
    scaled = (x * inverse).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + col, mask, other=0).to(tl.float32)
    out = (weight * scaled).to(X.dtype.element_ty)
    if RESIDUAL:
        residual = tl.load(R + row * RS + col, mask, other=0).to(tl.float32)
        out = (residual + out.to(tl.float32)).to(X.dtype.element_ty)
    tl.store(Output + row * OS + col, out, mask)


def rms_tail(x, norm, residual=None, *, rms_square="torch"):
    if rms_square not in ("torch", "fused"):
        raise ValueError("rms_square must be torch or fused")
    if x.device.type != "cuda" or x.ndim != 3 or x.shape[0] != 1 or (x.stride(-1) != 1):
        raise ValueError("Expected CUDA (1,N,H) with contiguous last dimension")
    if norm.weight.dtype != x.dtype:
        raise ValueError("Weight/input dtype mismatch")
    if residual is not None and (
        residual.shape != x.shape or residual.dtype != x.dtype or residual.stride(-1) != 1
    ):
        raise ValueError("Residual layout mismatch")
    if rms_square == "torch":
        xf = x.to(torch.float32)
        variance = xf.pow(2).mean(-1, keepdim=True)
    else:
        from loopcache.kernels.ouro_square import square_fp32

        variance = square_fp32(x).mean(-1, keepdim=True)
    out = torch.empty_like(x)
    _rms_tail[x.shape[1],](
        x,
        variance,
        norm.weight,
        residual if residual is not None else x,
        out,
        x.stride(1),
        out.stride(1),
        residual.stride(1) if residual is not None else x.stride(1),
        x.shape[-1],
        float(norm.variance_epsilon),
        residual is not None,
        triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def _ouro_many(adapter, layer, x, freqs, cache, slots, positions, layout):
    def norm(value, module, residual=None):
        return rms_tail(value, module, residual, rms_square=adapter._rms_square)

    residual = x
    h = norm(x, layer.input_layernorm)
    attn = layer.self_attn
    b, n, _ = h.shape
    d = attn.head_dim
    q = attn.q_proj(h).view(b, n, adapter.config.num_attention_heads, d).transpose(1, 2)
    k = attn.k_proj(h).view(b, n, adapter.config.num_key_value_heads, d).transpose(1, 2)
    v = attn.v_proj(h).view(b, n, adapter.config.num_key_value_heads, d).transpose(1, 2)
    q, k = rope_pair(q, k, *freqs)
    y = (
        adapter._attention_many(q, k, v, cache, slots, positions, layout)
        .transpose(1, 2)
        .reshape(b, n, -1)
    )
    x = norm(attn.o_proj(y), layer.input_layernorm_2, residual)
    return norm(layer.mlp(norm(x, layer.post_attention_layernorm)), layer.post_attention_layernorm_2, x)


@contextlib.contextmanager
def patched(adapter, *, rms_square="torch"):
    previous = adapter._ouro_many
    adapter._rms_square = rms_square
    adapter._ouro_many = types.MethodType(_ouro_many, adapter)
    try:
        yield
    finally:
        adapter._ouro_many = previous
        del adapter._rms_square
