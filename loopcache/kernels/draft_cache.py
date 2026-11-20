import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LAYER"])
def _scatter_private(
    K,
    V,
    PK,
    PV,
    TAGS,
    POS,
    DEP,
    VALID,
    ENDS,
    TMASK,
    BASE,
    MENDS,
    MTAIL,
    TRUSTED,
    SDEP,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    R: tl.constexpr,
    CORE: tl.constexpr,
    LAYER,
    CAP: tl.constexpr,
    BUILD_META: tl.constexpr,
    MISSING: tl.constexpr,
    UNCOMMITTED: tl.constexpr,
    WINDOW: tl.constexpr,
    BW: tl.constexpr,
    MN: tl.constexpr,
    MT: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    ds = tl.arange(0, BD)
    pos = tl.load(POS + W + row)
    dep = tl.load(DEP + W + row)
    valid = tl.load(VALID + W + row) & (dep < R)
    slot = dep * CORE + LAYER
    ring = pos % CAP
    off = ((slot * H + h) * CAP + ring) * D + ds
    k = tl.load(K + h * KH + (W + row) * KT + ds * KD, valid & (ds < D), other=0)
    v = tl.load(V + h * VH + (W + row) * VT + ds * VD, valid & (ds < D), other=0)
    tl.store(PK + off, k, valid & (ds < D))
    tl.store(PV + off, v, valid & (ds < D))
    if h == 0:
        tl.store(TAGS + slot * CAP + ring, pos, valid)
    if BUILD_META:
        if h == 0:
            specrow = W + row
            base = tl.load(BASE)
            local_start = pos - WINDOW
            if UNCOMMITTED:
                local_start = tl.maximum(local_start, base)
            kvend = tl.load(ENDS + row)
            specend = tl.load(ENDS + specrow)
            if not MISSING:
                specend = tl.minimum(specend, local_start - 1)
            tl.store(MENDS + row, kvend)
            tl.store(MENDS + specrow, specend)
            tl.store(SDEP + row, tl.minimum(tl.load(DEP + row), R - 1))
            tl.store(SDEP + specrow, tl.minimum(dep, R - 1))
            cols = tl.arange(0, 2 * BW)
            colvalid = cols < 2 * W
            kv_mask = tl.load(TMASK + row * MN + cols * MT, colvalid, other=False)
            spec_mask = tl.load(TMASK + specrow * MN + cols * MT, colvalid, other=False)
            if not MISSING:
                colpos = tl.load(POS + cols, colvalid, other=0)
                spec_mask = spec_mask & ((colpos < local_start) | (cols == specrow))
            tl.store(MTAIL + row * (2 * W) + cols, kv_mask, colvalid)
            tl.store(MTAIL + specrow * (2 * W) + cols, spec_mask, colvalid)
            if row == 0:
                lanes = tl.arange(0, BW)
                trusted_count = tl.sum(
                    tl.load(VALID + lanes, lanes < W, other=False).to(tl.int64), axis=0
                )
                tl.store(TRUSTED, base + trusted_count)


@triton.jit(do_not_specialize=["LAYER", "PHASE"])
def _local_partial_tc(
    Q,
    K,
    V,
    KP,
    VP,
    PK,
    PV,
    TAGS,
    TK,
    TV,
    TMASK,
    ENDS,
    POS,
    DEP,
    VALID,
    BASE,
    TRUSTED,
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
    N: tl.constexpr,
    W: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    WINDOW: tl.constexpr,
    CORE: tl.constexpr,
    LAYER,
    NPRE: tl.constexpr,
    PHASE,
    CAP: tl.constexpr,
    DTYPE: tl.constexpr,
    SCALE: tl.constexpr,
    TWOTERM: tl.constexpr,
    MISSING: tl.constexpr,
    MIX: tl.constexpr,
    UNCOMMITTED: tl.constexpr,
):
    row = tl.program_id(0) + W
    kh = tl.program_id(1)
    ds = tl.arange(0, BD)
    hs = tl.arange(0, BH)
    heads = kh * G + hs
    pos = tl.load(POS + row)
    dep = tl.load(DEP + row)
    valid = tl.load(VALID + row)
    base = tl.load(BASE)
    trusted = tl.load(TRUSTED)
    original_end = tl.load(ENDS + row)
    slot = dep * CORE + LAYER
    main_slot = NPRE + slot
    mkp = tl.load(KP + main_slot).to(tl.pointer_type(DTYPE))
    mvp = tl.load(VP + main_slot).to(tl.pointer_type(DTYPE))
    q = tl.load(
        Q + heads[:, None] * QH + row * QN + ds[None, :] * QD,
        (hs[:, None] < G) & (ds[None, :] < D),
        other=0,
    )
    m = tl.full((BH,), float("-inf"), tl.float32)
    den = tl.full((BH,), 0.0, tl.float32)
    acc = tl.full((BH, BD), 0.0, tl.float32)
    if valid:
        for start in range(0, WINDOW, BT):
            offsets = start + tl.arange(0, BT)
            ts = pos - WINDOW + offsets
            inrange = (offsets < WINDOW) & (ts >= 0) & (ts < pos)
            ring = ts % CAP
            tag = tl.load(TAGS + slot * CAP + ring, inrange, other=-2)
            own_main = inrange & ((ts < base) | (dep <= PHASE) & (ts < trusted))
            own_pool = inrange & ~own_main & (tag == ts)
            fallback_main = inrange & ~own_main & ~own_pool & (ts <= original_end)
            tail_ix = W + ts - base
            visible_tail = tl.load(
                TMASK + row * MN + tail_ix * MT, inrange & (tail_ix >= W) & (tail_ix < N), other=False
            )
            fallback_tail = inrange & ~own_main & ~own_pool & ~fallback_main & visible_tail
            shared_main = inrange & (ts <= original_end)
            shared_tail = inrange & ~shared_main & visible_tail
            shared_valid = shared_main | shared_tail
            any_valid = own_main | own_pool | fallback_main | fallback_tail
            if MISSING:
                any_valid = any_valid & ~((ts <= original_end) | visible_tail)
            if UNCOMMITTED:
                any_valid = any_valid & (ts >= base)
            dimmask = ds[None, :] < D
            kbase = tl.where(
                own_main,
                mkp + kh * KH + ts * KT,
                tl.where(
                    own_pool,
                    PK + ((slot * HK + kh) * CAP + ring) * D,
                    tl.where(fallback_main, K + kh * KH + ts * KT, TK + kh * TH + tail_ix * TT),
                ),
            )
            kdim = tl.where(own_pool, 1, tl.where(fallback_tail, TD, KD))
            if MIX == 2:
                shared_kbase = tl.where(shared_main, K + kh * KH + ts * KT, TK + kh * TH + tail_ix * TT)
                kbase = tl.where(shared_valid, shared_kbase, kbase)
                kdim = tl.where(shared_valid, tl.where(shared_main, KD, TD), kdim)
            kval = tl.load(
                kbase[:, None] + ds[None, :] * kdim[:, None], any_valid[:, None] & dimmask, other=0
            )
            score = tl.dot(q, tl.trans(kval)).to(tl.float32) * SCALE
            score = tl.where((hs[:, None] < G) & any_valid[None, :], score, float("-inf"))
            new_m = tl.maximum(m, tl.max(score, axis=1))
            safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
            prob = tl.exp(score - safe_m[:, None])
            alpha = tl.exp(m - safe_m)
            vbase = tl.where(
                own_main,
                mvp + kh * VH + ts * VT,
                tl.where(
                    own_pool,
                    PV + ((slot * HK + kh) * CAP + ring) * D,
                    tl.where(fallback_main, V + kh * VH + ts * VT, TV + kh * UH + tail_ix * UT),
                ),
            )
            vdim = tl.where(own_pool, 1, tl.where(fallback_tail, UD, VD))
            if MIX == 1 or MIX == 3:
                prefer_shared = shared_valid
                if MIX == 3:
                    prefer_shared = prefer_shared & (PHASE > dep)
                shared_vbase = tl.where(shared_main, V + kh * VH + ts * VT, TV + kh * UH + tail_ix * UT)
                vbase = tl.where(prefer_shared, shared_vbase, vbase)
                vdim = tl.where(prefer_shared, tl.where(shared_main, VD, UD), vdim)
            vval = tl.load(
                vbase[:, None] + ds[None, :] * vdim[:, None], any_valid[:, None] & dimmask, other=0
            )
            hi = prob.to(q.dtype)
            acc = tl.dot(hi, vval, acc * alpha[:, None])
            if TWOTERM:
                lo = (prob - hi.to(tl.float32)).to(q.dtype)
                acc = tl.dot(lo, vval, acc)
            den = den * alpha + tl.sum(prob, axis=1)
            m = new_m
    out = ((row - W) * HQ + heads) * (D + 2)
    tl.store(OUT + out[:, None] + ds[None, :], acc, (hs[:, None] < G) & (ds[None, :] < D))
    tl.store(OUT + out + D, m, hs < G)
    tl.store(OUT + out + D + 1, den, hs < G)


@triton.jit
def _merge_local(
    TMP,
    LOCAL,
    OUT,
    OH: tl.constexpr,
    ON: tl.constexpr,
    OD: tl.constexpr,
    W: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0) + W
    h = tl.program_id(1)
    localbase = ((row - W) * HQ + h) * (D + 2)
    ld = tl.load(LOCAL + localbase + D + 1)
    if ld > 0:
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
        numerator = tl.sum(acc * alpha[:, None], axis=0)
        denominator = tl.sum(den * alpha, axis=0)
        lm = tl.load(LOCAL + localbase + D)
        la = tl.load(LOCAL + localbase + ds, ds < D, other=0.0)
        bothmax = tl.maximum(gmax, lm)
        left = tl.where(denominator > 0, tl.exp(gmax - bothmax), 0.0)
        right = tl.where(ld > 0, tl.exp(lm - bothmax), 0.0)
        numerator = numerator * left + la * right
        denominator = denominator * left + ld * right
        out = numerator / tl.maximum(denominator, 1e-30)
        tl.store(OUT + h * OH + row * ON + ds * OD, out, ds < D)


class DraftSlidingPool:
    def __init__(self, engine, window, mode="own", full_backend="gather", full_split_size=1024):
        self.window = window
        self.full_split_size = full_split_size
        self.full_workspace = None
        self.R = engine.R
        self.W = engine.W
        self.core = engine.a.n_core
        self.capacity = engine.W
        main = engine.cache.keys[engine.slot_keys[0]]
        _, self.hkv, _, self.dim = main.shape
        self.keys = torch.empty(
            (self.R * self.core, self.hkv, self.capacity, self.dim), device=main.device, dtype=main.dtype
        )
        self.values = torch.empty_like(self.keys)
        self.tags = torch.full(
            (self.R * self.core, self.capacity), -1, device=main.device, dtype=torch.long
        )
        self.allocated_bytes = sum(
            (x.numel() * x.element_size() for x in (self.keys, self.values, self.tags))
        )
        self.workspace = {}
        self.local_scratch = None
        self.observed_tiles = set()
        self._metadata_source = None
        self._metadata = (
            torch.empty((2 * self.W,), device=main.device, dtype=torch.long),
            torch.empty((2 * self.W, 2 * self.W), device=main.device, dtype=torch.bool),
            torch.empty((), device=main.device, dtype=torch.long),
            torch.empty((2 * self.W,), device=main.device, dtype=torch.long),
        )
        self.row_ids = torch.arange(2 * self.W, device=main.device)
        self.self_mask = self.row_ids[:, None] == self.row_ids[None, :]
        self.is_spec = self.row_ids >= self.W

    def reset(self):
        self.tags.fill_(-1)
        self._metadata_source = None

    def invalidate_from(self, absolute_position):
        self.tags.masked_fill_(self.tags >= absolute_position, -1)
