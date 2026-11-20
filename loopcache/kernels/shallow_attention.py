import torch
import triton
import triton.language as tl
from loopcache.kernels.draft_cache import (
    _scatter_private,
    DraftSlidingPool,
    _merge_local,
    _local_partial_tc,
)
from loopcache.kernels.shared_attention import _partial
from loopcache.kernels.grouped_attention import _gqa_partial
from loopcache.kernels.indexed_attention import _merge


@triton.jit(do_not_specialize=["LAYER", "PHASE", "POSITIVE", "NEGATIVE"])
def _scatter_gated(
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
    FULL_MASK,
    FULL_ROWS,
    FULL_COUNT,
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
    FALLBACK: tl.constexpr,
    WINDOW: tl.constexpr,
    PHASE,
    POSITIVE,
    NEGATIVE,
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
            delta = dep - PHASE
            full = valid & ((POSITIVE > 0) & (delta >= POSITIVE) | (NEGATIVE > 0) & (-delta >= NEGATIVE))
            tl.store(FULL_MASK + row, False)
            tl.store(FULL_MASK + W + row, full)
            base = tl.load(BASE)
            local_start = pos - tl.where(full, WINDOW, FALLBACK)
            kvend = tl.load(ENDS + row)
            specend = tl.load(ENDS + W + row)
            specend = tl.where(full | (FALLBACK > 0), tl.minimum(specend, local_start - 1), specend)
            tl.store(MENDS + row, kvend)
            tl.store(MENDS + W + row, specend)
            tl.store(SDEP + row, tl.minimum(tl.load(DEP + row), R - 1))
            tl.store(SDEP + W + row, tl.minimum(dep, R - 1))
            cols = tl.arange(0, 2 * BW)
            colvalid = cols < 2 * W
            kv_mask = tl.load(TMASK + row * MN + cols * MT, colvalid, other=False)
            spec_mask = tl.load(TMASK + (W + row) * MN + cols * MT, colvalid, other=False)
            colpos = tl.load(POS + cols, colvalid, other=0)
            spec_mask = spec_mask & (
                ~full & (FALLBACK == 0) | (colpos < local_start) | (cols == W + row)
            )
            tl.store(MTAIL + row * (2 * W) + cols, kv_mask, colvalid)
            tl.store(MTAIL + (W + row) * (2 * W) + cols, spec_mask, colvalid)
            if row == 0:
                lanes = tl.arange(0, BW)
                trusted_count = tl.sum(
                    tl.load(VALID + lanes, lanes < W, other=False).to(tl.int64), axis=0
                )
                tl.store(TRUSTED, base + trusted_count)
                depths = tl.load(DEP + W + lanes, lanes < W, other=R)
                active = tl.load(VALID + W + lanes, lanes < W, other=False) & (depths < R)
                gaps = depths - PHASE
                selected = active & (
                    (POSITIVE > 0) & (gaps >= POSITIVE) | (NEGATIVE > 0) & (-gaps >= NEGATIVE)
                )
                ordinal = tl.cumsum(selected.to(tl.int32), axis=0) - 1
                tl.store(FULL_ROWS + ordinal, lanes + W, selected)
                tl.store(FULL_COUNT, tl.sum(selected.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["LAYER", "PHASE"])
def _gated_full_partial(
    Q,
    K,
    V,
    KP,
    VP,
    ENDS,
    POS,
    DEP,
    VALID,
    BASE,
    OUT,
    FULL_ROWS,
    FULL_COUNT,
    QH: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
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
    DTYPE: tl.constexpr,
    SCALE: tl.constexpr,
    TWOTERM: tl.constexpr,
    MIX: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    count = tl.load(FULL_COUNT)
    safe_count = tl.maximum(count, 1)
    for task in range(pid, count * HK * SPLITS, GRID):
        ordinal = task % safe_count
        kh = task // safe_count % HK
        sp = task // (safe_count * HK)
        row = tl.load(FULL_ROWS + ordinal)
        ds = tl.arange(0, BD)
        hs = tl.arange(0, BH)
        heads = kh * G + hs
        pos = tl.load(POS + row)
        dep = tl.load(DEP + row)
        valid = tl.load(VALID + row)
        base = tl.load(BASE)
        original_end = tl.load(ENDS + row)
        slot = NPRE + dep * CORE + LAYER
        kp = tl.load(KP + slot).to(tl.pointer_type(DTYPE))
        vp = tl.load(VP + slot).to(tl.pointer_type(DTYPE))
        q = tl.load(
            Q + heads[:, None] * QH + row * QN + ds[None, :] * QD,
            (hs[:, None] < G) & (ds[None, :] < D),
            other=0,
        )
        m = tl.full((BH,), float("-inf"), tl.float32)
        den = tl.full((BH,), 0.0, tl.float32)
        acc = tl.full((BH, BD), 0.0, tl.float32)
        if valid:
            end = tl.minimum((sp + 1) * CHUNK, tl.minimum(pos, base))
            for start in range(sp * CHUNK, end, BT):
                ts = start + tl.arange(0, BT)
                allow = (ts < end) & (ts >= pos - WINDOW)
                km = kp + kh * KH + ts[:, None] * KT + ds[None, :] * KD
                if MIX == 2:
                    km = tl.where(
                        (ts <= original_end)[:, None],
                        K + kh * KH + ts[:, None] * KT + ds[None, :] * KD,
                        km,
                    )
                kval = tl.load(km, allow[:, None] & (ds[None, :] < D), other=0)
                score = tl.dot(q, tl.trans(kval)).to(tl.float32) * SCALE
                score = tl.where((hs[:, None] < G) & allow[None, :], score, float("-inf"))
                new_m = tl.maximum(m, tl.max(score, axis=1))
                safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
                prob = tl.exp(score - safe_m[:, None])
                alpha = tl.exp(m - safe_m)
                vm = vp + kh * VH + ts[:, None] * VT + ds[None, :] * VD
                if MIX == 1 or MIX == 3:
                    shared = ts <= original_end
                    if MIX == 3:
                        shared = shared & (PHASE > dep)
                    vm = tl.where(shared[:, None], V + kh * VH + ts[:, None] * VT + ds[None, :] * VD, vm)
                vval = tl.load(vm, allow[:, None] & (ds[None, :] < D), other=0)
                hi = prob.to(q.dtype)
                acc = tl.dot(hi, vval, acc * alpha[:, None])
                if TWOTERM:
                    lo = (prob - hi.to(tl.float32)).to(q.dtype)
                    acc = tl.dot(lo, vval, acc)
                den = den * alpha + tl.sum(prob, axis=1)
                m = new_m
        out = (((row - W) * HQ + heads) * SPLITS + sp) * (D + 2)
        tl.store(OUT + out[:, None] + ds[None, :], acc, (hs[:, None] < G) & (ds[None, :] < D))
        tl.store(OUT + out + D, m, hs < G)
        tl.store(OUT + out + D + 1, den, hs < G)


@triton.jit(do_not_specialize=["LAYER", "PHASE"])
def _gated_local_partial(
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
    FULL_MASK,
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
    FALLBACK: tl.constexpr,
):
    full = tl.load(FULL_MASK + W + tl.program_id(0))
    if full:
        _local_partial_tc(
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
            QH,
            QN,
            QD,
            KH,
            KT,
            KD,
            VH,
            VT,
            VD,
            TH,
            TT,
            TD,
            UH,
            UT,
            UD,
            MN,
            MT,
            N,
            W,
            HQ,
            HK,
            G,
            D,
            BD,
            16,
            BH,
            W,
            CORE,
            LAYER,
            NPRE,
            PHASE,
            CAP,
            DTYPE,
            SCALE,
            TWOTERM,
            MISSING,
            MIX,
            True,
        )
    elif FALLBACK > 0:
        _local_partial_tc(
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
            QH,
            QN,
            QD,
            KH,
            KT,
            KD,
            VH,
            VT,
            VD,
            TH,
            TT,
            TD,
            UH,
            UT,
            UD,
            MN,
            MT,
            N,
            W,
            HQ,
            HK,
            G,
            D,
            BD,
            16 if FALLBACK <= 16 else 64,
            BH,
            FALLBACK,
            CORE,
            LAYER,
            NPRE,
            PHASE,
            CAP,
            DTYPE,
            SCALE,
            TWOTERM,
            MISSING,
            MIX,
            False,
        )
    else:
        row = tl.program_id(0)
        kh = tl.program_id(1)
        ds = tl.arange(0, BD)
        hs = tl.arange(0, BH)
        heads = kh * G + hs
        out = (row * HQ + heads) * (D + 2)
        tl.store(OUT + out[:, None] + ds[None, :], 0.0, (hs[:, None] < G) & (ds[None, :] < D))
        tl.store(OUT + out + D, float("-inf"), hs < G)
        tl.store(OUT + out + D + 1, 0.0, hs < G)


@triton.jit
def _gated_reduce(
    TMP,
    TAIL,
    OUT,
    FULL_MASK,
    W: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    full = tl.load(FULL_MASK + W + row)
    if full:
        ss = tl.arange(0, BS)
        ds = tl.arange(0, BD)
        src = ((row * HQ + h) * SPLITS + ss) * (D + 2)
        dst = (row * HQ + h) * (D + 2)
        m = tl.load(TMP + src + D, ss < SPLITS, other=float("-inf"))
        den = tl.load(TMP + src + D + 1, ss < SPLITS, other=0.0)
        tm = tl.load(TAIL + dst + D)
        td = tl.load(TAIL + dst + D + 1)
        gmax = tl.maximum(tl.max(m, axis=0), tm)
        safe = tl.where(gmax == float("-inf"), 0.0, gmax)
        alpha = tl.where(den > 0, tl.exp(m - safe), 0.0)
        ta = tl.where(td > 0, tl.exp(tm - safe), 0.0)
        acc = tl.load(
            TMP + src[:, None] + ds[None, :], (ss[:, None] < SPLITS) & (ds[None, :] < D), other=0.0
        )
        tailacc = tl.load(TAIL + dst + ds, ds < D, other=0.0)
        numerator = tl.sum(acc * alpha[:, None], axis=0) + tailacc * ta
        denominator = tl.sum(den * alpha, axis=0) + td * ta
        tl.store(OUT + dst + ds, numerator, ds < D)
        tl.store(OUT + dst + D, gmax)
        tl.store(OUT + dst + D + 1, denominator)
    else:
        ds = tl.arange(0, BD)
        dst = (row * HQ + h) * (D + 2)
        copy_acc = tl.load(TAIL + dst + ds, ds < D, other=0.0)
        copy_m = tl.load(TAIL + dst + D)
        copy_den = tl.load(TAIL + dst + D + 1)
        tl.store(OUT + dst + ds, copy_acc, ds < D)
        tl.store(OUT + dst + D, copy_m)
        tl.store(OUT + dst + D + 1, copy_den)


class ShallowLayerPool(DraftSlidingPool):
    def __init__(self, engine, full_gate, full_split_size=256):
        super().__init__(engine, engine.capacity, full_split_size=full_split_size)
        self.fallback_window = 0
        if isinstance(full_gate[0], int):
            full_gate = [tuple(full_gate)] * engine.a.n_core
        self.full_gate = tuple(full_gate)
        device = self.keys.device
        self.full_mask = torch.empty((2 * self.W,), device=device, dtype=torch.bool)
        self.full_rows = torch.empty((self.W,), device=device, dtype=torch.int32)
        self.full_count = torch.empty((), device=device, dtype=torch.int32)
        self._gate_source = None
        splits = triton.cdiv(engine.capacity, full_split_size)
        self.full_blocks = min(
            self.W * self.hkv * splits,
            4 * torch.cuda.get_device_properties(device).multi_processor_count,
        )

    def reset(self):
        super().reset()
        if hasattr(self, "full_mask"):
            self.full_mask.zero_()
            self.full_count.zero_()
            self._gate_source = None

    def attention(self, engine, q, k, v, shared_attention=None):
        _, hq, n, d = q.shape
        if n != 2 * self.W:
            raise ValueError("Shallow KV requires the fixed two-path core rows")
        if d not in (64, 96, 128) or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Unsupported Shallow KV attention dtype/head size")
        key = engine.cache.keys[engine.physical_slot]
        value = engine.cache.values[engine.physical_slot]
        positions = engine.attn_positions
        depths = engine.draft_depths
        pair = self.full_gate[engine.draft_core_layer]
        positive, negative = pair
        phase = engine.draft_phase
        if not (0 < positive <= engine.R - 1 - phase or 0 < negative <= phase):
            ends, tail, trusted, safe_depths = self._metadata
            _scatter_private[self.W, self.hkv](
                k,
                v,
                self.keys,
                self.values,
                self.tags,
                positions,
                depths,
                engine.read_valid,
                engine.prefix_ends,
                engine.tail_mask,
                engine.draft_base,
                ends,
                tail,
                trusted,
                safe_depths,
                *k.stride()[1:],
                *v.stride()[1:],
                self.W,
                self.hkv,
                d,
                triton.next_power_of_2(d),
                self.R,
                self.core,
                engine.draft_core_layer,
                self.capacity,
                False,
                False,
                False,
                self.window,
                triton.next_power_of_2(self.W),
                *engine.tail_mask.stride(),
                num_warps=4,
            )
            self._metadata_source = None
            return engine.shared_attention(q, key, value, k, v)
        gate_source = (pair, engine.draft_phase)
        build_metadata = (
            self._metadata_source is not engine.prefix_ends or self._gate_source != gate_source
        )
        ends, tail, trusted, safe_depths = self._metadata
        _scatter_gated[self.W, self.hkv](
            k,
            v,
            self.keys,
            self.values,
            self.tags,
            positions,
            depths,
            engine.read_valid,
            engine.prefix_ends,
            engine.tail_mask,
            engine.draft_base,
            ends,
            tail,
            trusted,
            safe_depths,
            self.full_mask,
            self.full_rows,
            self.full_count,
            *k.stride()[1:],
            *v.stride()[1:],
            self.W,
            self.hkv,
            d,
            triton.next_power_of_2(d),
            self.R,
            self.core,
            engine.draft_core_layer,
            self.capacity,
            build_metadata,
            self.fallback_window,
            self.window,
            engine.draft_phase,
            pair[0],
            pair[1],
            triton.next_power_of_2(self.W),
            *engine.tail_mask.stride(),
            num_warps=4,
        )
        if build_metadata:
            self._metadata_source = engine.prefix_ends
            self._gate_source = gate_source
        splits = triton.cdiv(engine.capacity, engine.split_size) + 1
        wskey = (hq, n, d, splits)
        if wskey not in self.workspace:
            self.workspace[wskey] = (
                torch.empty((n, hq, splits, d + 2), device=q.device, dtype=torch.float32),
                torch.empty((self.W, hq, d + 2), device=q.device, dtype=torch.float32),
                torch.empty(q.shape, device=q.device, dtype=q.dtype),
            )
        partial, local, out = self.workspace[wskey]
        g = hq // self.hkv
        gqa_config = (engine.gqa_query_block, None) if engine.gqa_query_block else None
        common = (
            q,
            key,
            value,
            ends,
            k,
            v,
            tail,
            partial,
            *q.stride()[1:],
            *key.stride()[1:],
            *value.stride()[1:],
            *k.stride()[1:],
            *v.stride()[1:],
            *tail.stride(),
            ends.stride(0),
        )
        if gqa_config is not None and g > 1:
            query_requested, heads_requested = gqa_config
            heads = min(g, 8) if heads_requested is None else heads_requested
            queries = min(16, 32 // heads) if query_requested is None else query_requested
            if (
                heads not in (1, 2, 4, 8)
                or heads > g
                or queries not in (2, 4, 8, 16, 32)
                or (not 16 <= queries * heads <= 128)
            ):
                raise ValueError("Invalid Shallow KV GQA tile")
            bm = queries * heads
            _gqa_partial[triton.cdiv(n, queries) * triton.cdiv(g, heads), self.hkv, splits](
                *common,
                n,
                hq,
                g,
                d,
                n,
                bm,
                64,
                triton.next_power_of_2(d),
                splits,
                queries,
                heads,
                triton.cdiv(g, heads),
                engine.split_size,
                engine.capacity,
                d ** (-0.5),
                engine.probability_mode == "two_term",
                num_warps=8 if bm >= 128 else 4,
            )
        else:
            _partial[triton.cdiv(n, engine.query_block), hq, splits](
                *common,
                n,
                hq,
                g,
                d,
                n,
                engine.query_block,
                64,
                triton.next_power_of_2(d),
                splits,
                engine.split_size,
                engine.capacity,
                d ** (-0.5),
                engine.probability_mode == "two_term",
                num_warps=4,
            )
        dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[q.dtype]
        fsplits = triton.cdiv(engine.capacity, self.full_split_size)
        if self.full_workspace is None:
            self.full_workspace = (
                torch.empty((self.W, hq, fsplits, d + 2), device=q.device, dtype=torch.float32),
                torch.empty((self.W, hq, d + 2), device=q.device, dtype=torch.float32),
            )
        fulltmp, tailtmp = self.full_workspace
        _gated_full_partial[self.full_blocks,](
            q,
            key,
            value,
            engine.kptr,
            engine.vptr,
            engine.prefix_ends,
            positions,
            safe_depths,
            self.full_mask,
            engine.draft_base,
            fulltmp,
            self.full_rows,
            self.full_count,
            *q.stride()[1:],
            *key.stride()[1:],
            *value.stride()[1:],
            self.W,
            hq,
            self.hkv,
            hq // self.hkv,
            d,
            triton.next_power_of_2(d),
            64,
            max(16, triton.next_power_of_2(hq // self.hkv)),
            self.window,
            self.core,
            engine.draft_core_layer,
            engine.npre,
            engine.draft_phase,
            dtype,
            d ** (-0.5),
            engine.probability_mode == "two_term",
            0,
            fsplits,
            self.full_split_size,
            self.full_blocks,
            num_warps=4,
        )
        bt = 16 if max(self.W, self.fallback_window) <= 16 else 64
        _gated_local_partial[self.W, self.hkv](
            q,
            key,
            value,
            engine.kptr,
            engine.vptr,
            self.keys,
            self.values,
            self.tags,
            k,
            v,
            engine.tail_mask,
            engine.prefix_ends,
            positions,
            safe_depths,
            engine.read_valid,
            engine.draft_base,
            trusted,
            tailtmp,
            self.full_mask,
            *q.stride()[1:],
            *key.stride()[1:],
            *value.stride()[1:],
            *k.stride()[1:],
            *v.stride()[1:],
            *engine.tail_mask.stride(),
            n,
            self.W,
            hq,
            self.hkv,
            hq // self.hkv,
            d,
            triton.next_power_of_2(d),
            bt,
            max(16, triton.next_power_of_2(hq // self.hkv)),
            self.window,
            self.core,
            engine.draft_core_layer,
            engine.npre,
            engine.draft_phase,
            self.capacity,
            dtype,
            d ** (-0.5),
            engine.probability_mode == "two_term",
            False,
            0,
            False,
            self.fallback_window,
            num_warps=4,
        )
        _gated_reduce[self.W, hq](
            fulltmp,
            tailtmp,
            local,
            self.full_mask,
            self.W,
            hq,
            d,
            triton.next_power_of_2(d),
            fsplits,
            triton.next_power_of_2(fsplits),
            num_warps=4,
        )
        _merge[n, hq](
            partial,
            out,
            *out.stride()[1:],
            hq,
            d,
            triton.next_power_of_2(d),
            splits,
            triton.next_power_of_2(splits),
            num_warps=4,
        )
        _merge_local[self.W, hq](
            partial,
            local,
            out,
            *out.stride()[1:],
            self.W,
            hq,
            d,
            triton.next_power_of_2(d),
            splits,
            triton.next_power_of_2(splits),
            num_warps=4,
        )
        return out
