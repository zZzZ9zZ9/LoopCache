import torch
from loopcache.kernels.coda_attention import SharedCodaAttention
from loopcache.kernels.indexed_attention import indexed_attention_gpu


def final_readout_mask(depths, read_valid, *, total_depth, step_size=1, selected_indices=None):
    current = depths if selected_indices is None else depths.index_select(0, selected_indices)
    return read_valid & (current + step_size == total_depth)


def overwrite_final(
    q,
    shared_out,
    positions,
    slots,
    read_valid,
    final_mask,
    key_ptrs,
    value_ptrs,
    kv_shape,
    kv_stride,
    *,
    max_length,
    value_stride=None,
):
    rows = torch.arange(q.shape[2], device=q.device, dtype=torch.long)
    index = torch.where(final_mask, rows, q.shape[2]).min().clamp(max=q.shape[2] - 1).reshape(1)
    one_valid = final_mask.any().reshape(1)
    exact = indexed_attention_gpu(
        q.index_select(2, index),
        key_ptrs,
        value_ptrs,
        positions.index_select(0, index),
        slots.index_select(0, index),
        one_valid,
        kv_shape,
        kv_stride,
        max_length=max_length,
        value_stride=value_stride,
    )
    return torch.where(final_mask[None, None, :, None], exact, shared_out)


class SharedCodaFinalOriginal(SharedCodaAttention):
    def __init__(self, adapter, **kwargs):
        kwargs.setdefault("probability_mode", "two_term")
        super().__init__(adapter, **kwargs)

    def __enter__(self):
        super().__enter__()
        shared_method = self.attention
        ctx = self

        def attention(engine, q, k, v, cache, slots, positions, layout=None):
            shared = shared_method(engine, q, k, v, cache, slots, positions, layout)
            if (
                engine.a is not ctx.a
                or ctx.active_coda is None
                or q.dtype not in (torch.bfloat16, torch.float16)
                or (engine.capacity < ctx.min_length)
            ):
                return shared
            step = 1
            selected = engine.sparse_selection.indices if hasattr(engine, "readout_width") else None
            final = final_readout_mask(
                engine.depths,
                engine.read_valid,
                total_depth=engine.R,
                step_size=step,
                selected_indices=selected,
            )
            if final.numel() != q.shape[2]:
                raise RuntimeError("Final readout metadata/query shape mismatch")
            return overwrite_final(
                q,
                shared,
                engine.positions,
                engine.slots,
                engine.read_valid,
                final,
                engine.kptr,
                engine.vptr,
                engine.kshape,
                engine.kstride,
                max_length=engine.capacity,
                value_stride=engine.vstride,
            )

        self.attention = attention
        self.cls.attention = attention
        return self
