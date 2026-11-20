import torch
from ..kernels.indexed_attention import indexed_attention_gpu, scatter_kv_from_tables


def _pack(x):
    b, h, n, d = x.shape
    return x.permute(1, 0, 2, 3).reshape(1, h, b * n, d)


def _unpack(x, b, n):
    return x.reshape(x.shape[1], b, n, x.shape[-1]).permute(1, 0, 2, 3)


def pointer_tables(owner, cache, device):
    token = (id(cache), str(device))
    if token in owner._batch_pointer_tables:
        return owner._batch_pointer_tables[token]
    batch = owner.batch_size
    local = [(slot, key) for slot, key in cache.keys.items() if key.device == device]
    if not local:
        raise ValueError("No KV slots on device")
    sample = local[0][1]
    offset = 1 - min(cache.keys)
    count = max(cache.keys) + offset + 1
    key_addresses = [local[0][1].data_ptr()] * count
    value_addresses = [cache.values[local[0][0]].data_ptr()] * count
    for slot, key in local:
        key_addresses[slot + offset] = key.data_ptr()
        value_addresses[slot + offset] = cache.values[slot].data_ptr()
    displacement = (
        torch.arange(batch, device=device) * sample.stride(0) * sample.element_size()
    )
    kp = torch.tensor(key_addresses, device=device)[:, None] + displacement
    vp = torch.tensor(value_addresses, device=device)[:, None] + displacement
    result = (
        kp.flatten(),
        vp.flatten(),
        sample[:1].shape,
        sample[:1].stride(),
        cache.values[local[0][0]][:1].stride(),
        offset,
    )
    owner._batch_pointer_tables[token] = result
    return result


def indexed_batch_attention(owner, q, k, v, cache, slots, positions, valid):
    B, H, N, D = q.shape
    kp, vp, shape, ks, vs, offset = pointer_tables(owner, cache, q.device)
    if isinstance(slots, int):
        slots = torch.full_like(positions, slots)
    bix = torch.arange(B, device=q.device)[:, None]
    indices = ((slots + offset) * B + bix).flatten()
    qp, kpck, vpck = (_pack(q), _pack(k), _pack(v))
    scatter_kv_from_tables(
        kpck,
        vpck,
        kp,
        vp,
        positions.flatten(),
        indices,
        valid.flatten(),
        ks,
        value_stride=vs,
    )
    out = indexed_attention_gpu(
        qp,
        kp,
        vp,
        positions.flatten(),
        indices,
        valid.flatten(),
        shape,
        ks,
        max_length=cache.capacity,
        value_stride=vs,
        split_size=1024,
    )
    return _unpack(out, B, N)


def periodic_attention(
    model,
    cache,
    slot,
    q,
    k,
    v,
    positions,
    write,
    ends,
    mask,
    split_size,
    probability_mode,
):
    from .shared_prefix import shared_prefix_batch

    if model.context_parallel is not None:
        return model.context_parallel.attention(
            q,
            k,
            v,
            cache,
            slot,
            positions,
            write,
            ends,
            mask,
            split_size,
            probability_mode,
        )
    batch, _, lanes, _ = q.shape
    key_ptrs, value_ptrs, _, key_stride, value_stride, offset = pointer_tables(
        model, cache, q.device
    )
    rows = torch.arange(batch, device=q.device)[:, None]
    slots = ((slot + offset) * batch + rows).expand(-1, lanes).contiguous().view(-1)
    scatter_kv_from_tables(
        _pack(k),
        _pack(v),
        key_ptrs,
        value_ptrs,
        positions.flatten(),
        slots,
        write.flatten(),
        key_stride,
        value_stride=value_stride,
    )
    return shared_prefix_batch(
        q,
        cache.keys[slot],
        cache.values[slot],
        ends,
        k,
        v,
        mask,
        split_size,
        probability_mode,
    )
