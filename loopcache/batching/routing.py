import torch

from .shared_prefix import shared_prefix_batch


class RoutedKV:
    def __init__(self, engine, options):
        self.layers = options["draft_full_layers"]
        self.gate = options.get("draft_full_gate") or (1, 1)
        self.width = engine.width
        self.depth = engine.depth
        self.pools = {}
        for layer in range(self.layers):
            sample = engine.cache.keys[engine.model.core_start + layer]
            b, h, _, d = sample.shape
            size = (engine.depth + 1) * self.width
            keys = torch.empty(
                (b, h, size, d), device=engine.model.device, dtype=sample.dtype
            )
            tags = torch.full((b, size), -1, device=keys.device, dtype=torch.long)
            self.pools[layer] = keys, torch.empty_like(keys), tags

    def reset(self):
        for _, _, tags in self.pools.values():
            tags.fill_(-1)

    def invalidate(self, cut):
        for _, _, tags in self.pools.values():
            tags.masked_fill_(tags >= cut[:, None], -1)

    def apply(self, out, engine, q, k, v, slot):
        model = engine.model
        layer = (slot - model.core_start) % model.n_core
        if layer >= self.layers:
            return out
        gate = self.gate if isinstance(self.gate[0], int) else self.gate[layer]
        ahead, behind = gate
        phase = engine.core_phase
        width = self.width
        keys, values, tags = self.pools[layer]
        device = keys.device
        pos = engine.positions[:, width:]
        spec_valid = engine.spec_valid
        index = torch.where(
            spec_valid,
            engine.spec_depth * width + pos % width,
            self.depth * width + engine.lanes,
        )
        b, heads, _, dim = keys.shape
        expanded = index[:, None, :, None].expand(-1, heads, -1, dim)
        sk = k[:, :, width:].to(device, non_blocking=True)
        sv = v[:, :, width:].to(device, non_blocking=True)
        keys.scatter_(2, expanded, sk)
        values.scatter_(2, expanded, sv)
        tags.scatter_(1, index, torch.where(spec_valid, pos, -1))
        base = engine.prompt_length + engine.count - 1
        positions = base[:, None] + engine.lanes
        ring = positions % width
        trusted = positions < (base + engine.trusted_count)[:, None]
        tk = k[:, :, :width].to(device, non_blocking=True)
        tv = v[:, :, :width].to(device, non_blocking=True)
        fallback_k = torch.where(trusted[:, None, :, None], tk, sk)
        fallback_v = torch.where(trusted[:, None, :, None], tv, sv)
        common = engine.mask[:, width:, width:] | (
            trusted[:, None, :] & (positions[:, None, :] < pos[:, :, None])
        )
        result = out[:, :, width:]
        for depth in range(self.depth):
            if not (
                (ahead and depth - phase >= ahead)
                or (behind and phase - depth >= behind)
            ):
                continue
            selected = spec_valid & (engine.spec_depth == depth)
            owner_slot = model.core_start + depth * model.n_core + layer
            owner = model.slot_devices[owner_slot]
            indices = depth * width + ring
            available = tags.gather(1, indices) == positions
            source = indices[:, None, :, None].expand(-1, heads, -1, dim)
            tail_k = torch.where(
                available[:, None, :, None], keys.gather(2, source), fallback_k
            )
            tail_v = torch.where(
                available[:, None, :, None], values.gather(2, source), fallback_v
            )
            boundary = base + engine.trusted_count if depth <= phase else base
            ends = torch.minimum(pos, boundary[:, None]) - 1
            mask = (
                selected[:, :, None]
                & (positions[:, None, :] > ends[:, :, None])
                & (positions[:, None, :] <= pos[:, :, None])
                & (available[:, None, :] | common)
            )
            ends = torch.where(selected, ends, -1)
            attend = (
                model.context_parallel._attend
                if model.context_parallel is not None
                else shared_prefix_batch
            )
            own = attend(
                q[:, :, width:].to(owner, non_blocking=True),
                engine.cache.keys[owner_slot],
                engine.cache.values[owner_slot],
                ends.to(owner, non_blocking=True),
                tail_k.to(owner, non_blocking=True),
                tail_v.to(owner, non_blocking=True),
                mask.to(owner, non_blocking=True),
                engine.split_size,
                engine.probability_mode,
            )
            result = torch.where(
                selected[:, None, :, None].to(q.device, non_blocking=True),
                own.to(q.device, non_blocking=True),
                result,
            )
        return torch.cat((out[:, :, :width], result), 2)
