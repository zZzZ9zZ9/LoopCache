import copy
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from ..model import LoopCacheModel


@dataclass
class PreparedBatch:
    cache: object
    logits: torch.Tensor
    lengths: torch.Tensor

    @property
    def capacity(self):
        return self.cache.capacity


class BatchModel(LoopCacheModel):
    def __init__(
        self,
        name,
        checkpoint=None,
        *,
        devices,
        batch_size,
        context_parallel=False,
        context_group=None,
        **kwargs,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.shard_devices = tuple(
            torch.device(f"cuda:{d}" if isinstance(d, int) else d) for d in devices
        )
        if not self.shard_devices or len(set(self.shard_devices)) != len(
            self.shard_devices
        ):
            raise ValueError("Specify distinct CUDA devices")
        if any(device.type != "cuda" for device in self.shard_devices):
            raise ValueError("CUDA devices are required")
        self.max_batch_size = batch_size
        self.batch_size = batch_size
        self.context_parallel = None
        if context_parallel:
            from ..context_parallel import ContextParallel

            if len(devices) != 1:
                raise ValueError("Context parallelism uses one device per process")
            self.context_parallel = ContextParallel(context_group)
        super().__init__(name, checkpoint, device=self.shard_devices[0], **kwargs)
        self.shard_devices = tuple(
            torch.device(self.device if d.index is None else d)
            for d in self.shard_devices
        )
        self.model.requires_grad_(False)
        core = list(
            self.inner.layers if self.family == "ouro" else self.inner.core_block
        )
        prelude = [] if self.family == "ouro" else list(self.inner.prelude)
        coda = [] if self.family == "ouro" else list(self.inner.coda)
        self.core_start = len(prelude)
        depth_blocks = prelude + core * self.total_depth
        slots = list(range(len(depth_blocks))) + [-i - 1 for i in range(len(coda))]
        blocks = depth_blocks + coda
        self.slot_devices = {}
        self.replicas = {}
        self._block_devices = {}
        physical_owners = {}
        if batch_size == 1 and 1 < len(devices) <= len(core):
            physical_owners = {
                id(block): i * len(devices) // len(core) for i, block in enumerate(core)
            }
        self._keep_hidden_local = set(physical_owners) - {id(core[-1])}
        for index, (slot, block) in enumerate(zip(slots, blocks)):
            if physical_owners:
                owner = physical_owners.get(id(block), 0)
            else:
                owner = min(
                    len(devices) - 1,
                    (2 * index + 1) * len(devices) // (2 * len(blocks)),
                )
            device = self.shard_devices[owner]
            self.slot_devices[slot] = device
            key = (id(block), device)
            if key not in self.replicas:
                replica = (
                    block
                    if physical_owners or device == self.device
                    else copy.deepcopy(block)
                )
                self.replicas[key] = replica.to(device).eval()
                self._block_devices.setdefault(id(block), []).append(device)
        self._recurrent_owners = torch.tensor(
            [
                self.shard_devices.index(self.slot_devices[slot])
                for slot in range(len(depth_blocks))
            ],
            device=self.device,
        )
        self._batch_pointer_tables = {}
        self._batch_engine = None
        self._prefill_start = None

    def batch_freqs(self, x, positions):
        if self.family == "huginn":
            return self.model.freqs_cis[0][positions]
        rotary = (
            self.inner.rotary_emb if self.family == "ouro" else self.model.rotary_emb
        )
        return rotary(x, positions)

    def _batch_layer(self, kind, block, x, freqs, cache, slots, positions, valid):
        if isinstance(slots, int):
            groups = [(self.slot_devices[slots], valid)]
        elif len(self._block_devices[id(block)]) == 1:
            groups = [(self._block_devices[id(block)][0], valid)]
        else:
            owners = self._recurrent_owners[torch.where(valid, slots, 0)]
            groups = [
                (d, valid & (owners == self.shard_devices.index(d)))
                for d in self._block_devices[id(block)]
            ]
        result = None
        for dev, active in groups:
            local = self.replicas[id(block), dev]
            data = x.to(dev, non_blocking=True)
            rope = (
                tuple(t.to(dev, non_blocking=True) for t in freqs)
                if isinstance(freqs, tuple)
                else freqs.to(dev, non_blocking=True)
            )
            layer_slots = (
                slots if isinstance(slots, int) else slots.to(dev, non_blocking=True)
            )
            with torch.cuda.device(dev):
                self._layer_valid = active.to(dev, non_blocking=True)
                y = getattr(self, kind)(
                    local,
                    data,
                    rope,
                    cache,
                    layer_slots,
                    positions.to(dev, non_blocking=True),
                    None,
                )
            if id(block) in self._keep_hidden_local:
                return y
            y = y.to(self.device, non_blocking=True)
            if len(groups) > 1:
                y = torch.where(active[..., None], y, 0)
            result = y if result is None else result + y
        return result

    def _attention_many(self, q, k, v, cache, slots, positions, layout=None):
        if self._batch_engine is not None:
            return self._batch_engine.attention(
                q, k, v, cache, slots, positions, self._layer_valid
            )
        start = self._prefill_start
        if self.context_parallel is not None:
            return self.context_parallel.prefill(q, k, v, cache, slots, start)
        cache.position = start
        keys, values = cache.update(k, v, slots)
        length = q.shape[2]
        if start == 0:
            return F.scaled_dot_product_attention(
                q, keys, values, is_causal=True, enable_gqa=True
            )
        mask = causal_lower_right(length, keys.shape[2]) if length > 1 else None
        return F.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask, enable_gqa=True
        )

    def initialize_rows(self, tokens, positions, cache, valid):
        if self.family == "ouro":
            return self.inner.embed_tokens(tokens), None
        hidden = self.inner.wte(tokens) * self.model.emb_scale
        rope = self.batch_freqs(hidden, positions)
        for i, block in enumerate(self.inner.prelude):
            hidden = self._batch_layer(
                "_native_many", block, hidden, rope, cache, i, positions, valid
            )
        return self.state_bank[0][positions], hidden

    def advance_rows(self, x, injected, positions, depth, cache, valid):
        rope = self.batch_freqs(x, positions)
        if self.family == "ouro":
            for i, block in enumerate(self.inner.layers):
                self._core_layer = i
                x = self._batch_layer(
                    "_ouro_many",
                    block,
                    x,
                    rope,
                    cache,
                    depth * self.n_core + i,
                    positions,
                    valid,
                )
            return self.inner.norm(x)
        x = self.inner.adapter(torch.cat((x, injected), dim=-1))
        for i, block in enumerate(self.inner.core_block):
            self._core_layer = i
            x = self._batch_layer(
                "_native_many",
                block,
                x,
                rope,
                cache,
                self.core_start + depth * self.n_core + i,
                positions,
                valid,
            )
        return x

    def finish_rows(self, x, positions, cache, valid):
        if self.family == "ouro":
            return x
        if self.family == "huginn":
            x = self.inner.ln_f(x)
        rope = self.batch_freqs(x, positions)
        for i, block in enumerate(self.inner.coda):
            x = self._batch_layer(
                "_native_many", block, x, rope, cache, -i - 1, positions, valid
            )
        return self.inner.ln_f(x)

    def prepare_batch(self, input_ids, budget, seed, chunk_size):
        rows = [
            torch.as_tensor(ids, dtype=torch.long, device=self.device)
            for ids in input_ids
        ]
        if not 1 <= len(rows) <= self.max_batch_size or any(
            t.ndim != 1 or not t.numel() for t in rows
        ):
            raise ValueError("Expected a nonempty token sequence for each request")
        if budget < 1 or chunk_size < 1:
            raise ValueError(
                "Generation budget and prefill chunk size must be positive"
            )
        self.batch_size = len(rows)
        lengths = torch.tensor([len(t) for t in rows], device=self.device)
        ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        capacity = ids.shape[1] + budget
        cache = self.prepare_request(capacity, seed, True)
        if self.context_parallel is not None:
            cache = self.context_parallel.make_cache(capacity)
        self._batch_pointer_tables.clear()
        last_hidden = None
        try:
            for start in range(0, ids.shape[1], chunk_size):
                self._prefill_start = start
                chunk = ids[:, start : start + chunk_size]
                positions = torch.arange(
                    start, start + chunk.shape[1], device=self.device
                ).expand_as(chunk)
                valid = positions < lengths[:, None]
                x, injected = self.initialize_rows(chunk, positions, cache, valid)
                for depth in range(self.total_depth):
                    x = self.advance_rows(x, injected, positions, depth, cache, valid)
                x = self.finish_rows(x, positions, cache, valid)
                last = (lengths - 1 - start).clamp(0, chunk.shape[1] - 1)
                selected = x[torch.arange(len(rows), device=self.device), last]
                last_hidden = (
                    selected
                    if last_hidden is None
                    else torch.where((lengths > start)[:, None], selected, last_hidden)
                )
        finally:
            self._prefill_start = None
        return PreparedBatch(cache, self.model.lm_head(last_hidden), lengths)


def load_batch_model(
    name,
    checkpoint=None,
    *,
    devices=None,
    batch_size=1,
    context_parallel=False,
    **kwargs,
):
    if devices is None:
        if context_parallel:
            devices = [f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"]
        else:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if isinstance(devices, (int, str, torch.device)):
        devices = [devices]
    devices = tuple(
        torch.device(f"cuda:{d}" if isinstance(d, int) else d) for d in devices
    )
    if torch.device("cuda") in devices:
        current = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
        devices = tuple(
            torch.device("cuda", current) if d == torch.device("cuda") else d
            for d in devices
        )
    if (
        not devices
        or len(set(devices)) != len(devices)
        or any(d.type != "cuda" for d in devices)
    ):
        raise ValueError("Expected one or more distinct CUDA devices")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if context_parallel or len(devices) > 1:
        from ..placement import RequestPlacement

        return RequestPlacement(
            name,
            checkpoint,
            devices=devices,
            batch_size=batch_size,
            context_parallel=context_parallel,
            **kwargs,
        )
    return BatchModel(
        name, checkpoint, devices=devices, batch_size=batch_size, **kwargs
    )
