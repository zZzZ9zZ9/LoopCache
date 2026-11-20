import torch


class StaticKV:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.keys, self.values, self.lengths = ({}, {}, {})
        self.position = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None, **kwargs):
        slot = int(layer_idx)
        start = self.position
        end = start + key_states.shape[-2]
        if start < 0 or end > self.capacity:
            raise ValueError(f"KV range [{start}, {end}) exceeds capacity {self.capacity}")
        old_len = self.lengths.get(slot, 0)
        if start > old_len:
            raise RuntimeError(f"Uninitialized KV gap: slot={slot}, start={start}, valid={old_len}")
        if slot not in self.keys:
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self.keys[slot] = torch.empty(shape, dtype=key_states.dtype, device=key_states.device)
            self.values[slot] = torch.empty_like(self.keys[slot])
        self.keys[slot][..., start:end, :].copy_(key_states)
        self.values[slot][..., start:end, :].copy_(value_states)
        self.lengths[slot] = max(old_len, end)
        return (self.keys[slot][..., :end, :], self.values[slot][..., :end, :])

    def crop(self, length):
        if length < 0:
            raise ValueError("negative cache length")
        for slot in self.lengths:
            self.lengths[slot] = min(self.lengths[slot], length)

    def get_seq_length(self, layer_idx=0):
        return self.lengths.get(int(layer_idx), 0)

    @property
    def allocated_bytes(self):
        return sum((x.numel() * x.element_size() for x in (*self.keys.values(), *self.values.values())))
