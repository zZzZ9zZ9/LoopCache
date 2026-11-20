import copy
import math
import os
import torch
from torch import nn


def yarn_parameters(
    dim, base, original_length, target_length, device=None, beta_fast=32.0, beta_slow=1.0
):
    if dim <= 0 or dim % 2 or base <= 1 or (original_length < 1):
        raise ValueError("YaRN requires a positive even rotary dim, base > 1 and positive context")
    if target_length < original_length or not beta_fast > beta_slow > 0:
        raise ValueError("target context must cover original context; beta_fast > beta_slow > 0")
    factor = float(target_length) / original_length
    frequencies = base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)

    def correction(rotations):
        return dim * math.log(original_length / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (high - low)).clamp(
        0, 1
    )
    extrapolation = 1 - ramp
    inv = 1 / (factor * frequencies) * (1 - extrapolation) + 1 / frequencies * extrapolation
    amplitude = 1.0 if factor <= 1 else 1.0 + 0.1 * math.log(factor)
    return (inv, amplitude)


class YarnRotaryEmbedding(nn.Module):
    def __init__(self, inv_freq, amplitude):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = amplitude
        self.rope_type = "yarn"

    @torch.no_grad()
    def forward(self, x, position_ids):
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return (cos.to(x.dtype), sin.to(x.dtype))


def yarn_complex_table(inv_freq, amplitude, capacity):
    device_type = inv_freq.device.type if inv_freq.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        phases = torch.outer(
            torch.arange(capacity, dtype=torch.float32, device=inv_freq.device), inv_freq.float()
        )
        return torch.stack(
            (phases.cos()[None, :, None, :] * amplitude, phases.sin()[None, :, None, :] * amplitude),
            dim=4,
        )


class RequestContext:
    def __init__(self, adapter, mode=None, target_length=None):
        self.mode = mode if mode is not None else os.environ.get("LOOP_CONTEXT_EXTENSION", "none")
        if self.mode not in ("none", "yarn"):
            raise ValueError("context_extension must be 'none' or 'yarn'")
        env_target = os.environ.get("LOOP_CONTEXT_TARGET")
        self.target_length = (
            target_length if target_length is not None else int(env_target) if env_target else None
        )
        if self.target_length is not None and (
            type(self.target_length) is not int or self.target_length < 1
        ):
            raise ValueError("context_target must be a positive integer")
        self.original_length = int(adapter.context_limit)
        self.original_scaling = copy.deepcopy(getattr(adapter.config, "rope_scaling", None))
        self.rotary_owner = adapter.inner if adapter.family == "ouro" else adapter.model
        self.native = (
            adapter.model.freqs_cis if adapter.family == "huginn" else self.rotary_owner.rotary_emb
        )
        self.active_key = None

    def prepare(self, adapter, capacity, allow_context_extension=False):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        extended = capacity > self.original_length
        if extended and self.mode != "yarn" and (not allow_context_extension):
            raise ValueError(
                f"Requested {capacity} positions exceeds configured context {self.original_length}; select context_extension='yarn'"
            )
        target = (
            max(capacity, self.original_length) if self.target_length is None else self.target_length
        )
        if self.mode == "yarn" and extended and (target < capacity):
            raise ValueError(f"context_target={target} does not cover request capacity={capacity}")
        enabled = self.mode == "yarn" and extended
        key = (
            enabled,
            target if enabled else None,
            capacity if adapter.family == "huginn" and enabled else None,
        )
        if key != self.active_key:
            if enabled:
                config = adapter.config
                dim = getattr(config, "head_dim", None)
                if dim is None:
                    dim = (
                        config.hidden_size // config.num_attention_heads
                        if adapter.family == "ouro"
                        else config.n_embd // config.num_attention_heads
                    )
                dim = int(dim * getattr(config, "partial_rotary_factor", 1.0))
                base = float(config.rope_base if adapter.family == "huginn" else config.rope_theta)
                inv, amplitude = yarn_parameters(dim, base, self.original_length, target, adapter.device)
                if adapter.family == "huginn":
                    adapter.model.freqs_cis = yarn_complex_table(inv, amplitude, capacity)
                else:
                    self.rotary_owner.rotary_emb = YarnRotaryEmbedding(inv, amplitude)
            elif adapter.family == "huginn":
                adapter.model.freqs_cis = self.native
            else:
                self.rotary_owner.rotary_emb = self.native
            self.active_key = key
        factor = target / self.original_length if enabled else 1.0
        adapter.context_metadata = dict(
            mode=self.mode,
            enabled=enabled,
            original_length=self.original_length,
            capacity=capacity,
            target_length=target if enabled else self.original_length,
            factor=factor,
            attention_factor=1.0 + 0.1 * math.log(factor) if enabled else None,
            beta_fast=32.0 if enabled else None,
            beta_slow=1.0 if enabled else None,
            original_rope_scaling=copy.deepcopy(self.original_scaling),
            existing_scaling_replaced=bool(enabled and self.original_scaling),
            fixed_for_entire_request=True,
        )
        return adapter.context_metadata
