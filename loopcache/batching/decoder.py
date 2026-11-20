import torch

from .attention import periodic_attention
from .routing import RoutedKV


def is_eos(tokens, eos):
    return (tokens[..., None] == eos).any(-1)


class CyclicDecoder:
    def __init__(self, model, prepared, budget, options, eos_ids):
        self.model = model
        self.cache = prepared.cache
        self.batch = model.batch_size
        self.depth = model.total_depth
        self.budget = budget
        self.capacity = prepared.capacity
        self.prompt_length = prepared.lengths
        self.width = min(options["max_active"], max(1, 256 // self.batch))
        self.draft_depth = options["draft_depth"]
        self.interval = options["readout_interval"]
        self.policy = options["cyclic_policy"]
        self.margin = options["margin"]
        self.work_penalty = options["work_penalty"]
        self.split_size = options["split_size"]
        self.probability_mode = options.get("probability_mode", "single")
        if self.width < 1 or self.interval < 1 or self.split_size < 1:
            raise ValueError(
                "Window, readout interval, and split size must be positive"
            )
        batch, width = (self.batch, self.width)
        device = model.device
        self.lanes = torch.arange(width, device=device)[None].expand(batch, -1)
        self.tokens = torch.zeros((batch, width), device=device, dtype=torch.long)
        self.tokens[:, 0] = prepared.logits.argmax(-1)
        self.valid = self.lanes == 0
        self.fresh = self.valid.clone()
        self.spec_depth = torch.zeros_like(self.tokens)
        self.trusted_active = torch.zeros_like(self.valid)
        self.merged = torch.zeros_like(self.valid)
        self.count = torch.ones(batch, device=device, dtype=torch.long)
        self.output = torch.zeros((batch, budget + 1), device=device, dtype=torch.long)
        self.output[:, 0] = self.tokens[:, 0]
        self.eos = torch.tensor(eos_ids, device=device, dtype=torch.long)
        self.finished = (self.count >= budget) | is_eos(self.tokens[:, 0], self.eos)
        hidden_size = getattr(model.config, "hidden_size", None) or model.config.n_embd
        self.trusted_hidden = torch.zeros(
            (batch, width, hidden_size), device=device, dtype=model.dtype
        )
        self.spec_hidden = torch.zeros_like(self.trusted_hidden)
        self.injected = torch.zeros_like(self.trusted_hidden)
        self.draft_banks = (
            RoutedKV(self, options) if options.get("draft_full_layers", 0) else None
        )
        self.core_phase = None
        names = (
            "tokens",
            "valid",
            "fresh",
            "spec_depth",
            "trusted_active",
            "merged",
            "count",
            "output",
            "finished",
            "trusted_hidden",
            "spec_hidden",
            "injected",
        )
        self.initial = {name: getattr(self, name).clone() for name in names}

    def reset(self):
        for name, value in self.initial.items():
            getattr(self, name).copy_(value)
        if self.draft_banks is not None:
            self.draft_banks.reset()

    def set_attention(self, positions, read, write, base, primary):
        self.positions = positions
        self.write = write
        self.ends = torch.where(
            read,
            torch.where(
                write, positions, torch.minimum(positions - 1, base[:, None] - 1)
            ),
            -1,
        )
        n = positions.shape[1]
        indices = torch.arange(n, device=positions.device)
        ancestors = (positions[:, None, :] < positions[:, :, None]) & (
            positions[:, None, :] > self.ends[:, :, None]
        )
        self.mask = (
            read[:, :, None]
            & ~write[:, :, None]
            & (
                ancestors & primary[:, None, :]
                | (indices[:, None] == indices[None, :])[None] & read[:, None, :]
            )
        )

    def attention(self, q, k, v, cache, slot, positions, valid):
        device = q.device
        pos = self.positions.to(device, non_blocking=True)
        write = self.write.to(device, non_blocking=True)
        ends = self.ends.to(device, non_blocking=True)
        mask = self.mask.to(device, non_blocking=True)
        out = periodic_attention(
            self.model,
            cache,
            slot,
            q,
            k,
            v,
            pos,
            write,
            ends,
            mask,
            self.split_size,
            self.probability_mode,
        )
        if self.draft_banks is not None and self.core_phase is not None:
            out = self.draft_banks.apply(out, self, q, k, v, slot)
        return out

    def tick(self, phase):
        model = self.model
        batch, width = (self.batch, self.width)
        valid = self.valid & ~self.finished[:, None]
        new = valid & self.fresh
        pos = (
            self.prompt_length[:, None] + self.count[:, None] - 1 + self.lanes
        ).clamp(max=self.capacity - 1)
        if model.family == "ouro":
            initial = model.inner.embed_tokens(self.tokens)
        else:
            index = (
                torch.where(new, self.lanes, width)
                .min(1)
                .values.clamp(max=width - 1)[:, None]
            )
            positions = pos.gather(1, index)
            active = new.any(1, keepdim=True)
            self.set_attention(positions, active, active, positions[:, 0] + 1, active)
            prelude = (
                model.inner.wte(self.tokens.gather(1, index)) * model.model.emb_scale
            )
            rope = model.batch_freqs(prelude, positions)
            for i, block in enumerate(model.inner.prelude):
                prelude = model._batch_layer(
                    "_native_many",
                    block,
                    prelude,
                    rope,
                    self.cache,
                    i,
                    positions,
                    active,
                )
            self.injected.copy_(torch.where(new[..., None], prelude, self.injected))
            initial = model.state_bank[0][pos]
        if phase == 0:
            self.trusted_active.copy_(valid)
            self.merged.copy_(valid & (self.spec_depth == 0))
            self.trusted_hidden.copy_(initial)
        trusted = valid & self.trusted_active
        spec = valid & ~self.merged & (self.spec_depth < self.depth)
        x = torch.cat(
            (
                self.trusted_hidden,
                torch.where(new[..., None], initial, self.spec_hidden),
            ),
            1,
        )
        positions = torch.cat((pos, pos), 1)
        rope = model.batch_freqs(x, positions)
        read_valid = torch.cat((trusted, spec), 1)
        write_valid = torch.cat((trusted, torch.zeros_like(spec)), 1)
        base = self.prompt_length + self.count - 1
        self.set_attention(
            positions,
            read_valid,
            write_valid,
            base + trusted.long().sum(1),
            torch.cat((torch.zeros_like(trusted), spec), 1),
        )
        self.core_phase = phase
        self.spec_valid = spec
        self.trusted_count = trusted.long().sum(1)
        if model.family == "ouro":
            for i, block in enumerate(model.inner.layers):
                x = model._batch_layer(
                    "_ouro_many",
                    block,
                    x,
                    rope,
                    self.cache,
                    phase * model.n_core + i,
                    positions,
                    read_valid,
                )
            x = model.inner.norm(x)
        else:
            x = model.inner.adapter(
                torch.cat((x, torch.cat((self.injected, self.injected), 1)), -1)
            )
            for i, block in enumerate(model.inner.core_block):
                x = model._batch_layer(
                    "_native_many",
                    block,
                    x,
                    rope,
                    self.cache,
                    model.n_pre + phase * model.n_core + i,
                    positions,
                    read_valid,
                )
        self.core_phase = None
        recurrent = x
        spec_depth = torch.where(self.merged, phase + 1, self.spec_depth + spec.long())
        readout = (phase + 1) % self.interval == 0 or phase + 1 == self.depth
        trusted_due = trusted & (phase + 1 >= self.draft_depth) & readout
        spec_due = spec & (spec_depth >= self.draft_depth) & readout
        child = torch.roll(self.tokens, -1, 1)
        child_valid = torch.roll(valid, -1, 1) & (self.lanes < width - 1)
        if readout:
            if model.family != "ouro":
                if model.family == "huginn":
                    x = model.inner.ln_f(x)
                read = torch.cat((trusted_due, spec_due), 1)
                write_coda = torch.cat(
                    (
                        trusted_due & (phase + 1 == self.depth),
                        torch.zeros_like(spec_due),
                    ),
                    1,
                )
                self.set_attention(
                    positions,
                    read,
                    write_coda,
                    base + (trusted.long().sum(1) if phase + 1 == self.depth else 0),
                    torch.cat((trusted_due, spec_due & ~trusted_due), 1),
                )
                for i, block in enumerate(model.inner.coda):
                    x = model._batch_layer(
                        "_native_many",
                        block,
                        x,
                        rope,
                        self.cache,
                        -(i + 1),
                        positions,
                        read,
                    )
                x = model.inner.ln_f(x)
            logits = model.model.lm_head(x)
            proposal = logits.argmax(-1)
            old = torch.cat((child, child), 1)
            advantage = (
                logits.gather(-1, proposal[..., None])[..., 0].float()
                - logits.gather(-1, old[..., None])[..., 0].float()
            )
        else:
            proposal = torch.cat((child, child), 1)
            advantage = torch.zeros_like(proposal, dtype=torch.float)
        trusted_prediction, spec_prediction = (proposal[:, :width], proposal[:, width:])
        trusted_advantage, spec_advantage = (advantage[:, :width], advantage[:, width:])
        have_spec = spec_due | self.merged & trusted_due
        spec_prediction = torch.where(self.merged, trusted_prediction, spec_prediction)
        spec_advantage = torch.where(self.merged, trusted_advantage, spec_advantage)
        if self.policy == "spec":
            use_trusted = trusted_due & ~have_spec
        elif self.policy in ("depth", "depth_cost"):
            use_trusted = trusted_due & (~have_spec | (phase + 1 >= spec_depth))
        elif self.policy in ("confidence", "confidence_cost") and readout:
            top = logits.topk(2, -1).values.float()
            conf = top[..., 0] - top[..., 1]
            spec_confidence = torch.where(self.merged, conf[:, :width], conf[:, width:])
            use_trusted = trusted_due & (
                ~have_spec | (conf[:, :width] >= spec_confidence)
            )
        else:
            use_trusted = trusted_due
        chosen = torch.where(use_trusted, trusted_prediction, spec_prediction)
        advantage_chosen = torch.where(use_trusted, trusted_advantage, spec_advantage)
        observed = trusted_due | have_spec
        work = trusted.long() * (phase + 1) + torch.where(
            self.merged, 0, spec_depth * valid.long()
        )
        spent = torch.flip(torch.cumsum(torch.flip(work, [1]), 1), [1]) - work
        permit = advantage_chosen >= self.margin + (
            self.work_penalty * spent
            if self.policy in ("cost", "confidence_cost", "depth_cost")
            else 0
        )
        final = trusted & (phase + 1 == self.depth)
        chosen = torch.where(final, trusted_prediction, chosen)
        event = valid & observed & (~child_valid | (chosen != child) & (permit | final))
        event = (
            event
            & (self.lanes < width - 1)
            & (self.count[:, None] + self.lanes < self.budget)
        )
        cut = torch.where(event, self.lanes, width).min(1).values
        if self.draft_banks is not None:
            self.draft_banks.invalidate(base + cut + 1)
        create = (self.lanes == cut[:, None] + 1) & (cut[:, None] < width)
        keep = valid & (self.lanes <= cut[:, None])
        replacement = chosen.gather(1, cut.clamp(max=width - 1)[:, None])
        tokens = torch.where(create, replacement, self.tokens)
        next_valid = keep | create
        next_fresh = create
        next_spec_depth = torch.where(create, 0, spec_depth)
        next_trusted = trusted & keep
        next_merged = self.merged & keep
        next_spec_hidden = torch.where(create[..., None], 0, recurrent[:, width:])
        next_trusted_hidden = torch.where(create[..., None], 0, recurrent[:, :width])
        next_injected = torch.where(create[..., None], 0, self.injected)
        commits = torch.zeros(batch, device=model.device, dtype=torch.long)
        if phase + 1 == self.depth:
            bad = final & (~child_valid | (child != trusted_prediction))
            first_bad = torch.where(bad, self.lanes, width).min(1).values
            commits = torch.minimum(final.long().sum(1), first_bad + 1)
            commits = torch.minimum(commits, self.budget - self.count)
            stop = (self.lanes < commits[:, None]) & is_eos(
                trusted_prediction, self.eos
            )
            commits = torch.minimum(
                commits, torch.where(stop, self.lanes + 1, width + 1).min(1).values
            )
            self.finished.logical_or_(stop.any(1))
            write = self.lanes < commits[:, None]
            indexes = torch.where(write, self.count[:, None] + self.lanes, self.budget)
            self.output.scatter_(1, indexes, torch.where(write, trusted_prediction, 0))
            gather = (self.lanes + commits[:, None]).clamp(max=width - 1)
            inrange = self.lanes + commits[:, None] < width
            tokens = tokens.gather(1, gather)
            next_valid = next_valid.gather(1, gather) & inrange
            next_fresh = next_fresh.gather(1, gather) & inrange
            next_spec_depth = next_spec_depth.gather(1, gather)
            next_trusted = torch.zeros_like(next_trusted)
            next_merged = next_merged.gather(1, gather) & inrange
            hidden_indices = gather[..., None].expand(
                -1, -1, next_spec_hidden.shape[-1]
            )
            next_spec_hidden = next_spec_hidden.gather(1, hidden_indices)
            next_trusted_hidden = next_trusted_hidden.gather(1, hidden_indices)
            next_injected = next_injected.gather(1, hidden_indices)
            root = (self.lanes == 0) & ((commits > 0) & ~next_valid[:, 0])[:, None]
            bonus = trusted_prediction.gather(
                1, (commits - 1).clamp(min=0, max=width - 1)[:, None]
            )
            tokens = torch.where(root, bonus, tokens)
            next_valid = next_valid | root
            next_fresh = next_fresh | root
            next_spec_depth = torch.where(root, 0, next_spec_depth)
            next_merged = next_merged & ~root
            next_spec_hidden = torch.where(root[..., None], 0, next_spec_hidden)
            next_trusted_hidden = torch.where(root[..., None], 0, next_trusted_hidden)
            next_injected = torch.where(root[..., None], 0, next_injected)
            self.count.add_(commits)
            self.finished.logical_or_(self.count >= self.budget)
        self.tokens.copy_(tokens)
        self.valid.copy_(next_valid)
        self.fresh.copy_(next_fresh)
        self.spec_depth.copy_(next_spec_depth)
        self.trusted_active.copy_(next_trusted)
        self.merged.copy_(next_merged)
        self.spec_hidden.copy_(next_spec_hidden)
        self.trusted_hidden.copy_(next_trusted_hidden)
        self.injected.copy_(next_injected)

    def cycle(self):
        for phase in range(self.depth):
            self.tick(phase)
