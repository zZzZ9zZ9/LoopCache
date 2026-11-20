import math

import torch

from .attention import indexed_batch_attention
from .decoder import is_eos


class DynamicDecoder:
    def __init__(self, model, prepared, budget, options, eos_ids):
        self.model = model
        self.cache = prepared.cache
        self.prompt_lengths = prepared.lengths
        self.budget = budget
        self.draft_depth = options["draft_depth"]
        self.depth = model.total_depth
        self.width = min(math.ceil(self.depth / self.draft_depth) + 1, max(2, budget))
        self.readout_width = min(options.get("readout_width", self.width), self.width)
        self.policy = options["policy"]
        self.margin = options.get("margin", 0.0)
        self.patience = options.get("patience", 2)
        self.work_penalty = options.get("work_penalty", 0.0)
        batch = model.batch_size
        self.lanes = torch.arange(self.width, device=model.device).expand(batch, -1)
        self.tokens = torch.zeros_like(self.lanes)
        self.tokens[:, 0] = prepared.logits.argmax(-1)
        self.valid = self.lanes == 0
        self.depths = torch.zeros_like(self.tokens)
        self.candidates = torch.full_like(self.tokens, -1)
        self.streaks = torch.zeros_like(self.tokens)
        self.count = torch.ones(batch, device=model.device, dtype=torch.long)
        self.eos = torch.tensor(eos_ids, device=model.device, dtype=torch.long)
        self.finished = (self.count >= budget) | is_eos(self.tokens[:, 0], self.eos)
        self.output = torch.zeros(
            (batch, budget + 1), dtype=torch.long, device=model.device
        )
        self.output[:, 0] = self.tokens[:, 0]
        dim = getattr(model.config, "hidden_size", None) or model.config.n_embd
        self.hidden = torch.zeros(
            (batch, self.width, dim), device=model.device, dtype=model.dtype
        )
        self.injected = torch.zeros_like(self.hidden)
        self.anchor = torch.zeros_like(self.hidden)
        self.initial = {
            key: getattr(self, key).clone()
            for key in (
                "tokens",
                "valid",
                "depths",
                "candidates",
                "streaks",
                "count",
                "finished",
                "output",
                "hidden",
                "injected",
                "anchor",
            )
        }

    def reset(self):
        for key, tensor in self.initial.items():
            getattr(self, key).copy_(tensor)

    def attention(self, q, k, v, cache, slots, positions, valid):
        model = self.model
        if model.context_parallel is None:
            return indexed_batch_attention(
                model, q, k, v, cache, slots, positions, valid
            )
        n = positions.shape[1]
        tail = torch.zeros((*positions.shape, n), device=q.device, dtype=torch.bool)

        def at(slot, selected):
            ends = torch.where(selected, positions, -1)
            return model.context_parallel.attention(
                q, k, v, cache, slot, positions, selected, ends, tail, 1024, "two_term"
            )

        if isinstance(slots, int):
            return at(slots, valid)
        result = torch.zeros_like(q)
        for depth in range(model.total_depth):
            slot = model.core_start + depth * model.n_core + model._core_layer
            selected = valid & (slots == slot)
            result = torch.where(selected[:, None, :, None], at(slot, selected), result)
        return result

    def cycle(self):
        model = self.model
        valid = self.valid & ~self.finished[:, None]
        positions = (
            self.prompt_lengths[:, None] + self.count[:, None] - 1 + self.lanes
        ).clamp(max=self.cache.capacity - 1)
        new = valid & (self.depths == 0)
        if model.family == "ouro":
            initial = model.inner.embed_tokens(self.tokens)
        else:
            indices = (
                torch.where(new, self.lanes, self.width)
                .min(-1)
                .values.clamp(max=self.width - 1)[:, None]
            )
            _, injected = model.initialize_rows(
                self.tokens.gather(1, indices),
                positions.gather(1, indices),
                self.cache,
                new.any(-1, keepdim=True),
            )
            self.injected.copy_(torch.where(new[..., None], injected, self.injected))
            initial = model.state_bank[0][positions]
        recurrent = model.advance_rows(
            torch.where(new[..., None], initial, self.hidden),
            self.injected,
            positions,
            self.depths,
            self.cache,
            valid,
        )
        next_depth = self.depths + valid.long()
        child = self.tokens.roll(-1, 1)
        have_child = valid.roll(-1, 1) & (self.lanes < self.width - 1)
        due = valid & (next_depth >= self.draft_depth)
        if self.readout_width < self.width:
            mandatory = due & (
                (next_depth == self.draft_depth)
                | (next_depth == self.depth)
                | ~have_child
            )
            motion = (recurrent.float() - self.anchor.float()).square().sum(
                -1
            ) / self.anchor.float().square().sum(-1).clamp(min=1e-12)
            scores = torch.where(
                mandatory, torch.inf, torch.where(due, motion, -torch.inf)
            )
            indices = (
                scores.topk(self.readout_width, dim=-1, sorted=False)
                .indices.sort(-1)
                .values
            )
        else:
            indices = self.lanes
        observed = due.gather(1, indices)
        state = recurrent.gather(
            1, indices[..., None].expand(-1, -1, recurrent.shape[-1])
        )
        state = model.finish_rows(
            state, positions.gather(1, indices), self.cache, observed
        )
        logits = model.model.lm_head(state)
        proposal = logits.argmax(-1)
        advantage = (
            logits.gather(-1, proposal[..., None]).squeeze(-1).float()
            - logits.gather(-1, child.gather(1, indices)[..., None]).squeeze(-1).float()
        )
        proposals = child.scatter(1, indices, proposal)
        gains = torch.zeros_like(self.tokens, dtype=torch.float).scatter_(
            1, indices, advantage
        )
        observed = torch.zeros_like(valid).scatter_(1, indices, observed)
        streaks = torch.where(
            observed,
            torch.where(self.candidates == proposals, self.streaks + 1, 1),
            self.streaks,
        )
        if self.policy == "always":
            permit = valid
        elif self.policy == "patience":
            permit = (streaks >= self.patience) & (gains >= self.margin)
        elif self.policy == "cost":
            spent = (next_depth * valid).flip((1,)).cumsum(1).flip(
                (1,)
            ) - next_depth * valid
            permit = gains >= self.margin + self.work_penalty * spent
        else:
            permit = gains >= self.margin
        final = valid[:, 0] & (next_depth[:, 0] == self.depth)
        event = observed & (
            ~have_child | ((child != proposals) & (permit | (next_depth == self.depth)))
        )
        event &= (self.lanes < self.width - 1) & (
            self.count[:, None] + self.lanes < self.budget
        )
        cut = torch.where(event, self.lanes, self.width).min(1).values
        create = (self.lanes == cut[:, None] + 1) & (cut[:, None] < self.width)
        keep = valid & (self.lanes <= cut[:, None])
        token = proposals.gather(1, cut.clamp(max=self.width - 1)[:, None])
        destination = torch.where(final, self.count, self.budget)
        self.output.scatter_(1, destination[:, None], proposals[:, :1])
        self.count.add_(final.long())
        self.finished.logical_or_(
            (final & is_eos(proposals[:, 0], self.eos)) | (self.count >= self.budget)
        )

        def advance(dst, value):
            shifted = value.roll(-1, 1)
            if dst is self.valid:
                shifted &= self.lanes < self.width - 1
            commit = final[:, None, None] if value.ndim == 3 else final[:, None]
            dst.copy_(torch.where(commit, shifted, value))

        advance(self.valid, keep | create)
        advance(self.tokens, torch.where(create, token, self.tokens))
        advance(self.depths, torch.where(create, 0, next_depth))
        advance(
            self.candidates,
            torch.where(create, -1, torch.where(observed, proposals, self.candidates)),
        )
        advance(self.streaks, torch.where(create, 0, streaks))
        advance(self.hidden, torch.where(create[..., None], 0, recurrent))
        advance(self.injected, torch.where(create[..., None], 0, self.injected))
        if self.readout_width < self.width:
            advance(
                self.anchor,
                torch.where(
                    create[..., None],
                    0,
                    torch.where(observed[..., None], recurrent, self.anchor),
                ),
            )
