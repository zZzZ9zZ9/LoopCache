import math
import torch
from loopcache.kernels.indexed_attention import indexed_attention_gpu, scatter_kv_from_tables


class DeviceEngine:
    def __init__(
        self,
        adapter,
        cache,
        prompt_length,
        max_new_tokens,
        draft_depth,
        first,
        policy="always",
        margin=0.0,
        patience=2,
        work_penalty=0.0,
        eos_ids=(),
    ):
        if getattr(adapter, "_sharded_devices", None):
            raise NotImplementedError(
                "Dynamic decoding requires a single-device KV cache"
            )
        self.a = adapter
        self.cache = cache
        self.prompt_length = prompt_length
        self.budget = max_new_tokens
        self.capacity = cache.capacity
        self.R = adapter.total_depth
        self.d = draft_depth
        self.W = math.ceil(self.R / self.d) + 1
        self.policy = policy
        self.margin = margin
        self.patience = patience
        self.work_penalty = work_penalty
        dev = adapter.device
        self.lanes = torch.arange(self.W, device=dev, dtype=torch.long)
        self.tokens = torch.zeros(self.W, device=dev, dtype=torch.long)
        self.tokens[0] = first
        self.depths = torch.zeros_like(self.tokens)
        self.valid = self.lanes == 0
        self.count = torch.ones((), device=dev, dtype=torch.long)
        self.output = torch.zeros(max_new_tokens + 1, device=dev, dtype=torch.long)
        self.output[0] = first
        self.finished = torch.zeros((), device=dev, dtype=torch.bool)
        self.eos = torch.tensor(list(eos_ids), device=dev, dtype=torch.long)
        if eos_ids:
            self.finished.copy_((first == self.eos).any())
        width = adapter.config.hidden_size if adapter.family == "ouro" else adapter.config.n_embd
        self.hidden = torch.zeros((1, self.W, width), device=dev, dtype=adapter.dtype)
        self.injected = torch.zeros_like(self.hidden)
        self.candidates = torch.full_like(self.tokens, -1)
        self.streaks = torch.zeros_like(self.tokens)
        self.stats = torch.zeros(7, device=dev, dtype=torch.long)
        npre = 0 if adapter.family == "ouro" else adapter.n_pre
        ncoda = 0 if adapter.family == "ouro" else len(adapter.inner.coda)
        self.slot_keys = list(range(npre + self.R * adapter.n_core)) + [-(i + 1) for i in range(ncoda)]
        keys = [cache.keys[s] for s in self.slot_keys]
        vals = [cache.values[s] for s in self.slot_keys]
        if len({(t.shape, t.stride()) for t in keys}) != 1:
            raise ValueError("Graph KV layouts differ")
        self.kptr = torch.tensor([t.data_ptr() for t in keys], device=dev, dtype=torch.long)
        self.vptr = torch.tensor([t.data_ptr() for t in vals], device=dev, dtype=torch.long)
        self.kshape = keys[0].shape
        self.kstride = keys[0].stride()
        self.vstride = vals[0].stride()
        self.npre = npre
        self.ncoda = ncoda
        x = torch.empty((1, 1, width), device=dev, dtype=adapter.dtype)
        self.rotary = adapter._mixed_freqs(x, list(range(self.capacity)))
        self.saved_attention = adapter._attention_many
        self.initial = {
            name: getattr(self, name).clone()
            for name in (
                "tokens",
                "depths",
                "valid",
                "count",
                "output",
                "finished",
                "hidden",
                "injected",
                "candidates",
                "streaks",
                "stats",
            )
        }

    def prepare_fused_scheduler(self):
        from loopcache.kernels.dynamic_schedule import allocate_schedule_buffers

        self.schedule_buffers = allocate_schedule_buffers(
            self.tokens, self.hidden, with_anchor=hasattr(self, "last_readout_hidden")
        )
        self.schedule_observed = torch.ones((), device=self.a.device, dtype=torch.bool)
        self.schedule_advantage = torch.zeros(self.W, device=self.a.device, dtype=torch.float32)

    def finish_fused(
        self, proposals, advantage, nextdepth, valid, recurrent, observed, fixed_depth=False, step_size=1
    ):
        from loopcache.kernels.dynamic_schedule import fused_schedule, commit_schedule

        anchor = getattr(self, "last_readout_hidden", None)
        fused_schedule(
            proposals=proposals,
            advantage=advantage,
            nextdepth=nextdepth,
            valid=valid,
            tokens=self.tokens,
            candidates=self.candidates,
            streaks=self.streaks,
            recurrent=recurrent,
            injected=self.injected,
            count=self.count,
            output=self.output,
            finished=self.finished,
            stats=self.stats,
            observed=observed,
            eos=self.eos,
            out=self.schedule_buffers,
            budget=self.budget,
            prompt_length=self.prompt_length,
            capacity=self.capacity,
            draft_depth=self.d,
            total_depth=self.R,
            policy=self.policy,
            margin=self.margin,
            patience=self.patience,
            work_penalty=self.work_penalty,
            last_readout_hidden=anchor,
            fixed_depth=fixed_depth,
            step_size=step_size,
        )
        commit_schedule(
            self.schedule_buffers,
            tokens=self.tokens,
            depths=self.depths,
            valid=self.valid,
            candidates=self.candidates,
            streaks=self.streaks,
            hidden=self.hidden,
            injected=self.injected,
            last_readout_hidden=anchor,
        )

    def reset(self):
        for k, v in self.initial.items():
            getattr(self, k).copy_(v)

    def attention(self, q, k, v, cache, slots, positions, layout=None):
        scatter_kv_from_tables(
            k,
            v,
            self.kptr,
            self.vptr,
            self.positions,
            self.slots,
            self.write_valid,
            self.kstride,
            value_stride=self.vstride,
        )
        return indexed_attention_gpu(
            q,
            self.kptr,
            self.vptr,
            self.positions,
            self.slots,
            self.read_valid,
            self.kshape,
            self.kstride,
            max_length=self.capacity,
            value_stride=self.vstride,
        )

    def tick(self):
        a = self.a
        valid = self.valid & ~self.finished
        raw_positions = self.prompt_length + self.count - 1 + self.lanes
        valid = valid & (raw_positions < self.capacity)
        self.positions = raw_positions.clamp(max=self.capacity - 1)
        if a.family == "huginn":
            freqs = self.rotary.index_select(1, self.positions)
        else:
            freqs = tuple((t.index_select(1, self.positions) for t in self.rotary))
        new = valid & (self.depths == 0)
        if a.family == "ouro":
            initial = a.inner.embed_tokens(self.tokens[None])
        else:
            new_index = torch.where(new, self.lanes, self.W).min().clamp(max=self.W - 1).reshape(1)
            new_ids = self.tokens.gather(0, new_index)
            wave_positions = self.positions
            self.positions = wave_positions.index_select(0, new_index)
            new_freqs = (
                freqs.index_select(1, new_index)
                if a.family == "huginn"
                else tuple((t.index_select(1, new_index) for t in freqs))
            )
            injected = a.inner.wte(new_ids[None]) * a.model.emb_scale
            self.write_valid = new.any().reshape(1)
            self.read_valid = self.write_valid
            for i, block in enumerate(a.inner.prelude):
                self.slots = torch.full_like(new_index, i)
                injected = a._native_many(block, injected, new_freqs, self.cache, None, None)
            self.positions = wave_positions
            self.injected.copy_(torch.where(new[None, :, None], injected, self.injected))
            initial = a.state_bank.index_select(1, self.positions)
        x = torch.where(new[None, :, None], initial, self.hidden)
        self.write_valid = valid
        self.read_valid = valid
        if a.family == "ouro":
            for i, block in enumerate(a.inner.layers):
                self.slots = self.depths * a.n_core + i
                x = a._ouro_many(block, x, freqs, self.cache, None, None, None)
            x = a.inner.norm(x)
        else:
            x = a.inner.adapter(torch.cat((x, self.injected), dim=-1))
            for i, block in enumerate(a.inner.core_block):
                self.slots = self.npre + self.depths * a.n_core + i
                x = a._native_many(block, x, freqs, self.cache, None, None)
        nextdepth = self.depths + valid.long()
        recurrent = x
        if hasattr(self, "readout_width"):
            from loopcache.sparse_readout import sparse_readout

            proposals, advantage, observed = sparse_readout(self, recurrent, nextdepth, valid)
        else:
            observed = torch.ones((), device=a.device, dtype=torch.bool)
            if a.family != "ouro":
                if a.family == "huginn":
                    x = a.inner.ln_f(x)
                self.write_valid = valid & (nextdepth >= self.d)
                self.read_valid = self.write_valid
                for i, block in enumerate(a.inner.coda):
                    self.slots = torch.full_like(self.depths, self.npre + self.R * a.n_core + i)
                    x = a._native_many(block, x, freqs, self.cache, None, None)
                x = a.inner.ln_f(x)
            logits = a.model.lm_head(x)[0]
        childtokens = torch.roll(self.tokens, -1)
        torch.roll(valid, -1) & (self.lanes < self.W - 1)
        if not hasattr(self, "readout_width"):
            proposals = logits.argmax(-1)
            proposals = torch.where(observed, proposals, childtokens)
            advantage = (
                logits.gather(1, proposals[:, None])[:, 0].float()
                - logits.gather(1, childtokens[:, None])[:, 0].float()
            )
        self.finish_fused(proposals, advantage, nextdepth, valid, recurrent, observed)

    def capture(self):
        self.a._attention_many = self.attention
        try:
            stream = torch.cuda.Stream(device=self.a.device)
            stream.wait_stream(torch.cuda.current_stream(self.a.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.tick()
                self.reset()
            torch.cuda.current_stream(self.a.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.tick()
            self.reset()
        finally:
            self.a._attention_many = self.saved_attention
