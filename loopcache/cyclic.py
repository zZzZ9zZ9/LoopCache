import torch
from loopcache.kernels.indexed_attention import scatter_kv_from_tables
from loopcache.kernels.shared_attention import shared_prefix_attention


class CyclicDeviceEngine:
    def __init__(
        self,
        a,
        cache,
        prompt_length,
        budget,
        first,
        *,
        draft_depth,
        max_active,
        cyclic_policy,
        margin,
        work_penalty,
        eos_ids,
        split_size,
        readout_interval,
        sparse_prelude,
        draft_full_layers,
        draft_full_gate,
        gqa_query_block=None,
    ):
        self.a, self.cache = (a, cache)
        self.prompt_length, self.budget = (prompt_length, budget)
        self.R, self.W, self.d = (a.total_depth, max_active, draft_depth)
        self.capacity = cache.capacity
        self.policy, self.margin, self.work_penalty = (cyclic_policy, margin, work_penalty)
        self.split_size = split_size
        self.query_block = 16
        self.probability_mode = "single"
        self.sparse_prelude = sparse_prelude
        self.readout_interval = readout_interval
        self.gqa_query_block = gqa_query_block
        dev = a.device
        W = self.W
        self.lanes = torch.arange(W, device=dev, dtype=torch.long)
        self.rows = torch.arange(2 * W, device=dev, dtype=torch.long)
        self.tokens = torch.zeros(W, device=dev, dtype=torch.long)
        self.tokens[0] = first
        self.valid = self.lanes == 0
        self.fresh = self.valid.clone()
        self.spec_depth = torch.zeros(W, device=dev, dtype=torch.long)
        self.kv_active = torch.zeros(W, device=dev, dtype=torch.bool)
        self.merged = torch.zeros(W, device=dev, dtype=torch.bool)
        self.candidates = torch.full_like(self.tokens, -1)
        self.streaks = torch.zeros_like(self.tokens)
        self.count = torch.ones((), device=dev, dtype=torch.long)
        self.output = torch.zeros(budget + 1, device=dev, dtype=torch.long)
        self.output[0] = first
        self.eos = torch.tensor(list(eos_ids), device=dev, dtype=torch.long)
        self.finished = torch.zeros((), device=dev, dtype=torch.bool)
        if eos_ids:
            self.finished.copy_((first == self.eos).any())
        if budget == 1:
            self.finished.fill_(True)
        H = a.config.hidden_size if a.family == "ouro" else a.config.n_embd
        self.kv_hidden = torch.zeros((1, W, H), device=dev, dtype=a.dtype)
        self.spec_hidden = torch.zeros_like(self.kv_hidden)
        self.injected = torch.zeros_like(self.kv_hidden)
        self.stats = torch.zeros(16, device=dev, dtype=torch.long)
        self.npre = 0 if a.family == "ouro" else a.n_pre
        self.ncoda = 0 if a.family == "ouro" else len(a.inner.coda)
        self.slot_keys = list(range(self.npre + self.R * a.n_core)) + [
            -(i + 1) for i in range(self.ncoda)
        ]
        keys = [cache.keys[s] for s in self.slot_keys]
        values = [cache.values[s] for s in self.slot_keys]
        if len({(k.shape, k.stride()) for k in keys}) != 1:
            raise ValueError("Uniform KV shape required")
        self.kptr = torch.tensor([k.data_ptr() for k in keys], device=dev, dtype=torch.long)
        self.vptr = torch.tensor([v.data_ptr() for v in values], device=dev, dtype=torch.long)
        self.kstride, self.vstride = (keys[0].stride(), values[0].stride())
        self.rotary = a._mixed_freqs(
            torch.empty((1, 1, H), device=dev, dtype=a.dtype), list(range(self.capacity))
        )
        self.saved_attention = a._attention_many
        names = (
            "tokens",
            "valid",
            "fresh",
            "spec_depth",
            "kv_active",
            "merged",
            "candidates",
            "streaks",
            "count",
            "output",
            "finished",
            "kv_hidden",
            "spec_hidden",
            "injected",
            "stats",
        )
        self.initial = {name: getattr(self, name).clone() for name in names}
        from loopcache.kernels.attention_mask import Builder
        from loopcache.kernels.cyclic_schedule import allocate
        from loopcache.kernels.shallow_attention import ShallowLayerPool

        self.attention_metadata_builder = Builder()
        self.draft_in_core = False
        self.draft_full_layers = draft_full_layers
        self.draft_pool = ShallowLayerPool(self, draft_full_gate) if draft_full_layers else None
        self.cyclic_schedule_buffers = allocate(self)

    def reset(self):
        for name, value in self.initial.items():
            getattr(self, name).copy_(value)
        if self.draft_pool is not None:
            self.draft_pool.reset()

    def freqs(self, positions):
        return (
            self.rotary.index_select(1, positions)
            if self.a.family == "huginn"
            else tuple((t.index_select(1, positions) for t in self.rotary))
        )

    def set_attention(self, positions, read_valid, write_valid, base_end, primary):
        self.attn_positions = positions
        self.read_valid = read_valid
        self.write_valid = write_valid
        self.prefix_ends, self.tail_mask = self.attention_metadata_builder(
            positions, read_valid, write_valid, base_end, primary
        )

    def attention(self, q, k, v, cache, slots, positions, layout=None):
        scatter_kv_from_tables(
            k,
            v,
            self.kptr,
            self.vptr,
            self.attn_positions,
            self.slots,
            self.write_valid,
            self.kstride,
            value_stride=self.vstride,
        )
        if (
            self.draft_pool is not None
            and self.draft_in_core
            and (self.draft_core_layer < self.draft_full_layers)
        ):
            return self.draft_pool.attention(self, q, k, v, shared_prefix_attention)
        key = self.cache.keys[self.physical_slot]
        value = self.cache.values[self.physical_slot]
        return self.shared_attention(q, key, value, k, v)

    def shared_attention(self, q, key, value, k, v):
        if self.gqa_query_block is not None and q.shape[1] > key.shape[1]:
            from loopcache.kernels.grouped_attention import shared_prefix_attention_gqa

            return shared_prefix_attention_gqa(
                q,
                key,
                value,
                self.prefix_ends,
                k,
                v,
                self.tail_mask,
                max_length=self.capacity,
                split_size=self.split_size,
                probability_mode=self.probability_mode,
                query_block=self.gqa_query_block,
                head_block=min(q.shape[1] // key.shape[1], 8),
            )
        return shared_prefix_attention(
            q,
            key,
            value,
            self.prefix_ends,
            k,
            v,
            self.tail_mask,
            max_length=self.capacity,
            split_size=self.split_size,
            probability_mode=self.probability_mode,
            query_block=self.query_block,
        )

    def slot(self, index, physical, n):
        self.physical_slot = physical
        self.slots = torch.full((n,), index, device=self.a.device, dtype=torch.long)

    def tick(self, phase):
        a = self.a
        W = self.W
        valid = self.valid & ~self.finished
        positions = (self.prompt_length + self.count - 1 + self.lanes).clamp(max=self.capacity - 1)
        new = valid & self.fresh
        if a.family == "ouro":
            initial = a.inner.embed_tokens(self.tokens[None])
        else:
            if not self.sparse_prelude or phase % self.readout_interval == 0:
                ix = torch.where(new, self.lanes, W).min().clamp(max=W - 1).reshape(1)
                pp = positions.index_select(0, ix)
                pv = new.any().reshape(1)
                self.set_attention(pp, pv, pv, pp[0] + 1, pv)
                xx = a.inner.wte(self.tokens.index_select(0, ix)[None]) * a.model.emb_scale
                ff = self.freqs(pp)
                for i, block in enumerate(a.inner.prelude):
                    self.slot(i, i, 1)
                    xx = a._native_many(block, xx, ff, self.cache, None, None)
                self.injected.copy_(torch.where(new[None, :, None], xx, self.injected))
            initial = a.state_bank.index_select(1, positions)
        if phase == 0:
            self.kv_active.copy_(valid)
            self.merged.copy_(valid & (self.spec_depth == 0))
            self.kv_hidden.copy_(initial)
        kvvalid = valid & self.kv_active
        specvalid = valid & ~self.merged & (self.spec_depth < self.R)
        spec = torch.where(new[None, :, None], initial, self.spec_hidden)
        x = torch.cat((self.kv_hidden, spec), dim=1)
        pp = torch.cat((positions, positions))
        ff = self.freqs(pp)
        spec_read = specvalid
        rv = torch.cat((kvvalid, spec_read))
        wv = torch.cat((kvvalid, torch.zeros_like(specvalid)))
        base = self.prompt_length + self.count - 1
        trusted_end = base + kvvalid.long().sum()
        primary = torch.cat((torch.zeros_like(kvvalid), specvalid))
        self.set_attention(pp, rv, wv, trusted_end, primary)
        if self.draft_pool is not None:
            self.draft_depths = torch.cat((torch.full_like(self.spec_depth, phase), self.spec_depth))
            self.draft_base = base
            self.draft_phase = phase
            self.draft_in_core = True
        if a.family == "ouro":
            for i, block in enumerate(a.inner.layers):
                if self.draft_pool is not None:
                    self.draft_core_layer = i
                self.slot(phase * a.n_core + i, phase * a.n_core + i, 2 * W)
                x = a._ouro_many(block, x, ff, self.cache, None, None, None)
            x = a.inner.norm(x)
        else:
            inj = torch.cat((self.injected, self.injected), dim=1)
            x = a.inner.adapter(torch.cat((x, inj), dim=-1))
            for i, block in enumerate(a.inner.core_block):
                if self.draft_pool is not None:
                    self.draft_core_layer = i
                slot = self.npre + phase * a.n_core + i
                self.slot(slot, slot, 2 * W)
                x = a._native_many(block, x, ff, self.cache, None, None)
        self.draft_in_core = False
        recurrent = x
        sd = torch.where(self.merged, phase + 1, self.spec_depth + specvalid.long())
        readout_tick = (phase + 1) % self.readout_interval == 0 or phase + 1 == self.R
        kd = kvvalid & (phase + 1 >= self.d) & readout_tick
        sdue = specvalid & (sd >= self.d) & readout_tick
        childtokens = torch.roll(self.tokens, -1)
        torch.roll(valid, -1) & (self.lanes < W - 1)
        if readout_tick:
            if a.family != "ouro":
                if a.family == "huginn":
                    x = a.inner.ln_f(x)
                coda_spec_read = sdue
                read = torch.cat((kd, coda_spec_read))
                trusted = torch.cat((kd & (phase + 1 == self.R), torch.zeros_like(sdue)))
                coda_end = base + (kvvalid.long().sum() if phase + 1 == self.R else 0)
                primary = torch.cat((kd, sdue & ~kd))
                self.set_attention(pp, read, trusted, coda_end, primary)
                for i, block in enumerate(a.inner.coda):
                    self.slot(self.npre + self.R * a.n_core + i, -(i + 1), 2 * W)
                    x = a._native_many(block, x, ff, self.cache, None, None)
                x = a.inner.ln_f(x)
            logits = a.model.lm_head(x)[0]
            proposal = logits.argmax(-1)
            old = torch.cat((childtokens, childtokens))
            advantage = (
                logits.gather(1, proposal[:, None])[:, 0].float()
                - logits.gather(1, old[:, None])[:, 0].float()
            )
        else:
            proposal = torch.cat((childtokens, childtokens))
            advantage = torch.zeros(2 * W, device=a.device, dtype=torch.float32)
        kp, sp = (proposal[:W], proposal[W:])
        ka, sa = (advantage[:W], advantage[W:])
        have_s = sdue | self.merged & kd
        sp = torch.where(self.merged, kp, sp)
        sa = torch.where(self.merged, ka, sa)
        have_k = kd
        if self.policy == "confidence" and readout_tick:
            top = logits.topk(2, dim=-1).values.float()
            confidence = top[:, 0] - top[:, 1]
            sc = torch.where(self.merged, confidence[:W], confidence[W:])
            take_k = have_k & (~have_s | (confidence[:W] >= sc))
        else:
            take_k = have_k
        chosen = torch.where(take_k, kp, sp)
        adv = torch.where(take_k, ka, sa)
        observed = have_k | have_s
        from loopcache.kernels.cyclic_schedule import transition

        transition(
            self,
            chosen=chosen,
            adv=adv,
            observed=observed,
            kp=kp,
            sd=sd,
            valid=valid,
            kvvalid=kvvalid,
            specvalid=specvalid,
            kd=kd,
            sdue=sdue,
            new=new,
            recurrent=recurrent,
            phase=phase,
        )

    def cycle(self):
        for phase in range(self.R):
            self.tick(phase)

    def capture(self):
        self.a._attention_many = self.attention
        try:
            stream = torch.cuda.Stream(device=self.a.device)
            stream.wait_stream(torch.cuda.current_stream(self.a.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.cycle()
                self.reset()
            torch.cuda.current_stream(self.a.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.cycle()
            self.reset()
        finally:
            self.a._attention_many = self.saved_attention
