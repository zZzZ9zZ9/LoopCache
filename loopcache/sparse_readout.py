import torch


def prepare_sparse_readout(engine, width):
    if not isinstance(width, int) or not 2 <= width <= engine.W:
        raise ValueError(f"readout_width must be an integer in [2,{engine.W}]")
    from loopcache.kernels.readout import allocate_selection

    engine.sparse_selection = allocate_selection(engine.lanes, width)
    engine.readout_width = width
    engine.last_readout_hidden = torch.zeros_like(engine.hidden)
    engine.readout_batches = torch.zeros((), device=engine.a.device, dtype=torch.long)
    engine.sparse_readout_stats = torch.zeros(7, device=engine.a.device, dtype=torch.long)
    for name in ["last_readout_hidden", "readout_batches", "sparse_readout_stats"]:
        engine.initial[name] = getattr(engine, name).clone()


def sparse_readout(engine, recurrent, nextdepth, valid):
    a = engine.a
    from loopcache.kernels.readout import select_gpu, finish_gpu

    indices, selected_valid, observed = select_gpu(
        recurrent,
        engine.last_readout_hidden,
        nextdepth,
        valid,
        first_depth=engine.d,
        final_depth=engine.R,
        buffers=engine.sparse_selection,
        stats=engine.sparse_readout_stats,
        batches=engine.readout_batches,
    )
    wave_positions = engine.positions
    engine.positions = wave_positions.index_select(0, indices)
    engine.write_valid = selected_valid
    engine.read_valid = selected_valid
    freqs = (
        engine.rotary.index_select(1, engine.positions)
        if a.family == "huginn"
        else tuple((t.index_select(1, engine.positions) for t in engine.rotary))
    )
    x = recurrent.index_select(1, indices)
    if a.family != "ouro":
        if a.family == "huginn":
            x = a.inner.ln_f(x)
        for i, block in enumerate(a.inner.coda):
            engine.slots = torch.full_like(indices, engine.npre + engine.R * a.n_core + i)
            x = a._native_many(block, x, freqs, engine.cache, None, None)
        x = a.inner.ln_f(x)
    logits = a.model.lm_head(x)[0]
    small_proposals = logits.argmax(-1)
    held = torch.roll(engine.tokens, -1)
    small_held = held.index_select(0, indices)
    small_advantage = (
        logits.gather(1, small_proposals[:, None])[:, 0].float()
        - logits.gather(1, small_held[:, None])[:, 0].float()
    )
    proposals, advantage = finish_gpu(
        small_proposals,
        small_advantage,
        engine.tokens,
        recurrent,
        engine.last_readout_hidden,
        engine.sparse_selection,
    )
    engine.positions = wave_positions
    return (proposals, advantage, observed)
