import time

import torch

from .decoder import CyclicDecoder
from .model import load_batch_model

__all__ = ["load_batch_model", "generate_batch"]


def synchronize(model):
    for dev in model.shard_devices:
        torch.cuda.synchronize(dev)


def capture(model, decoder):
    previous = [torch.cuda.current_stream(dev) for dev in model.shard_devices]
    streams = [torch.cuda.Stream(device=dev) for dev in model.shard_devices]
    try:
        for old, new in zip(previous, streams):
            new.wait_stream(old)
            torch.cuda.set_stream(new)
        torch.cuda.set_device(model.device)
        decoder.cycle()
        decoder.reset()
        synchronize(model)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=streams[0]):
            for stream in streams[1:]:
                stream.wait_stream(streams[0])
            decoder.cycle()
            for stream in streams[1:]:
                streams[0].wait_stream(stream)
        decoder.reset()
        synchronize(model)
        return graph
    finally:
        for stream in previous:
            torch.cuda.set_stream(stream)
        torch.cuda.set_device(model.device)


@torch.inference_mode()
def generate_batch(
    model,
    input_ids,
    max_new_tokens,
    *,
    options=None,
    method="auto",
    profile="tuned",
    confidence_margin=None,
    shallow_layers=None,
    loop_gap=None,
    eos_ids=(),
    seed=1234,
    prefill_chunk=256,
):
    from ..config import resolve_config
    from ..placement import RequestPlacement

    if isinstance(model, RequestPlacement):
        return model.run(
            input_ids,
            max_new_tokens,
            method=method,
            profile=profile,
            options=options,
            confidence_margin=confidence_margin,
            shallow_layers=shallow_layers,
            loop_gap=loop_gap,
            eos_ids=eos_ids,
            seed=seed,
            prefill_chunk=prefill_chunk,
        )
    if not len(input_ids):
        raise ValueError("Expected at least one request")
    config = resolve_config(
        model.name,
        method,
        max(map(len, input_ids)),
        profile,
        confidence_margin=confidence_margin,
        shallow_layers=shallow_layers,
        loop_gap=loop_gap,
    )
    settings = {**config["options"], **(options or {})}
    if (
        settings["poll_interval"] < 1
        or not 1 <= settings["draft_depth"] < model.total_depth
    ):
        raise ValueError(
            "Expected a positive poll interval and a first draft before the final depth"
        )
    start = time.perf_counter()
    with model.attention_context():
        prepared = model.prepare_batch(input_ids, max_new_tokens, seed, prefill_chunk)
    synchronize(model)
    prefilled = time.perf_counter()
    if config["method"] == 1:
        from .dynamic import DynamicDecoder

        scheduler = DynamicDecoder
    else:
        scheduler = CyclicDecoder
    decoder = scheduler(model, prepared, max_new_tokens, settings, eos_ids)
    model._batch_engine = decoder
    try:
        need_decode = not bool(decoder.finished.all())
        graph = capture(model, decoder) if need_decode else None
        begin = time.perf_counter()
        cycles = 0
        while need_decode:
            for _ in range(settings["poll_interval"]):
                graph.replay()
                cycles += 1
            need_decode = not bool(decoder.finished.all())
            if need_decode and cycles > max_new_tokens * model.total_depth:
                raise RuntimeError("Decoder stopped making progress")
        synchronize(model)
        end = time.perf_counter()
        tokens = [
            row[:length]
            for row, length in zip(decoder.output.tolist(), decoder.count.tolist())
        ]
    finally:
        model._batch_engine = None
    decoded = sum(len(row) - 1 for row in tokens)
    return dict(
        method=config["method"],
        token_ids=tokens,
        decode_seconds=end - begin,
        decode_tokens=decoded,
        decode_tokens_per_second=decoded / (end - begin) if decoded else 0,
        prefill_seconds=prefilled - start,
        graph_setup_seconds=begin - prefilled,
        cuda_graph=graph is not None,
    )
