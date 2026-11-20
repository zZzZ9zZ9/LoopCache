from contextlib import ExitStack
import threading
import time

import torch

from .config import resolve_config
from .cyclic import CyclicDeviceEngine
from .dynamic import DeviceEngine
from .kernels.coda_readout import SharedCodaFinalOriginal
from .kernels.native import patched as fuse_native
from .kernels.ouro import patched as fuse_ouro
from .sparse_readout import prepare_sparse_readout

_decode_lock = threading.RLock()


@torch.inference_mode()
def generate(
    adapter,
    input_ids,
    *,
    method="auto",
    profile="tuned",
    max_new_tokens=128,
    seed=1234,
    eos_ids=(),
    prefill_chunk_size=512,
    confidence_margin=None,
    shallow_layers=None,
    loop_gap=None,
):
    if max_new_tokens < 1 or prefill_chunk_size < 1:
        raise ValueError("The output budget and prefill chunk size must be positive")
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=adapter.device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("Expected one nonempty sequence of token IDs")
    config = resolve_config(
        adapter.name,
        method,
        len(ids),
        profile,
        confidence_margin=confidence_margin,
        shallow_layers=shallow_layers,
        loop_gap=loop_gap,
    )
    options = config["options"].copy()
    poll_interval = options.pop("poll_interval")
    if adapter.device.type != "cuda":
        raise ValueError("CUDA Graph decoding requires a CUDA device")
    with (
        _decode_lock,
        torch.cuda.device(adapter.device),
        adapter.attention_context(),
        ExitStack() as stack,
    ):
        start = time.perf_counter()
        cache = adapter.prepare_request(len(ids) + max_new_tokens, seed)
        first = adapter.prefill(ids, cache, prefill_chunk_size).argmax()
        torch.cuda.synchronize(adapter.device)
        prefill_seconds = time.perf_counter() - start
        if max_new_tokens == 1 or (eos_ids and int(first) in eos_ids):
            return dict(
                token_ids=[int(first)],
                method=config["method"],
                decode_tokens=0,
                prefill_seconds=prefill_seconds,
                graph_setup_seconds=0.0,
                decode_seconds=0.0,
                graph_replays=0,
            )
        dynamic = config["method"] == 1
        if adapter.family == "ouro":
            stack.enter_context(fuse_ouro(adapter, rms_square="torch" if dynamic else "fused"))
        elif dynamic and adapter.family == "raven":
            stack.enter_context(
                SharedCodaFinalOriginal(
                    adapter, query_block=16, key_block=32, split_size=512, probability_mode="two_term"
                )
            )
        elif not dynamic:
            stack.enter_context(fuse_native(adapter, mode="norm_rope"))
        if dynamic:
            width = options.pop("readout_width", None)
            engine = DeviceEngine(
                adapter, cache, len(ids), max_new_tokens, first=first, eos_ids=eos_ids, **options
            )
            if width is not None:
                prepare_sparse_readout(engine, width)
            engine.prepare_fused_scheduler()
        else:
            engine = CyclicDeviceEngine(
                adapter, cache, len(ids), max_new_tokens, first, eos_ids=eos_ids, **options
            )
        capture_start = time.perf_counter()
        engine.capture()
        try:
            torch.cuda.synchronize(adapter.device)
            decode_start = time.perf_counter()
            replays = 0
            limit = max_new_tokens * (adapter.total_depth if dynamic else 1) + poll_interval
            while not bool(engine.finished):
                if replays >= limit:
                    raise RuntimeError("Decoder made no progress")
                for _ in range(poll_interval):
                    engine.graph.replay()
                replays += poll_interval
            torch.cuda.synchronize(adapter.device)
            decode_seconds = time.perf_counter() - decode_start
            if (
                dynamic
                and hasattr(engine, "sparse_readout_stats")
                and int(engine.sparse_readout_stats[6])
            ):
                raise RuntimeError("Insufficient readout slots for verified tokens")
            token_ids = engine.output[: int(engine.count)].tolist()
        finally:
            engine.graph.reset()
    return dict(
        token_ids=token_ids,
        method=config["method"],
        decode_tokens=len(token_ids) - 1,
        prefill_seconds=prefill_seconds,
        graph_setup_seconds=decode_start - capture_start,
        decode_seconds=decode_seconds,
        graph_replays=replays,
    )
