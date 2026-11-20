from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Checkpoint:
    repo_id: str
    family: str
    depth: int
    draft_depth: int
    core_layers: int


MODELS = {
    "ouro-1.4b": Checkpoint("ByteDance/Ouro-1.4B", "ouro", 4, 1, 24),
    "ouro-1.4b-thinking": Checkpoint("ByteDance/Ouro-1.4B-Thinking", "ouro", 4, 1, 24),
    "ouro-2.6b": Checkpoint("ByteDance/Ouro-2.6B", "ouro", 4, 1, 48),
    "ouro-2.6b-thinking": Checkpoint("ByteDance/Ouro-2.6B-Thinking", "ouro", 4, 1, 48),
    "huginn-3.5b": Checkpoint("tomg-group-umd/huginn-0125", "huginn", 32, 4, 4),
    "raven-llama-3.2": Checkpoint("smcleish/Recurrent-Llama-3.2-train-recurrence-32", "raven", 32, 4, 6),
    "raven-olmo-2": Checkpoint("smcleish/Recurrent-OLMo-2-0425-train-recurrence-32", "raven", 32, 4, 6),
    "raven-tinyllama": Checkpoint(
        "smcleish/Recurrent-TinyLlama-3T-train-recurrence-32", "raven", 32, 4, 8
    ),
}


def resolve_config(
    name,
    method="auto",
    length=1,
    profile="tuned",
    *,
    confidence_margin=None,
    shallow_layers=None,
    loop_gap=None,
):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}")
    if profile not in ("tuned", "uniform"):
        raise ValueError("profile must be tuned or uniform")
    if length < 1:
        raise ValueError("Expected a nonempty prompt")
    model = MODELS[name]
    ouro = model.family == "ouro"
    raven = model.family == "raven"
    use_shallow_kv = ouro or profile == "uniform"
    confidence = use_shallow_kv
    if confidence_margin is not None:
        if (
            type(confidence_margin) not in (float, int)
            or not isfinite(confidence_margin)
            or confidence_margin < 0
        ):
            raise ValueError("confidence_margin must be finite and nonnegative")
        confidence = True
    if shallow_layers is not None and (
        type(shallow_layers) is not int or not 0 <= shallow_layers <= model.core_layers
    ):
        raise ValueError(f"shallow_layers must be an integer from 0 to {model.core_layers}")
    if loop_gap is not None and (type(loop_gap) is not int or not 1 <= loop_gap < model.depth):
        raise ValueError(f"loop_gap must be an integer from 1 to {model.depth - 1}")
    if method == "auto":
        method = 1 if length < 8192 else (3 if use_shallow_kv else 2)
    if str(method) not in ("1", "2", "3"):
        raise ValueError("method must be auto, 1, 2 or 3")
    method = int(method)
    if method != 3 and (shallow_layers is not None or loop_gap is not None):
        raise ValueError("shallow_layers and loop_gap require method 3")
    if shallow_layers == 0:
        if loop_gap is not None:
            raise ValueError("loop_gap requires at least one shallow layer")
        method = 2
    if method == 1:
        options = dict(
            draft_depth=1 if raven else model.draft_depth,
            policy="margin" if confidence else ("patience" if raven else "always"),
            margin=confidence_margin
            if confidence_margin is not None
            else (0.125 if confidence else 0.0),
            patience=2,
            poll_interval=4,
        )
        if raven:
            options["readout_width"] = 16 if name == "raven-llama-3.2" else 8
        return dict(method=1, backend="dynamic", options=options)
    options = dict(
        draft_depth=model.draft_depth,
        max_active=5 if ouro else (8 if raven else 16),
        cyclic_policy="confidence" if confidence else "cost",
        margin=confidence_margin
        if confidence_margin is not None
        else (0.125 if profile == "uniform" else 0.0),
        work_penalty=0.0 if confidence else 0.1,
        readout_interval=1 if ouro else (8 if name in ("raven-llama-3.2", "raven-olmo-2") else 4),
        split_size=1024 if ouro else (2048 if name in ("raven-llama-3.2", "raven-tinyllama") else 512),
        draft_full_layers=0,
        draft_full_gate=None,
        sparse_prelude=not ouro,
        poll_interval=1,
    )
    if method == 3:
        options["draft_full_layers"] = (
            shallow_layers if shallow_layers is not None else (2 if use_shallow_kv else 1)
        )
        gate = (0, 1) if ouro and profile == "tuned" else (1, 1)
        if ouro and model.core_layers == 48 and profile == "tuned":
            gate = [(0, 1), (0, 2)] + [(0, 0)] * (model.core_layers - 2)
        if loop_gap is not None:
            gate = (loop_gap, loop_gap)
        elif shallow_layers is not None and profile == "tuned" and ouro:
            if isinstance(gate, list):
                gate = gate[:2] + [(0, 1)] * (model.core_layers - 2)
        options["draft_full_gate"] = gate
    if name in ("raven-llama-3.2", "raven-tinyllama"):
        options["gqa_query_block"] = 8 if name == "raven-llama-3.2" else 4
    return dict(method=method, backend="cyclic", options=options)
