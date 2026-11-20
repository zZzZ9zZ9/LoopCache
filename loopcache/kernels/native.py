import ast
import contextlib
import hashlib
import inspect
import sys
import textwrap
import types
import torch
from loopcache.kernels.ouro import rms_tail, rope_pair
from loopcache.kernels.native_rope import complex_rope_pair, mixed_split_rope_pair

NORM_AST_HASHES = ("9b58c3285e739945fb161fc161a49dccac6292193269170b6cda046bee0b26d8",)
ROPE_AST_HASHES = {
    "llama": ("680d0dd1e33e17522b976c3fabb35cb8bfd7875dd8a49c6b7d79ee5f531494f3",),
    "olmo": ("f47e56c722c7d510b03ebd0307b0a540647c005ee5a42160df924bf19bdf45cb",),
    "huginn": ("8b2b56a8387a6180255e0a05cd6a5e25db9fa117ef2c172af3e8b2e1e74077d7",),
}
NORM_METHOD_AST_HASHES = {
    "_norm": "5ed575597bb44c204e6a36affa42b1b39f67821a6843fca16434391d1cba5063",
    "forward": "ce507e0ef70e3eadd093131462f08ff3b024a6c9f427560ec241f8b97f37668c",
}
MODES = ("none", "norm", "rope", "norm_rope")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _fingerprint(obj):
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        raise ValueError("Native fusion requires auditable source definitions") from exc
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if hasattr(node, "type_params"):
            if node.type_params:
                raise ValueError("Parameterized native source is outside the validated contract")
            node._fields = tuple((k for k in node._fields if k != "type_params"))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _supported_norm(module):
    cls = type(module)
    if _fingerprint(cls) not in NORM_AST_HASHES:
        return False
    for name, expected in NORM_METHOD_AST_HASHES.items():
        method = getattr(module, name, None)
        if not isinstance(method, types.MethodType) or method.__self__ is not module:
            return False
        function = method.__func__
        if function is not cls.__dict__.get(name) or hasattr(function, "__wrapped__"):
            return False
        if _fingerprint(function) != expected:
            return False
    return True


def validate(adapter, mode):
    if mode not in MODES:
        raise ValueError("fused_native must be none, norm, rope, or norm_rope")
    if mode == "none":
        return
    if adapter.family not in ("huginn", "raven"):
        raise ValueError("fused_native requires a Huginn/Raven adapter")
    if adapter.dtype not in DTYPES:
        raise ValueError("Native fusion supports BF16, FP16, or FP32")
    if getattr(adapter, "_loopcache_native_fusion_active", False):
        raise ValueError("Native fusion context is already active")
    if getattr(adapter, "_sharded_devices", None):
        raise ValueError("Native fusion currently supports the single-device cyclic decoder")
    if "norm" in mode:
        norms = [x for x in adapter.model.modules() if type(x).__name__ == "RMSNorm"]
        if not norms:
            raise ValueError("No supported native RMSNorm modules")
        if any((not _supported_norm(x) for x in norms)):
            raise ValueError("Unsupported RMSNorm source or callable override")
    if "rope" in mode:
        module = sys.modules[type(adapter.model).__module__]
        name = "apply_rotary_emb_complex_like" if adapter.family == "huginn" else "apply_rotary_pos_emb"
        kind = (
            "huginn"
            if adapter.family == "huginn"
            else "olmo"
            if hasattr(adapter.inner.core_block[0].attn, "q_norm")
            else "llama"
        )
        if _fingerprint(getattr(module, name)) not in ROPE_AST_HASHES[kind]:
            raise ValueError("Unsupported native rotary source definition")


@contextlib.contextmanager
def patched(adapter, *, mode="norm_rope"):
    validate(adapter, mode)
    if mode == "none":
        yield
        return
    saved = []
    rotary = None
    had_marker = "_loopcache_native_fusion_active" in adapter.__dict__
    marker = adapter.__dict__.get("_loopcache_native_fusion_active")
    adapter._loopcache_native_fusion_active = True
    try:
        if "norm" in mode:
            for module in adapter.model.modules():
                if type(module).__name__ != "RMSNorm":
                    continue
                old = module.forward
                proxy = types.SimpleNamespace(weight=module.weight, variance_epsilon=float(module.eps))
                had_override = "forward" in module.__dict__
                override = module.__dict__.get("forward")

                def forward(this, x, _old=old, _proxy=proxy):
                    weight = this.weight
                    if (
                        x.device.type != "cuda"
                        or x.ndim != 3
                        or x.shape[0] != 1
                        or (x.stride(-1) != 1)
                        or (x.dtype not in DTYPES)
                        or (x.dtype != weight.dtype)
                        or (x.device != weight.device)
                        or (weight.ndim != 1)
                        or (weight.numel() != x.shape[-1])
                        or (weight.stride(0) != 1)
                    ):
                        return _old(x)
                    return rms_tail(x, _proxy)

                saved.append((module, had_override, override))
                module.forward = types.MethodType(forward, module)
        if "rope" in mode:
            module = sys.modules[type(adapter.model).__module__]
            name = (
                "apply_rotary_emb_complex_like" if adapter.family == "huginn" else "apply_rotary_pos_emb"
            )
            old = getattr(module, name)
            rotary = (module, name, old)
            if adapter.family == "huginn":

                def apply(q, k, freqs_cis):
                    return complex_rope_pair(q, k, freqs_cis)
            else:
                olmo = hasattr(adapter.inner.core_block[0].attn, "q_norm")

                def apply(q, k, cos, sin, *args, **kwargs):
                    if args or kwargs:
                        return old(q, k, cos, sin, *args, **kwargs)
                    if q.device != k.device or q.device != cos.device or q.device != sin.device:
                        raise ValueError("rotary device mismatch")
                    return mixed_split_rope_pair(q, k, cos, sin) if olmo else rope_pair(q, k, cos, sin)

            setattr(module, name, apply)
        yield
    finally:
        if rotary is not None:
            setattr(rotary[0], rotary[1], rotary[2])
        for module, had_override, override in reversed(saved):
            if had_override:
                module.forward = override
            else:
                delattr(module, "forward")
        if had_marker:
            adapter._loopcache_native_fusion_active = marker
        else:
            delattr(adapter, "_loopcache_native_fusion_active")
