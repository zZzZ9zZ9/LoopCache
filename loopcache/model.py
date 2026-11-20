from dataclasses import dataclass
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers import AutoModelForCausalLM, AutoTokenizer
from .cache import StaticKV
from .config import MODELS
from .context import RequestContext


@dataclass
class TokenStates:
    hidden: torch.Tensor
    injected: torch.Tensor | None
    position: int
    depth: int = 0

    @property
    def count(self):
        return self.hidden.shape[1]


class LoopCacheModel:
    def __init__(
        self,
        name,
        model_path=None,
        device="cuda:0",
        dtype=torch.bfloat16,
        total_depth=None,
        attention="flash",
        context_extension="yarn",
        context_target=None,
    ):
        checkpoint = MODELS[name]
        self.family, natural_total, self.default_draft = (
            checkpoint.family,
            checkpoint.depth,
            checkpoint.draft_depth,
        )
        path = model_path or checkpoint.repo_id
        self.state_init = "request-bank"
        self.name, self.path = (name, str(path))
        self.total_depth = natural_total if total_depth is None else total_depth
        if self.total_depth < 1:
            raise ValueError("total depth must be positive")
        self.natural_depth = natural_total
        self.device, self.dtype, self.attention = (torch.device(device), dtype, attention)
        if attention == "flash" and (self.device.type != "cuda" or dtype == torch.float32):
            raise ValueError("Flash SDPA requires CUDA and FP16 or BF16")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        self.config = self.model.config
        expected_type = "ouro" if self.family == "ouro" else "huginn_raven"
        if self.config.model_type != expected_type:
            raise ValueError(f"{name} requires a {expected_type} checkpoint")
        if getattr(self.config, "test_time_noise", 0):
            raise ValueError("Additional recurrent noise is not supported")
        if self.family == "ouro":
            self.inner = self.model.model
            self.inner.total_ut_steps = self.total_depth
            self.n_core = len(self.inner.layers)
            self.context_limit = self.config.max_position_embeddings
        else:
            self.inner = self.model.transformer
            self.n_core = len(self.inner.core_block)
            self.n_pre = len(self.inner.prelude)
            self.context_limit = getattr(self.config, "max_position_embeddings", self.config.block_size)
        self.state_bank = None
        self._request_context = RequestContext(self, context_extension, context_target)
        self.context_metadata = None

    def prepare_context(self, capacity, allow_context_extension=False):
        return self._request_context.prepare(self, capacity, allow_context_extension)

    def attention_context(self):
        if self.attention == "auto":
            return sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION if self.attention == "flash" else SDPBackend.MATH)

    def prepare_request(self, capacity, seed, allow_context_extension=False):
        self.prepare_context(capacity, allow_context_extension)
        self.state_bank = None
        if self.family != "ouro":
            g = torch.Generator(device=self.device).manual_seed(seed)
            bank = torch.empty(
                (1, capacity, self.config.n_embd), device=self.device, dtype=torch.float32
            )
            std = float(self.config.init_values["std"])
            if std > 0:
                torch.nn.init.trunc_normal_(bank, mean=0, std=std, a=-3 * std, b=3 * std, generator=g)
                bank.mul_(float(self.model.emb_scale))
            else:
                bank.zero_()
            self.state_bank = bank.to(self.dtype)
        if self.family == "huginn" and capacity > self.model.freqs_cis.shape[1]:
            module = sys.modules[type(self.model).__module__]
            self.model.freqs_cis = module.precompute_freqs_cis(
                self.config.n_embd // self.config.num_attention_heads, capacity, self.config.rope_base, 1
            ).to(self.device)
        return StaticKV(capacity)

    def _freqs(self, x, position):
        count = x.shape[1]
        if self.family == "huginn":
            return self.model.freqs_cis[:, position : position + count]
        positions = torch.arange(position, position + count, device=self.device).unsqueeze(0)
        return self.model.rotary_emb(x, positions)

    def _native_block(self, block, x, freqs, cache, slot, position):
        cache.position = position
        return block(x, freqs, slot, past_key_values=cache)

    def initialize(self, ids, position, cache):
        ids = ids.to(self.device).reshape(1, -1)
        if self.family == "ouro":
            return TokenStates(self.inner.embed_tokens(ids), None, position)
        x = self.inner.wte(ids) * self.model.emb_scale
        freqs = self._freqs(x, position)
        for i, block in enumerate(self.inner.prelude):
            x = self._native_block(block, x, freqs, cache, i, position)
        if self.state_bank is None:
            raise RuntimeError("prepare_request must precede initialize")
        h = self.state_bank[:, position : position + x.shape[1]].clone()
        return TokenStates(h, x, position)

    def _ouro_block(self, layer, x, rope, cache, slot, position):
        residual = x
        h = layer.input_layernorm(x)
        attn = layer.self_attn
        b, n, _ = h.shape
        d = attn.head_dim
        q = attn.q_proj(h).view(b, n, self.config.num_attention_heads, d).transpose(1, 2)
        k = attn.k_proj(h).view(b, n, self.config.num_key_value_heads, d).transpose(1, 2)
        v = attn.v_proj(h).view(b, n, self.config.num_key_value_heads, d).transpose(1, 2)
        cos, sin = (t.unsqueeze(1) for t in rope)

        def rotate(t):
            a, b = t.chunk(2, dim=-1)
            return torch.cat((-b, a), dim=-1)

        q, k = (q * cos + rotate(q) * sin, k * cos + rotate(k) * sin)
        cache.position = position
        k, v = cache.update(k, v, slot)
        if n == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        elif position == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, causal_lower_right(n, k.shape[-2]), enable_gqa=True
            )
        y = y.transpose(1, 2).reshape(b, n, -1)
        x = residual + layer.input_layernorm_2(attn.o_proj(y))
        return x + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))

    def advance(self, state, cache):
        if state.depth >= self.total_depth:
            raise ValueError("state is already at target depth")
        x, position = (state.hidden, state.position)
        if self.family == "ouro":
            positions = torch.arange(position, position + state.count, device=self.device)[None]
            rope = self.inner.rotary_emb(x, positions)
            for i, layer in enumerate(self.inner.layers):
                x = self._ouro_block(layer, x, rope, cache, state.depth * self.n_core + i, position)
            x = self.inner.norm(x)
        else:
            freqs = self._freqs(x, position)
            x = self.inner.adapter(torch.cat((x, state.injected), dim=-1))
            for i, block in enumerate(self.inner.core_block):
                slot = self.n_pre + state.depth * self.n_core + i
                x = self._native_block(block, x, freqs, cache, slot, position)
        state.hidden = x
        state.depth += 1
        return state

    def readout(self, state, cache, last_only=False):
        x = state.hidden
        if self.family != "ouro":
            if self.family == "huginn":
                x = self.inner.ln_f(x)
            freqs = self._freqs(x, state.position)
            for i, block in enumerate(self.inner.coda):
                x = self._native_block(block, x, freqs, cache, -(i + 1), state.position)
            x = self.inner.ln_f(x)
        if last_only:
            x = x[:, -1:]
        return self.model.lm_head(x)[0].float()

    def full_forward(self, ids, position, cache, last_only=False):
        state = self.initialize(ids, position, cache)
        for _ in range(self.total_depth):
            self.advance(state, cache)
        return self.readout(state, cache, last_only)

    def prefill(self, ids, cache, chunk_size=512):
        if chunk_size < 1:
            raise ValueError("prefill chunk size must be positive")
        ids = ids.reshape(-1)
        for p in range(0, len(ids), chunk_size):
            logits = self.full_forward(ids[p : p + chunk_size], p, cache, last_only=True)
        return logits[-1]

    def _mixed_freqs(self, x, positions):
        if self.family == "huginn":
            return self.model.freqs_cis[:, positions]
        ids = torch.tensor([positions], device=self.device, dtype=torch.long)
        rotary = self.inner.rotary_emb if self.family == "ouro" else self.model.rotary_emb
        return rotary(x, ids)

    def _ouro_many(self, layer, x, freqs, cache, slots, positions, layout):
        residual = x
        h = layer.input_layernorm(x)
        attn = layer.self_attn
        b, n, _ = h.shape
        d = attn.head_dim
        q = attn.q_proj(h).view(b, n, self.config.num_attention_heads, d).transpose(1, 2)
        k = attn.k_proj(h).view(b, n, self.config.num_key_value_heads, d).transpose(1, 2)
        v = attn.v_proj(h).view(b, n, self.config.num_key_value_heads, d).transpose(1, 2)
        cos, sin = (t.unsqueeze(1) for t in freqs)

        def rotate(t):
            a, b = t.chunk(2, dim=-1)
            return torch.cat((-b, a), dim=-1)

        q, k = (q * cos + rotate(q) * sin, k * cos + rotate(k) * sin)
        y = (
            self._attention_many(q, k, v, cache, slots, positions, layout)
            .transpose(1, 2)
            .reshape(b, n, -1)
        )
        x = residual + layer.input_layernorm_2(attn.o_proj(y))
        return x + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))

    def _native_many(self, block, x, freqs, cache, slots, positions, layout=None):
        attn = block.attn
        olmo = hasattr(attn, "q_norm")
        h = x if olmo else block.norm_1(x)
        b, n, _ = h.shape
        q, k, v = attn.Wqkv(h).split(attn.chunks, dim=2)
        if olmo:
            q, k = (attn.q_norm(q), attn.k_norm(k))
        q = q.view(b, n, attn.n_head, attn.head_dim)
        k = k.view(b, n, attn.n_kv_heads, attn.head_dim)
        v = v.view(b, n, attn.n_kv_heads, attn.head_dim)
        if self.config.qk_bias:
            qb, kb = attn.qk_bias.split(1, dim=0)
            q, k = ((q + qb).to(q.dtype), (k + kb).to(q.dtype))
        module = sys.modules[type(self.model).__module__]
        if self.family == "huginn":
            q, k = module.apply_rotary_emb_complex_like(q, k, freqs_cis=freqs)
            q, k = (q.transpose(1, 2), k.transpose(1, 2))
        else:
            q, k = (q.transpose(1, 2), k.transpose(1, 2))
            q, k = module.apply_rotary_pos_emb(q, k, *freqs)
        y = self._attention_many(q, k, v.transpose(1, 2), cache, slots, positions, layout)
        y = attn.proj(y.transpose(1, 2).reshape(b, n, -1).contiguous())
        if self.family == "huginn":
            x = block.norm_2(y + x)
            return block.norm_4(block.mlp(block.norm_3(x)) + x)
        if olmo:
            x = block.norm_1(y) + x
            return block.norm_2(block.mlp(x)) + x
        x = y + x
        return block.mlp(block.norm_2(x)) + x

    def _attention_many(self, q, k, v, cache, slots, positions, layout=None):
        raise RuntimeError("A decode engine must bind attention before advancing speculative rows")
