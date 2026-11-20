import gc
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from .capacity import available_bytes, kv_bytes_per_token
from .config import MODELS, resolve_config


class RequestPlacement:
    def __init__(
        self,
        name,
        checkpoint=None,
        *,
        devices,
        batch_size,
        context_parallel=False,
        **model_kwargs,
    ):
        self.name = name
        self.checkpoint = checkpoint or MODELS[name].repo_id
        self.devices = tuple(torch.device(d) for d in devices)
        self.batch_size = batch_size
        self.context_parallel = context_parallel
        self.model_kwargs = model_kwargs
        self.rank = 0
        self.world = 1
        self.control = None
        self.nccl = None
        if not self.devices or batch_size < 1:
            raise ValueError("Provide at least one device and a positive batch size")
        if context_parallel:
            if (
                not dist.is_initialized()
                or dist.get_world_size() < 2
                or len(self.devices) != 1
            ):
                raise ValueError(
                    "Context parallelism requires one CUDA device per distributed process"
                )
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            self.control = (
                dist.group.WORLD
                if dist.get_backend() == "gloo"
                else dist.new_group(backend="gloo", timeout=timedelta(hours=2))
            )
        self.config = AutoConfig.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        depth = model_kwargs.get("total_depth") or MODELS[name].depth
        self.kv_bytes = kv_bytes_per_token(
            self.config, depth, model_kwargs.get("dtype", torch.bfloat16)
        )
        self.single = None
        self.single_key = None

    @property
    def eos_ids(self):
        try:
            config = GenerationConfig.from_pretrained(self.checkpoint)
        except OSError:
            config = GenerationConfig.from_model_config(self.config)
        ids = config.eos_token_id
        if ids is None:
            ids = self.tokenizer.eos_token_id
        return () if ids is None else (ids,) if isinstance(ids, int) else tuple(ids)

    def _release_single(self):
        if self.single is not None:
            self.single = self.single_key = None
            gc.collect()
            torch.cuda.empty_cache()

    def close(self):
        self._release_single()
        if self.nccl is not None and self.nccl is not dist.group.WORLD:
            dist.destroy_process_group(self.nccl)
        if self.control is not None and self.control is not dist.group.WORLD:
            dist.destroy_process_group(self.control)
        self.nccl = self.control = None

    def _try_single(self, device, rows, budget, settings):
        from .batching import generate_batch
        from .batching.model import BatchModel
        from .model import LoopCacheModel
        from .runtime import generate

        torch.cuda.set_device(device)
        batched = len(rows) > 1 or bool(settings.get("options"))
        key = (device, batched)
        if self.single_key != key:
            self._release_single()
        capacities = [self.kv_bytes * (len(row) + budget) for row in rows]
        required = len(rows) * max(capacities)
        free = available_bytes(device)
        if required > free and self.single is not None:
            self._release_single()
            free = available_bytes(device)
        if required > free:
            return ("oversize" if min(capacities) > free else "split"), None
        if self.single is None:
            try:
                if batched:
                    self.single = BatchModel(
                        self.name,
                        self.checkpoint,
                        devices=[device],
                        batch_size=self.batch_size,
                        **self.model_kwargs,
                    )
                else:
                    self.single = LoopCacheModel(
                        self.name, self.checkpoint, device=device, **self.model_kwargs
                    )
                self.single_key = key
            except torch.cuda.OutOfMemoryError:
                return "weights", None
        free = available_bytes(device)
        if required > free:
            return ("oversize" if min(capacities) > free else "split"), None
        try:
            if batched:
                result = generate_batch(self.single, rows, budget, **settings)
            else:
                single_settings = settings.copy()
                single_settings.pop("options", None)
                chunk = single_settings.pop("prefill_chunk", 256)
                result = generate(
                    self.single,
                    rows[0],
                    max_new_tokens=budget,
                    prefill_chunk_size=chunk,
                    **single_settings,
                )
                result["token_ids"] = [result["token_ids"]]
                result["cuda_graph"] = result["graph_replays"] > 0
        except torch.cuda.OutOfMemoryError:
            return ("oversize" if len(rows) == 1 else "split"), None
        self.single.state_bank = None
        result.update(placement="single", gpu_count=1)
        return "done", result

    def _probe(self, owner, device, rows, budget, settings):
        message = [None]
        if self.rank == owner:
            try:
                status, result = self._try_single(device, rows, budget, settings)
                if status != "done":
                    self._release_single()
                    gc.collect()
                    torch.cuda.empty_cache()
                message[0] = status, result
            except Exception as exc:
                if self.control is None:
                    raise
                message[0] = "error", f"{type(exc).__name__}: {exc}"
                dist.broadcast_object_list(message, src=owner, group=self.control)
                raise
        if self.control is not None:
            dist.broadcast_object_list(message, src=owner, group=self.control)
        return message[0]

    def _parallel(self, rows, budget, settings, devices):
        from .batching import generate_batch
        from .batching.model import BatchModel

        self._release_single()
        if self.context_parallel:
            torch.cuda.set_device(self.devices[0])
            if self.nccl is None:
                self.nccl = (
                    dist.group.WORLD
                    if dist.get_backend() == "nccl"
                    else dist.new_group(backend="nccl", timeout=timedelta(hours=2))
                )
        model = BatchModel(
            self.name,
            self.checkpoint,
            devices=devices,
            batch_size=self.batch_size,
            context_parallel=self.context_parallel,
            context_group=self.nccl,
            **self.model_kwargs,
        )
        try:
            result = generate_batch(model, rows, budget, **settings)
        finally:
            del model
            if self.nccl is not None and self.nccl is not dist.group.WORLD:
                dist.destroy_process_group(self.nccl)
                self.nccl = None
            gc.collect()
            torch.cuda.empty_cache()
        result.update(
            placement="context" if self.context_parallel else "layer",
            gpu_count=self.world if self.context_parallel else len(devices),
        )
        return result

    def _run_rows(self, rows, budget, settings):
        candidates = (
            [(rank, self.devices[0]) for rank in range(self.world)]
            if self.context_parallel
            else [(0, d) for d in self.devices]
        )
        statuses = []
        for owner, device in candidates:
            status, result = self._probe(owner, device, rows, budget, settings)
            if status == "error":
                raise RuntimeError(result)
            if status == "done":
                if self.rank != owner:
                    self._release_single()
                return [result]
            statuses.append(status)
        if "split" in statuses and len(rows) > 1:
            middle = len(rows) // 2
            return self._run_rows(rows[:middle], budget, settings) + self._run_rows(
                rows[middle:], budget, settings
            )
        devices = [
            d for (_, d), status in zip(candidates, statuses) if status != "weights"
        ]
        if "weights" in statuses and (self.context_parallel or len(devices) < 2):
            raise torch.cuda.OutOfMemoryError(
                "Insufficient GPU memory for the checkpoint weights on the candidate devices"
            )
        if len(candidates) == 1:
            raise torch.cuda.OutOfMemoryError(
                "The sequence and its output budget exceed the available GPU memory"
            )
        return [
            self._parallel(
                rows,
                budget,
                settings,
                self.devices if self.context_parallel else devices,
            )
        ]

    @torch.inference_mode()
    def run(self, input_ids, max_new_tokens, **settings):
        if not 0 < len(input_ids) <= self.batch_size or any(
            not len(row) for row in input_ids
        ):
            raise ValueError(
                "Provide one nonempty token sequence per request within the batch size"
            )
        if max_new_tokens < 1 or settings.get("prefill_chunk", 256) < 1:
            raise ValueError(
                "Generation budget and prefill chunk size must be positive"
            )
        config = resolve_config(
            self.name,
            settings.get("method", "auto"),
            max(map(len, input_ids)),
            settings.get("profile", "tuned"),
            **{
                k: settings[k]
                for k in ("confidence_margin", "shallow_layers", "loop_gap")
                if k in settings
            },
        )
        settings["method"] = config["method"]
        if settings.get("shallow_layers") == 0:
            settings.pop("shallow_layers")
        batches = self._run_rows(input_ids, max_new_tokens, settings)
        first = batches[0]
        if len(batches) == 1:
            seconds = first["decode_seconds"]
            return dict(
                first,
                batches=batches,
                decode_tokens_per_second=first["decode_tokens"] / seconds
                if seconds
                else 0.0,
            )
        tokens = [row for batch in batches for row in batch["token_ids"]]
        decoded = sum(b["decode_tokens"] for b in batches)
        seconds = sum(b["decode_seconds"] for b in batches)
        return dict(
            method=first["method"]
            if all(b["method"] == first["method"] for b in batches)
            else "mixed",
            token_ids=tokens,
            decode_tokens=decoded,
            decode_seconds=seconds,
            decode_tokens_per_second=decoded / seconds if seconds else 0.0,
            prefill_seconds=sum(b["prefill_seconds"] for b in batches),
            graph_setup_seconds=sum(b["graph_setup_seconds"] for b in batches),
            cuda_graph=any(b["cuda_graph"] for b in batches),
            batches=batches,
        )
