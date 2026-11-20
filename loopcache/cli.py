import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from .config import MODELS


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def cuda_devices(value):
    try:
        indices = [int(item.removeprefix("cuda:")) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated CUDA device indices"
        ) from exc
    if min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("use distinct nonnegative CUDA device indices")
    return [f"cuda:{index}" for index in indices]


def arguments():
    parser = argparse.ArgumentParser(
        prog="loopcache", description="Generate text with LoopCache."
    )
    parser.add_argument("--model", choices=MODELS, default="ouro-1.4b")
    parser.add_argument(
        "--model-path", help="Checkpoint directory or Hugging Face repository ID"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List checkpoint names and repository IDs",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="A single text prompt")
    source.add_argument(
        "--input",
        type=Path,
        help="JSONL file with an id and prompt or input_ids per row",
    )
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--max-new-tokens", type=positive_int, default=32)
    parser.add_argument("--method", choices=("auto", "1", "2", "3"), default="auto")
    parser.add_argument("--profile", choices=("tuned", "uniform"), default="tuned")
    parser.add_argument(
        "--confidence-margin",
        type=float,
        help="Minimum new–old logit gap for a draft update",
    )
    parser.add_argument(
        "--shallow-layers",
        type=int,
        help="Number of leading layers with independent KV in method 3",
    )
    parser.add_argument(
        "--loop-gap",
        type=positive_int,
        help="Minimum loop distance for independent KV in method 3",
    )
    parser.add_argument("--prefill-chunk", type=positive_int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--fixed-tokens",
        action="store_true",
        help="Generate the full token budget, including EOS",
    )
    parallel = parser.add_mutually_exclusive_group()
    parallel.add_argument(
        "--devices",
        type=cuda_devices,
        help="Candidate CUDA devices, in preference order, for single-GPU inference and capacity fallback",
    )
    parallel.add_argument(
        "--context-parallel",
        action="store_true",
        help="Use context parallelism when a sequence exceeds single-GPU capacity",
    )
    args = parser.parse_args()
    if not args.list_models and args.prompt is None and args.input is None:
        parser.error("supply --prompt or --input")
    if args.context_parallel and (
        "LOCAL_RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) < 2
    ):
        parser.error("--context-parallel requires at least two torchrun processes")
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not args.context_parallel:
        parser.error("pass --context-parallel when launching with torchrun")
    return args


def read_requests(args):
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("The prompt is empty")
        return [{"id": 0, "prompt": args.prompt}]
    requests = []
    with args.input.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{args.input}, line {number}: {exc.msg}") from exc
            if not isinstance(row, dict) or ("prompt" in row) == ("input_ids" in row):
                raise ValueError(
                    f"Line {number} must contain one of prompt and input_ids"
                )
            if "prompt" in row and not (
                isinstance(row["prompt"], str) and row["prompt"].strip()
            ):
                raise ValueError(f"Line {number} has an empty or invalid prompt")
            if "input_ids" in row:
                ids = row["input_ids"]
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(type(i) is not int or i < 0 for i in ids)
                ):
                    raise ValueError(
                        f"Line {number} needs a nonempty list of nonnegative token IDs"
                    )
            row.setdefault("id", len(requests))
            requests.append(row)
    if not requests:
        raise ValueError("Input contains no requests")
    return requests


def run(args, requests, devices, emit, output=None):
    from .placement import RequestPlacement

    runner = RequestPlacement(
        args.model,
        args.model_path,
        devices=devices,
        batch_size=args.batch_size,
        context_parallel=args.context_parallel,
    )
    try:
        eos = () if args.fixed_tokens else runner.eos_ids
        settings = {
            key: getattr(args, key)
            for key in (
                "method",
                "profile",
                "seed",
                "confidence_margin",
                "shallow_layers",
                "loop_gap",
            )
            if getattr(args, key) is not None
        }
        batch_index = 0
        for offset in range(0, len(requests), args.batch_size):
            rows = requests[offset : offset + args.batch_size]
            inputs = [
                r["input_ids"]
                if "input_ids" in r
                else runner.tokenizer.encode(r["prompt"])
                for r in rows
            ]
            if any(
                not ids or min(ids) < 0 or max(ids) >= runner.config.vocab_size
                for ids in inputs
            ):
                raise ValueError(
                    "A prompt contains no tokens or a token outside the model vocabulary"
                )
            result = runner.run(
                inputs,
                args.max_new_tokens,
                eos_ids=eos,
                prefill_chunk=args.prefill_chunk,
                **settings,
            )
            if emit:
                start = 0
                for batch in result["batches"]:
                    size = len(batch["token_ids"])
                    for row, prompt_ids, ids in zip(
                        rows[start : start + size],
                        inputs[start : start + size],
                        batch["token_ids"],
                        strict=True,
                    ):
                        print(
                            json.dumps(
                                dict(
                                    id=row["id"],
                                    method=batch["method"],
                                    prompt_tokens=len(prompt_ids),
                                    token_ids=ids,
                                    text=runner.tokenizer.decode(
                                        ids, skip_special_tokens=True
                                    ),
                                    batch_index=batch_index,
                                    batch_size=size,
                                    placement=batch["placement"],
                                    gpu_count=batch["gpu_count"],
                                    batch_decode_seconds=batch["decode_seconds"],
                                ),
                                ensure_ascii=False,
                            ),
                            flush=True,
                            file=output,
                        )
                    start += size
                    batch_index += 1
    finally:
        runner.close()


def main():
    args = arguments()
    if args.list_models:
        for name, checkpoint in MODELS.items():
            print(f"{name}\t{checkpoint.repo_id}")
        return
    requests = read_requests(args)
    import torch

    if args.context_parallel:
        import torch.distributed as dist

        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        sys.stdout.flush()
        with os.fdopen(os.dup(1), "w", encoding="utf-8") as output:
            os.dup2(2, 1)
            try:
                dist.init_process_group("gloo", timeout=timedelta(hours=2))
                try:
                    run(args, requests, [device], dist.get_rank() == 0, output)
                finally:
                    dist.destroy_process_group()
            finally:
                sys.stdout.flush()
                os.dup2(output.fileno(), 1)
    else:
        devices = args.devices or [
            f"cuda:{index}" for index in range(torch.cuda.device_count())
        ]
        run(args, requests, devices, True)
