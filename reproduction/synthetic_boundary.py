"""Public-safe SDPA probe for 16-bit byte addressing across the 4 GiB boundary.

Only synthetic tensors are used. A large batch stride makes a small logical
problem cross 2**32 bytes; the backing allocation requires about 4 GiB. Run each
layout in a fresh process to exclude graph-cache effects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_BATCH = 2
_TOKENS = 256
_HEADS = 1
_WIDTH = 128
_THRESHOLD = 0.03
_BYTE_ADDRESS_LIMIT = 2**32
_TensorMap = dict[str, torch.Tensor]


def operands(seed: int, dtype: torch.dtype = torch.bfloat16) -> _TensorMap:
    """Return deterministic, entirely synthetic CPU BSHD operands.

    Args:
        seed: Seed of a local CPU random generator.
        dtype: BF16 or FP16 dtype used by the chosen experiment.

    Returns:
        Query, key, value and upstream tensors with identical fixed dimensions.
    """
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn((_BATCH, _TOKENS, _HEADS, _WIDTH), generator=generator).to(dtype)
        for name in ("q", "k", "v", "do")
    }


def reference(values: _TensorMap) -> _TensorMap:
    """Compute all output and gradient coordinates independently in CPU FP64.

    Args:
        values: Logical BSHD operands.

    Returns:
        BHSD output and Q/K/V derivatives, using explicit softmax mathematics.
    """
    q, k, v = (values[name].double().transpose(1, 2).requires_grad_() for name in ("q", "k", "v"))
    probability = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(_WIDTH), dim=-1)
    output = probability @ v
    derivatives = torch.autograd.grad(output, (q, k, v), values["do"].double().transpose(1, 2))
    return dict(zip(("output", "dq", "dk", "dv"), (output.detach(), *derivatives), strict=True))


def address_span(tensor: torch.Tensor) -> int:
    """Return reachable bytes from data_ptr, excluding preceding backing cells.

    Args:
        tensor: Tensor with nonnegative strides.

    Returns:
        Zero for empty tensors; otherwise the final reachable byte offset plus one.
    """
    if not tensor.numel():
        return 0
    return (
        1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    ) * tensor.element_size()


def normalize_upstream(output: torch.Tensor, upstream: torch.Tensor) -> torch.Tensor:
    """Prototype the proposed layout-admission mitigation, without vendor changes.

    Args:
        output: Same-shaped cuDNN forward output supplying a dense output layout.
        upstream: Incoming gradient, possibly a view with a large reachable span.

    Returns:
        The same tensor when safe, otherwise an equal-valued packed copy.

    Raises:
        RuntimeError: Even the logical packed gradient cannot fit the address limit.
    """
    if address_span(upstream) <= _BYTE_ADDRESS_LIMIT:
        return upstream
    if upstream.numel() * upstream.element_size() > _BYTE_ADDRESS_LIMIT:
        message = "cuDNN dO exceeds the tested 32-bit byte range even when packed; choose another forward SDPA backend"
        raise RuntimeError(message)
    packed = torch.empty_like(output).copy_(upstream)
    if address_span(packed) > _BYTE_ADDRESS_LIMIT:
        message = "Output-matching gradient layout still exceeds the 32-bit byte range"
        raise RuntimeError(message)
    return packed


def execute(
    values: _TensorMap, batch_stride: int, padding: float, storage_offset: int = 0, *, guard: bool = False
) -> tuple[_TensorMap, dict[str, object]]:
    """Execute forced cuDNN with explicit upstream backing storage.

    Args:
        values: Logical CPU BSHD tensors, unchanged across layouts.
        batch_stride: Upstream batch stride in 16-bit elements.
        padding: Finite value for backing cells outside the logical tensor.
        storage_offset: Nonnegative upstream offset relative to its backing base.
        guard: Whether to prototype the proposed byte-span normalization.

    Returns:
        CPU BHSD results and actual layout/pointer metadata.
    """
    storage_elements = storage_offset + batch_stride + _TOKENS * _HEADS * _WIDTH
    backing = torch.full((storage_elements,), padding, device="cuda", dtype=values["do"].dtype)
    upstream = backing.as_strided(
        values["do"].shape, (batch_stride, _HEADS * _WIDTH, _WIDTH, 1), storage_offset=storage_offset
    )
    upstream.copy_(values["do"])
    upstream = upstream.transpose(1, 2)
    q, k, v = (values[name].to("cuda").transpose(1, 2).requires_grad_() for name in ("q", "k", "v"))
    original_stride = list(upstream.stride())
    original_pointer = upstream.data_ptr()
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if guard:
            upstream = normalize_upstream(output, upstream)
        derivatives = torch.autograd.grad(output, (q, k, v), upstream)
    metadata: dict[str, object] = {
        "shape_bhsd": list(q.shape),
        "q_stride": list(q.stride()),
        "do_stride": list(upstream.stride()),
        "do_pointer": upstream.data_ptr(),
        "output_pointer": output.data_ptr(),
        "storage_bytes": backing.numel() * backing.element_size(),
        "padding": padding,
        "dtype": str(upstream.dtype),
        "storage_offset_elements": upstream.storage_offset(),
        "guard": guard,
        "guard_copied": original_pointer != upstream.data_ptr(),
        "original_do_stride": original_stride,
        "effective_do_span_bytes": address_span(upstream),
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in zip(("output", "dq", "dk", "dv"), (output, *derivatives), strict=True)
    }, metadata


def compare(actual: torch.Tensor, expected: torch.Tensor) -> list[dict[str, object]]:
    """Measure every coordinate by batch, preserving nonfinite failures.

    Args:
        actual: Measured CPU tensor with leading batch axis.
        expected: Independent FP64 tensor of the same shape.

    Returns:
        Per-batch finite status, relative L2 error and maximum absolute error.
    """
    result: list[dict[str, object]] = []
    for left, right in zip(actual.double(), expected, strict=True):
        finite = bool(torch.isfinite(left).all())
        error = left - right
        result.append(
            {
                "finite": finite,
                "relative_l2": float(error.norm() / right.norm()) if finite else None,
                "max_absolute": float(error.abs().max()) if finite else None,
            }
        )
    return result


def main() -> int:
    """Validate bounds, run one isolated case, and persist its numerical status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-stride", type=int, default=2**31 - _TOKENS * _WIDTH // 2)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1487)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--storage-offset", type=int, default=0)
    parser.add_argument("--guard", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not math.isfinite(args.padding):
        parser.error("output must be new and padding finite")
    if not _TOKENS * _WIDTH <= args.batch_stride <= 2**31:
        parser.error("batch stride must be between the dense span and 2**31 elements")
    if not 0 <= args.storage_offset <= _WIDTH:
        parser.error("storage offset must be between zero and head width")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    values = operands(args.seed, {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype])
    truth = reference(values)
    actual, layout = execute(values, args.batch_stride, args.padding, args.storage_offset, guard=args.guard)
    metrics = {name: compare(actual[name], expected) for name, expected in truth.items()}
    failed = any(
        not row["finite"] or float(row["relative_l2"]) > _THRESHOLD for rows in metrics.values() for row in rows
    )
    report = {
        "status": "numerical_failure" if failed else "completed",
        "seed": args.seed,
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "layout": layout,
        "threshold": _THRESHOLD,
        "metrics": metrics,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "result.json").write_text(text)
    print(text, flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
