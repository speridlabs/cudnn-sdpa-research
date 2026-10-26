"""C6 probe: which SDPA backend does the DEFAULT framework path select?

Replicates the exact layout construction of code/synthetic_boundary.py
execute() but parameterizes backend policy: 'default' runs WITHOUT any
sdpa_kernel context manager, the others wrap SDPBackend explicitly.
One variant per fresh process. Saves result tensors for bitwise comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import synthetic_boundary as sb

_EXPLICIT = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def run_variant(values: dict[str, torch.Tensor], batch_stride: int, backend: str) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Execute the crossing layout under one backend-selection policy."""
    storage_elements = batch_stride + sb._TOKENS * sb._HEADS * sb._WIDTH
    backing = torch.full((storage_elements,), 0.0, device="cuda", dtype=values["do"].dtype)
    upstream = backing.as_strided(
        values["do"].shape, (batch_stride, sb._HEADS * sb._WIDTH, sb._WIDTH, 1), storage_offset=0
    )
    upstream.copy_(values["do"])
    upstream = upstream.transpose(1, 2)
    q, k, v = (values[name].to("cuda").transpose(1, 2).requires_grad_() for name in ("q", "k", "v"))
    original_stride = list(upstream.stride())
    original_pointer = upstream.data_ptr()
    if backend == "default":
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        with sdpa_kernel(_EXPLICIT[backend]):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    derivatives = torch.autograd.grad(output, (q, k, v), upstream)
    metadata: dict[str, object] = {
        "backend_policy": backend,
        "shape_bhsd": list(q.shape),
        "do_stride": list(upstream.stride()),
        "storage_bytes": backing.numel() * backing.element_size(),
        "effective_do_span_bytes": sb.address_span(upstream),
        "original_do_stride": original_stride,
        "pointer_changed_by_explicit_ctx": original_pointer != upstream.data_ptr(),
    }
    tensors = {name: t.detach().cpu() for name, t in zip(("output", "dq", "dk", "dv"), (output, *derivatives), strict=True)}
    return tensors, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("default", "math", "flash", "cudnn"), required=True)
    parser.add_argument("--batch-stride", type=int, default=2147467264)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1487)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    values = sb.operands(args.seed, {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype])
    truth = sb.reference(values)
    actual, layout = run_variant(values, args.batch_stride, args.backend)
    metrics = {name: sb.compare(actual[name], expected) for name, expected in truth.items()}
    failed = any(
        not row["finite"] or float(row["relative_l2"]) > sb._THRESHOLD
        for rows in metrics.values() for row in rows
    )
    report = {
        "status": "numerical_failure" if failed else "completed",
        "backend_policy": args.backend,
        "seed": args.seed,
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "layout": layout,
        "threshold": sb._THRESHOLD,
        "metrics": metrics,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(actual, args.output / "tensors.pt")
    print(json.dumps(report, indent=2), flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
