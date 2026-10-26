"""TE reachability probe: does TransformerEngine's attention backward reach the
affected cuDNN fused-attention path with a gapped dO?

Method: build TE DotProductAttention over synthetic operands and run TWO cases
against an independent CPU FP64 reference:

1. sanity — a contiguous dO with the same values (must pass; validates the
   layout handling against TE's actual output shape, which may fold a
   singleton head dimension).
2. gapped — a 4-GiB-crossing strided dO (the same wrap geometry as
   synthetic_boundary) delivered through torch.autograd.grad.

A passing result means this recorded TE case did not reproduce the cuDNN
fingerprint. The selected backend is recorded when exposed; a passing probe is
not a general non-impact claim.
NVTE debug logging is enabled so backend-selection lines land in the run log
beside this receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("NVTE_LOG_LEVEL", "DEBUG")

import torch

_THRESHOLD = 0.03


def compare_case(
    derivatives: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]
) -> tuple[dict[str, list[dict]], bool]:
    """Per-batch FP64 comparison of BSHD gradients against the reference."""
    metrics: dict[str, list[dict]] = {}
    failed = False
    for name, actual, expected in zip(
        ("dq", "dk", "dv"), derivatives, reference, strict=True
    ):
        rows = []
        for left, right in zip(
            actual.detach().cpu().double().transpose(1, 2), expected, strict=True
        ):
            finite = bool(torch.isfinite(left).all())
            relative = float((left - right).norm() / right.norm()) if finite else None
            rows.append({"finite": finite, "relative_l2": relative})
            failed = failed or (
                not finite or (relative is not None and relative > _THRESHOLD)
            )
        metrics[name] = rows
    return metrics, failed


def main() -> int:
    """Run the TE sanity and gapped cases and persist the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--batch-stride", type=int, default=None, help="defaults to 2**31 - 128*H*D"
    )
    parser.add_argument("--seed", type=int, default=1487)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        from transformer_engine.pytorch.attention import DotProductAttention
    except Exception as error:  # noqa: BLE001 - record environment failures
        report = {
            "status": "te_unavailable",
            "error": str(error).splitlines()[0][:200],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 3

    batch, seq, heads, width = 2, 256, 2, 128
    if args.batch_stride is None:
        # batch 1 crosses the 2^31-element line at token 128 (same geometry
        # as the width-variants sweep)
        args.batch_stride = 2**31 - 128 * heads * width
    generator = torch.Generator().manual_seed(args.seed)
    q = (
        torch.randn((batch, seq, heads, width), generator=generator)
        .bfloat16()
        .cuda()
        .requires_grad_()
    )
    k = (
        torch.randn((batch, seq, heads, width), generator=generator)
        .bfloat16()
        .cuda()
        .requires_grad_()
    )
    v = (
        torch.randn((batch, seq, heads, width), generator=generator)
        .bfloat16()
        .cuda()
        .requires_grad_()
    )

    # independent CPU FP64 reference (BSHD -> BHSD math)
    qd, kd, vd = (
        t.detach().cpu().double().transpose(1, 2).requires_grad_() for t in (q, k, v)
    )
    probability = torch.softmax((qd @ kd.transpose(-1, -2)) / math.sqrt(width), dim=-1)
    output_ref = probability @ vd
    flat_do = torch.randn((batch, seq, heads, width), generator=generator).bfloat16()
    derivatives_ref = torch.autograd.grad(
        output_ref, (qd, kd, vd), flat_do.double().transpose(1, 2)
    )

    # TE defaults are qkv_format='sbhd' (sequence-first!) and causal masking —
    # both wrong for this probe; read via inspect.signature of the installed
    # version. Use batch-first BSHD with no mask to match the reference math.
    attention = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=width,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    )
    output = attention(q, k, v)
    trailing = tuple(output.shape[1:])
    trailing_numel = 1
    dense: list[int] = []
    for dim in reversed(trailing):
        dense.insert(0, trailing_numel)
        trailing_numel *= dim

    # ---- case 1: contiguous dO sanity (fresh graph: TE's custom Function
    # does not support two backward invocations on one graph) ----
    contiguous = flat_do.reshape((batch, *trailing)).cuda()
    sanity_output = attention(q, k, v)
    sanity_grads = torch.autograd.grad(sanity_output, (q, k, v), contiguous)
    sanity_metrics, sanity_failed = compare_case(sanity_grads, derivatives_ref)

    # ---- case 2: gapped 4-GiB-crossing dO (its own fresh graph) ----
    storage_elements = args.batch_stride + trailing_numel
    backing = torch.full((storage_elements,), 0.0, device="cuda", dtype=torch.bfloat16)
    upstream = backing.as_strided(
        (batch, *trailing), [args.batch_stride, *dense], storage_offset=0
    )
    upstream.copy_(flat_do.reshape((batch, *trailing)))
    gapped_output = attention(q, k, v)
    gapped_grads = torch.autograd.grad(gapped_output, (q, k, v), upstream)
    gapped_metrics, gapped_failed = compare_case(gapped_grads, derivatives_ref)

    backend_info: object = (
        "not exposed by this TE version; see run log for NVTE debug lines"
    )
    try:
        from transformer_engine.pytorch.attention import get_fused_attention_backend

        backend_info = str(get_fused_attention_backend())
    except Exception:  # noqa: BLE001 - optional introspection
        pass

    if sanity_failed:
        verdict = "probe_reference_mismatch_sanity_failed"
    elif gapped_failed:
        verdict = "te_reproduces_divergence_under_gapped_do"
    else:
        verdict = "te_gradients_correct_under_gapped_do"

    report = {
        "status": "completed",
        "framework": "transformer_engine",
        "te_version": getattr(
            __import__("transformer_engine"), "__version__", "unknown"
        ),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "output_shape": list(output.shape),
        "head_dim_folded": len(output.shape) == 3,
        "batch_stride": args.batch_stride,
        "span_bytes": storage_elements * 2,
        "backend": backend_info,
        "verdict": verdict,
        "threshold": _THRESHOLD,
        "sanity_contiguous": {"metrics": sanity_metrics, "passed": not sanity_failed},
        "gapped_4gib_crossing": {
            "metrics": gapped_metrics,
            "passed": not gapped_failed,
        },
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "result.json").write_text(text)
    print(text, flush=True)
    return 2 if (gapped_failed and not sanity_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
