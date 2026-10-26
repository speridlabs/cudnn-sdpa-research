"""B2 (corrected): same-batch gapped-vs-packed dO gradient comparison.

The original B2 compared batched vs individual backward passes (x[:2] vs
x[:1]) which confounds batch size with dO layout, so it is invalid as
evidence. This redesign removes the confound: identical batch shape,
identical synthetic operands, identical leaf q/k/v; only the dO layout
differs between control and treatment.

Control    A: packed dO  (contiguous BHSD copy, sub-4-GiB address span)
Treatment  B: gapped dO  (strided view whose batch 1 crosses the 2^32-byte
             boundary; batch 0 stays below it)

A user able to produce a packed dO (for example via .contiguous()) can
detect the wrap corruption by comparing gradients directly - no FP64
reference is required. Expected outcome: batch 0 (below the boundary)
matches between A and B; batch 1 (crossing) diverges by the corruption
fingerprint. The packed control is additionally validated against the
independent FP64 reference to prove it is the correct side of the
comparison.

Exit codes: 0 detector confirmed the corruption; 2 no signal or sanity
failure. All numbers land in <output>/result.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import synthetic_boundary as sb

_THRESHOLD = 0.03


def _rel_l2(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Return the relative L2 distance, or None when either side is nonfinite.

    Args:
        left: Control measurement (FP64).
        right: Treatment measurement (FP64).

    Returns:
        ||left-right|| / ||right||, None if nonfinite or zero denominator.
    """
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return None
    denominator = float(right.norm())
    if denominator == 0.0:
        return None
    return float((left - right).norm() / denominator)


def _gradients(
    values: sb._TensorMap, batch_stride: int, *, packed: bool
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Run one forced-cuDNN backward and return raw BHSD CPU gradients.

    Args:
        values: Synthetic CPU operands from sb.operands().
        batch_stride: Batch stride in 16-bit elements (gapped case only).
        packed: True -> contiguous BHSD dO copy; False -> gapped strided view.

    Returns:
        ({"dq","dk","dv"} CPU tensors, layout metadata dict).
    """
    if packed:
        upstream = values["do"].to("cuda").transpose(1, 2).contiguous()
    else:
        storage_elements = batch_stride + sb._TOKENS * sb._HEADS * sb._WIDTH
        backing = torch.full(
            (storage_elements,), 0.0, device="cuda", dtype=values["do"].dtype
        )
        upstream = backing.as_strided(
            values["do"].shape,
            (batch_stride, sb._HEADS * sb._WIDTH, sb._WIDTH, 1),
            storage_offset=0,
        )
        upstream.copy_(values["do"])
        upstream = upstream.transpose(1, 2)
    q, k, v = (
        values[name].to("cuda").transpose(1, 2).requires_grad_()
        for name in ("q", "k", "v")
    )
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        dq, dk, dv = torch.autograd.grad(output, (q, k, v), upstream)
    metadata: dict[str, object] = {
        "packed": packed,
        "do_stride": list(upstream.stride()),
        "do_pointer": upstream.data_ptr(),
        "effective_do_span_bytes": sb.address_span(upstream),
    }
    grads = {name: t.detach().cpu() for name, t in (("dq", dq), ("dk", dk), ("dv", dv))}
    return grads, metadata


def main() -> int:
    """Run control and treatment in fresh layouts, compare, and persist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-stride", type=int, default=2**31 - 16384)
    parser.add_argument("--seed", type=int, default=1487)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False

    values = sb.operands(args.seed)
    truth = sb.reference(values)

    packed_grads, packed_meta = _gradients(values, 0, packed=True)
    gapped_grads, gapped_meta = _gradients(values, args.batch_stride, packed=False)

    divergence: dict[str, list[float | None]] = {}
    for name in ("dq", "dk", "dv"):
        divergence[name] = [
            _rel_l2(left, right)
            for left, right in zip(
                packed_grads[name].double(), gapped_grads[name].double(), strict=True
            )
        ]

    sanity = {
        name: sb.compare(packed_grads[name], expected)
        for name, expected in truth.items()
        if name != "output"
    }
    packed_sane = all(
        bool(row["finite"])
        and (row["relative_l2"] if row["relative_l2"] is not None else 1.0)
        <= _THRESHOLD
        for rows in sanity.values()
        for row in rows
    )

    crossing = max((divergence[name][1] or 0.0) for name in divergence)
    crossing_missing = any(divergence[name][1] is None for name in divergence)
    below = max((divergence[name][0] or 0.0) for name in divergence)
    confirmed = (
        packed_sane
        and (below <= _THRESHOLD)
        and (crossing > _THRESHOLD or crossing_missing)
    )

    report = {
        "status": "detector_confirms_corruption" if confirmed else "no_signal",
        "seed": args.seed,
        "threshold": _THRESHOLD,
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "batch_stride_gapped_elements": args.batch_stride,
        "packed_layout": packed_meta,
        "gapped_layout": gapped_meta,
        "rel_l2_packed_vs_gapped_per_batch": divergence,
        "crossing_batch_max_rel": crossing,
        "below_batch_max_rel": below,
        "packed_vs_reference_sanity": sanity,
        "packed_sane": packed_sane,
        "note": "same-batch comparison; the only variable is the dO layout",
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "result.json").write_text(text)
    print(text, flush=True)
    return 0 if confirmed else 2


if __name__ == "__main__":
    raise SystemExit(main())
