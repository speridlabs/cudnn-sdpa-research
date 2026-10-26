"""PaddlePaddle reachability probe: can a gapped (strided) dO reach paddle's
SDPA backward, and does it produce the wrap fingerprint?

Paddle's SDPA expects (B, S, H, D) tensors (unlike torch's (B,H,S,D)) — this
probe uses paddle's native layout throughout. Three cases:

1. sanity  — contiguous dO, no cat: validates layouts against an
   independent FP64 CPU reference (must pass).
2. cat     — natural trigger at receipt scale: attention output concatenated
   with a sibling; records the delivered-view structure and correctness.
3. gapped  — direct 4-GiB-crossing strided dO built with paddle.as_strided
   (batch 1 starts 2^31-16384 elements before the 2^32-byte boundary):
   the wrap test. A passing result means this recorded Paddle stack did not
   reproduce the cuDNN fingerprint; it is not a general non-impact claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import paddle

_THRESHOLD = 0.03
_BATCH, _SEQ, _HEADS, _WIDTH = 2, 256, 1, 128
_GAPPED_STRIDE = 2**31 - 16384


def _reference(
    q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, do_bshd: paddle.Tensor
) -> dict[str, paddle.Tensor]:
    """Independent FP64 CPU reference; returns BSHD gradients."""
    doubles = []
    for tensor in (q, k, v, do_bshd):
        copy = tensor.astype("float64").detach()
        copy.stop_gradient = False
        doubles.append(copy)
    qd, kd, vd, dod = doubles
    qh, kh, vh, doh = (t.transpose((0, 2, 1, 3)) for t in doubles)  # BHSD
    scores = paddle.matmul(qh, kh.transpose((0, 1, 3, 2))) / math.sqrt(_WIDTH)
    probability = paddle.nn.functional.softmax(scores, axis=-1)
    output = paddle.matmul(probability, vh)
    output.backward(doh)
    # qd/kd/vd are (B,S,H,D) leaves, so their .grad is already BSHD — no
    # extra transpose (the earlier double transpose compared BHSD against
    # BSHD and produced broadcast garbage).
    return {name: tensor.grad for name, tensor in (("dq", qd), ("dk", kd), ("dv", vd))}


def _compare(
    actual: dict[str, paddle.Tensor], reference: dict[str, paddle.Tensor]
) -> dict[str, list]:
    metrics: dict[str, list] = {}
    for name in ("dq", "dk", "dv"):
        rows = []
        for index in range(_BATCH):
            left = actual[name][index].astype("float64")
            right = reference[name][index]
            finite = bool(paddle.isfinite(left).all())
            relative = float((left - right).norm() / right.norm()) if finite else None
            rows.append({"finite": finite, "relative_l2": relative})
        metrics[name] = rows
    return metrics


def _case_failed(metrics: dict[str, list]) -> bool:
    return any(
        not row["finite"]
        or (row["relative_l2"] is not None and row["relative_l2"] > _THRESHOLD)
        for rows in metrics.values()
        for row in rows
    )


def _fresh_operands(seed: int, dtype: str):
    paddle.seed(seed)
    q = paddle.randn((_BATCH, _SEQ, _HEADS, _WIDTH), dtype=dtype)
    k = paddle.randn((_BATCH, _SEQ, _HEADS, _WIDTH), dtype=dtype)
    v = paddle.randn((_BATCH, _SEQ, _HEADS, _WIDTH), dtype=dtype)
    for tensor in (q, k, v):
        tensor.stop_gradient = False
    do = paddle.randn((_BATCH, _SEQ, _HEADS, _WIDTH), dtype=dtype)
    return q, k, v, do


def main() -> int:
    """Run the three paddle cases and persist the reachability receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1487)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True, exist_ok=False)

    gpu = paddle.device.get_device()
    dtype = "bfloat16"
    results: dict[str, dict] = {}

    # ---- case 1: sanity, contiguous dO ----
    q, k, v, do = _fresh_operands(args.seed, dtype)
    reference = _reference(q, k, v, do)
    out = paddle.nn.functional.scaled_dot_product_attention(q, k, v)
    out.backward(do)
    sanity = _compare(
        {name: tensor.grad for name, tensor in (("dq", q), ("dk", k), ("dv", v))},
        reference,
    )
    results["sanity_contiguous"] = {
        "metrics": sanity,
        "passed": not _case_failed(sanity),
    }

    # ---- case 2: natural cat trigger at receipt scale ----
    cat_width = 4
    q, k, v, do = _fresh_operands(args.seed, dtype)
    sibling = paddle.randn(
        (_BATCH, _SEQ, _HEADS, _WIDTH * (cat_width - 1)), dtype=dtype
    )
    out = paddle.nn.functional.scaled_dot_product_attention(q, k, v)
    full = paddle.concat((out, sibling), axis=-1)
    weights = paddle.arange(full.size, dtype="float32").reshape(full.shape) / float(
        full.size
    )
    # the upstream that actually reaches attention-backward is the weighted
    # slice over the attention columns, not a fresh random tensor
    upstream_slice = weights[:, :, :, :_WIDTH].astype(dtype)
    reference = _reference(q, k, v, upstream_slice)
    loss = (full.astype("float32") * weights).sum()
    loss.backward()
    cat = _compare(
        {name: tensor.grad for name, tensor in (("dq", q), ("dk", k), ("dv", v))},
        reference,
    )
    results["cat_trigger_receipt_scale"] = {
        "cat_width": cat_width,
        "metrics": cat,
        "passed": not _case_failed(cat),
    }

    # ---- case 3: direct 4-GiB-crossing gapped dO ----
    q, k, v, do = _fresh_operands(args.seed, dtype)
    reference = _reference(q, k, v, do)
    span = _GAPPED_STRIDE + _SEQ * _HEADS * _WIDTH
    backing = paddle.zeros((span,), dtype=dtype)
    block = _SEQ * _HEADS * _WIDTH
    backing[0:block] = paddle.reshape(do[0], (block,))
    backing[_GAPPED_STRIDE : _GAPPED_STRIDE + block] = paddle.reshape(do[1], (block,))
    gapped = paddle.as_strided(
        backing,
        (_BATCH, _SEQ, _HEADS, _WIDTH),
        (_GAPPED_STRIDE, _HEADS * _WIDTH, _WIDTH, 1),
        0,
    )
    out = paddle.nn.functional.scaled_dot_product_attention(q, k, v)
    out.backward(gapped)
    gapped_metrics = _compare(
        {name: tensor.grad for name, tensor in (("dq", q), ("dk", k), ("dv", v))},
        reference,
    )
    gapped_failed = _case_failed(gapped_metrics)
    crossing_batch_rel = {
        name: rows[1]["relative_l2"] for name, rows in gapped_metrics.items()
    }
    results["gapped_4gib_crossing"] = {
        "batch_stride_elements": _GAPPED_STRIDE,
        "span_bytes": span * 2,
        "metrics": gapped_metrics,
        "passed": not gapped_failed,
        "crossing_batch_relative_l2": crossing_batch_rel,
    }

    as_strided = hasattr(paddle, "as_strided")
    if results["gapped_4gib_crossing"]["passed"]:
        verdict = "paddle_sdpa_gradients_correct_under_gapped_do"
    elif abs((crossing_batch_rel.get("dq") or 0.0) - 0.13) < 0.05:
        verdict = "wrap_fingerprint_like_cudnn"
    else:
        verdict = "numerical_failure_under_gapped_do"

    report = {
        "status": "completed",
        "framework": "paddlepaddle",
        "paddle_version": paddle.__version__,
        "gpu": gpu,
        "dtype": dtype,
        "layout_convention": "paddle SDPA expects (B, S, H, D)",
        "as_strided_available": as_strided,
        "verdict": verdict,
        "threshold": _THRESHOLD,
        "cases": results,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "result.json").write_text(text)
    print(text, flush=True)
    return 2 if gapped_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
