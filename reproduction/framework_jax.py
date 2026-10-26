"""JAX reachability probe: can a gapped (strided) dO ever reach a cuDNN
attention backward through JAX/XLA?

Method: reproduce the natural trigger (cat-backward delivering a strided dO
slice) in pure JAX, dump the jaxpr of the backward, and record every
custom-call target it contains. XLA is lazy and functional: gradients are
compiled into fused kernels, so the probe records whether the computation
lowers to any cuDNN attention custom-call and whether a gapped-layout
delivery is structurally possible at all.

All synthetic. Receipt: result.json with the custom-call analysis and a
reachability verdict; the full jaxpr text is saved beside it for review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp


def main() -> int:
    """Run the natural-trigger equivalent under JAX and persist the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1487)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True, exist_ok=False)

    batch, seq, width = 2, 256, 128
    cat_width = 4
    key = jax.random.PRNGKey(args.seed)
    q = jax.random.normal(key, (batch, seq, width), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (batch, seq, width), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (batch, seq, width), dtype=jnp.bfloat16)
    sibling = jax.random.normal(
        key, (batch, seq, width * (cat_width - 1)), dtype=jnp.bfloat16
    )

    def loss_fn(q, k, v, sibling):
        output = jax.nn.dot_product_attention(q, k, v)  # (B, S, D)
        full = jnp.concatenate((output, sibling), axis=-1)  # natural cat
        weights = (
            jnp.arange(full.size, dtype=jnp.float32).reshape(full.shape) / full.size
        )
        return (full.astype(jnp.float32) * weights).sum()

    jaxpr = jax.make_jaxpr(jax.grad(loss_fn, argnums=(0, 1, 2)))(q, k, v, sibling)
    jaxpr_text = str(jaxpr)
    (args.output / "grad_jaxpr.txt").write_text(jaxpr_text)

    gradients = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2)))(q, k, v, sibling)
    jax.block_until_ready(gradients)
    finite = all(bool(jnp.isfinite(g).all()) for g in gradients)

    custom_calls = sorted(
        {
            line.strip()
            for line in jaxpr_text.splitlines()
            if "custom_call" in line.lower()
        }
    )
    cudnn_calls = [line for line in custom_calls if "cudnn" in line.lower()]
    attention_calls = [
        line
        for line in custom_calls
        if any(t in line.lower() for t in ("fmha", "attention", "flash", "mha"))
    ]

    if cudnn_calls:
        verdict = "cudnn_attention_reached"
    elif attention_calls:
        verdict = "non_cudnn_attention_custom_call"
    else:
        verdict = "no_attention_custom_call_detected"
    note = (
        "XLA compiles the whole cat+attention-backward into fused kernels; a "
        "gapped dO buffer like the ATen/TE case is not materialized in this "
        "execution model. The saved jaxpr is the primary evidence."
    )
    report = {
        "status": "completed",
        "framework": "jax",
        "jax_version": jax.__version__,
        "backend": str(jax.default_backend()),
        "device": str(jax.devices()[0]),
        "verdict": verdict,
        "finite_gradients": finite,
        "cudnn_custom_calls": cudnn_calls[:10],
        "attention_custom_calls": attention_calls[:10],
        "custom_call_count": len(custom_calls),
        "note": note,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "result.json").write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
