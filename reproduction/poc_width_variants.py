"""E1: does the wrap fire for head_dim 64/128/256 (kernel specializations
<true,64>/<true,128>/<true,256>)? Leaf-style q,k,v at realistic scale, gapped
upstream crossing 4 GiB, wrapped-vs-packed dQ ratio per width. All synthetic.

Environment-honest behavior: not every cuDNN build serves every head width on
every GPU (head_dim 256 is unavailable with cuDNN 9.10.2 on H200, for
example). Each width records kernel_unavailable instead of aborting the
sweep, and result.json always writes with whatever completed.
"""

import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(1)
B, S, H = 2, 256, 1
out = Path("results/poc-e1-widths-038")
out.mkdir(parents=True, exist_ok=True)
generator = torch.Generator().manual_seed(1487)
results = {}

for D in (64, 128, 256):
    # wrap geometry analogous to the 128 case: batch 1 starts 128*H*D elements
    # before the 2^31-element line so its last 128 tokens wrap onto early cells
    STRIDE_B = 2**31 - H * D * 128
    SPAN = STRIDE_B + S * H * D
    x = torch.randn((B, S, D), generator=generator).to(torch.bfloat16).cuda()
    Wq = torch.randn((D, D), generator=generator).to(torch.bfloat16).cuda()
    Wk = torch.randn((D, D), generator=generator).to(torch.bfloat16).cuda()
    Wv = torch.randn((D, D), generator=generator).to(torch.bfloat16).cuda()

    def gapped(flat_bhsd):
        bhsd = flat_bhsd.transpose(1, 2).reshape(B, S, H, D)
        backing = torch.zeros((SPAN,), device="cuda", dtype=torch.bfloat16)
        view = backing.as_strided(
            (B, S, H, D), (STRIDE_B, H * D, D, 1), storage_offset=0
        )
        view.copy_(bhsd.to(torch.bfloat16))
        return view.transpose(1, 2)

    dense = torch.randn((B, H, S, D), generator=generator).to(torch.bfloat16).cuda()

    def dq_norm(upstream):
        q = (x @ Wq).contiguous().view(B, S, H, D).transpose(1, 2).requires_grad_()
        k = (x @ Wk).contiguous().view(B, S, H, D).transpose(1, 2).requires_grad_()
        v = (x @ Wv).contiguous().view(B, S, H, D).transpose(1, 2).requires_grad_()
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        dq, _, _ = torch.autograd.grad(o, [q, k, v], grad_outputs=upstream)
        return float(dq.float().norm())

    try:
        wrapped = dq_norm(gapped(dense))
        packed = dq_norm(dense.contiguous())
        results[str(D)] = {
            "wrapped": round(wrapped, 2),
            "packed": round(packed, 2),
            "ratio": round(wrapped / packed, 3),
            "stride_b": STRIDE_B,
            "span_bytes": SPAN * 2,
            "verdict": "wraps"
            if wrapped / packed > 2.0
            else ("no_wrap_or_hidden" if wrapped / packed < 1.2 else "ambiguous"),
        }
    except RuntimeError as error:
        results[str(D)] = {
            "status": "kernel_unavailable",
            "stride_b": STRIDE_B,
            "span_bytes": SPAN * 2,
            "error": str(error).splitlines()[0][:160],
        }

report = {
    "poc": "E1-width-variants",
    "torch": torch.__version__,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(),
    "results": results,
    "note": (
        "same leaf-style proven-wrapping construction at realistic value scale; "
        "ratio>2 = corrupted; kernel_unavailable = this cuDNN build lacks the "
        "width on this GPU (recorded, not fatal)"
    ),
}
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
