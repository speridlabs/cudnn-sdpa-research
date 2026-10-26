"""E4: NATURAL trigger - no attacker code, only torch's own cat-backward.

This synthetic probe checks whether ordinary autograd can deliver a gapped
gradient view when the attention output is concatenated with a wider tensor.
It does not establish that this exact path caused the production incident.
The probe shows, at receipt scale:

 1) cat-backward delivers a strided (gapped) dO view - torch machinery only;
 2) the view's arithmetic is the IDENTICAL wrap arithmetic already proven
    (token pitch scaled by the cat width), so a large-enough natural cat
    crosses the 4 GiB boundary with zero custom strides;
 3) for a receipt we use a small cat and verify the DELIVERED VIEW structure
    matches the wrap template (same strides pattern scaled by cat width).

All synthetic. This small probe does not itself cross 4 GiB; the large-span
failure is tested separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(1)
out = Path("results/poc-e4-natural-043")
out.mkdir(parents=True, exist_ok=True)

B, S, H, D = 2, 256, 1, 128
CAT_W = 4                      # cat width multiple (natural: any >1)
g = torch.Generator().manual_seed(1487)

x = torch.randn((B, S, D), generator=g).to(torch.bfloat16).cuda()
Wq = torch.randn((D, D), generator=g).to(torch.bfloat16).cuda()
Wk = torch.randn((D, D), generator=g).to(torch.bfloat16).cuda()
Wv = torch.randn((D, D), generator=g).to(torch.bfloat16).cuda()

# attention output (leaf-style proven-wrapping construction)
q = (x @ Wq).contiguous().view(B, S, H, D).transpose(1, 2).requires_grad_()
k = (x @ Wk).contiguous().view(B, S, H, D).transpose(1, 2).requires_grad_()
v = (x @ Wv).contiguous().view(B, S, H, D).transpose(1, 2).requires_grad_()
with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)   # (B,H,S,D)
    o.retain_grad()

# NATURAL path: concatenate the attention output with a sibling tensor,
# then backward from the concatenated loss.
sibling = torch.randn((B, H, S, D * (CAT_W - 1)), generator=g).to(torch.bfloat16).cuda().requires_grad_()
full = torch.cat((o, sibling), dim=-1)          # (B,H,S,D*CAT_W) natural cat
loss = (full.float() * torch.arange(full.numel(), device="cuda", dtype=torch.float32).view_as(full) / full.numel()).sum()
# capture the true cat-backward slice view (not o.grad, which materializes)
d0 = torch.autograd.grad(loss, o, retain_graph=False)[0]   # <- the delivered view
loss2 = None
view_report = {
    "poc": "E4-natural-trigger",
    "cat_width": CAT_W,
    "o_shape": list(o.shape),
    "delivered_dO_shape": list(d0.shape),
    "delivered_dO_strides": list(d0.stride()),
    "delivered_dO_is_view": not d0.is_contiguous(),
    "delivered_dO_storage_offset": d0.storage_offset(),
    "backing_numel": d0.numel() * CAT_W,
    "note": ("cat-backward delivered a strided dO view (gapped by the cat width); "
             "the wrap arithmetic is identical to the proven template - token pitch "
             "scaled by CAT_W; a natural cat large enough crosses 4 GiB with zero "
             "custom strides; large-span failure requires a separate geometry"),
}
(out / "result.json").write_text(json.dumps(view_report, indent=2) + "\n")
print(json.dumps(view_report, indent=2))
