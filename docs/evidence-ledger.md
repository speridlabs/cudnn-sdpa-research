# Evidence ledger

This ledger separates dynamic measurements, static binary analysis, and
mitigation validation. The `evidence/` files are normalized receipt copies.
Pointer and address fields were removed from copied receipts; source and
destination hashes are recorded in [`source-manifest.json`](../evidence/source-manifest.json).

## Core numerical claims

| Claim                                                                                     | Evidence                                                                                                                                                                                                                                                        | Conclusion and limit                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The crossing layout changes backward gradients while the below-boundary control does not. | [`layout/same-batch-detector`](../evidence/receipts/layout/same-batch-detector/result.json), [`core/crossing-boundary`](../evidence/receipts/core/crossing-boundary/result.json), [`core/below-boundary`](../evidence/receipts/core/below-boundary/result.json) | Same-batch comparison: crossing batch `dQ 0.1284823`, `dK 0.1011135`, `dV 0`; below batch all zero. One H200 geometry.                                                                                                                                            |
| The packed result is a usable control against a separate reference.                       | Same-batch receipt, `packed_vs_reference_sanity` fields                                                                                                                                                                                                         | All three gradients are finite and around `0.0023–0.0024` relative-L2 from CPU FP64. This is a sanity tolerance, not exact equality.                                                                                                                              |
| The backend result differs from the CPU FP64 reference on the crossing batch.             | [`core/crossing-boundary`](../evidence/receipts/core/crossing-boundary/result.json)                                                                                                                                                                             | `dQ 0.1304754`, `dK 0.1022068`; this is not the same metric as packed-versus-gapped.                                                                                                                                                                              |
| The packed-layout guard control does not show the large deviation.                        | [`core/packed-guard-control`](../evidence/receipts/core/packed-guard-control/result.json), [`backend/cu130-packed-guard`](../evidence/receipts/backend/cu130-packed-guard/result.json)                                                                          | The guard copies the gapped operand before the backward call in the recorded test. This does not itself prove every production integration.                                                                                                                       |
| The native SM90 address path is causal for the captured prepass case.                     | [Native intervention measurements](../evidence/native/native-prepass-summary.json) and the publicly posted SASS excerpt in [issue #202035](https://github.com/pytorch/pytorch/issues/202035)                                                                    | Isolated byte-arithmetic widening reduces `D` relative-L2 `0.5726383726` to `4.3537051104e-8`, bitwise equal to packed vendor control across 6,365,184 rows. Both vendor controls retain 15 strict FP64 residuals; this is not a claim of exact FP64 correctness. |

## Dispatch, versions and layout

| Claim                                                       | Receipts                                                                                                                                                                                           | Limit                                                                                                           |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Explicit cuDNN reproduces on the listed cu126 H200 builds.  | [`versions/`](../evidence/receipts/versions/)                                                                                                                                                      | Discrete tested combinations only; no first-affected or fixed version established.                              |
| Default dispatch differs by wheel in the retained geometry. | [`backend/cu126-default`](../evidence/receipts/backend/cu126-default/result.json), [`backend/cu130-default`](../evidence/receipts/backend/cu130-default/result.json) and explicit backend controls | Specific build and input only. A pass under another backend does not validate cuDNN.                            |
| A concatenation backward can supply a gapped view.          | [`layout/cat-backward-view`](../evidence/receipts/layout/cat-backward-view/result.json)                                                                                                            | This case is small and does not cross 4 GiB by itself.                                                          |
| Width variants were explored.                               | [`layout/widths-cu126`](../evidence/receipts/layout/widths-cu126/result.json), [`layout/widths-cu130`](../evidence/receipts/layout/widths-cu130/result.json)                                       | Results are tied to H200 and these cuDNN builds; `kernel_unavailable` is not a pass.                            |
| Framework probes were limited.                              | [`frameworks/`](../evidence/receipts/frameworks/)                                                                                                                                                  | A result from one framework/build/layout does not establish that framework is generally affected or unaffected. |

## Sanitizer evidence

| Tool                       | Wrapped case     | Control          | Interpretation                                                                |
| -------------------------- | ---------------- | ---------------- | ----------------------------------------------------------------------------- |
| Compute Sanitizer memcheck | 0 errors         | 0 errors         | Does not detect this wrong-value read; reads remain within allocated backing. |
| Racecheck                  | 0 hazards        | 0 hazards        | No race distinction in these runs.                                            |
| Initcheck                  | 20,480 errors    | 20,480 errors    | Same count in both; not discriminating for the defect.                        |
| Synccheck                  | Did not complete | Did not complete | No conclusion. Raw multi-megabyte traces are not included.                    |

Short completed logs are in [`evidence/receipts/sanitizers/`](../evidence/receipts/sanitizers/).

## Security-impact observations kept bounded

The separate security campaign contains synthetic same-process planted-byte
and allocator-reuse observations. Its current reconciled description is:

- A0 recovered chosen synthetic bytes from the same process/allocation; this
  does not demonstrate another process, tenant, VM or GPU.
- A2 observed stale synthetic content after same-process storage reuse; the
  exact old/new pointer equality was not recorded.
- A5 adds and subtracts already-decoded `D` values. It is not DDP, NCCL,
  all-reduce of model gradients, or a network demonstration.
- E-T1 is a direct ABI/guard experiment whose frontend reachability remains
  unresolved. It does not establish a reachable out-of-bounds write.
- The original B2 batched-versus-single-example detector was confounded. Only
  the same-batch replacement is used for the numerical detector result here.

I have kept the raw scripts and payload receipts for these experiments out of
this repository. Their scope and release treatment are in
[`security-scope.md`](security-scope.md).

## Reconciliation rules

1. Prefer a later receipt over a historical summary only when the later receipt
   exists and has an environment identity.
2. Keep exact metric comparisons separate: backend-versus-FP64 is not
   packed-versus-gapped.
3. Keep dynamic hardware execution distinct from static cubin inspection.
4. A missing, incomplete or confounded run stays missing, incomplete or
   confounded; do not infer a result from adjacent runs.
5. Read `evidence/source-manifest.json` before modifying a receipt copy.
