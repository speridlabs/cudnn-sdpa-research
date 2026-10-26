# Tested versions, dispatch and framework scope

All dynamic GPU results summarized here were run on an NVIDIA H200. They are
discrete observations, not a continuous affected-version range. NVIDIA has not
confirmed the first affected release or a fixed release.

## cuDNN and PyTorch

| PyTorch build  | CUDA build | cuDNN package/runtime  | Backend exercised          | Result                                                          |
| -------------- | ---------: | ---------------------- | -------------------------- | --------------------------------------------------------------- |
| `2.14.1+cu126` |       12.6 | 9.10.2 / runtime 91002 | Explicit cuDNN             | Crossing layout reproduces. Fresh run retained 2026-10-26.      |
| `2.14.1+cu126` |       12.6 | 9.19.0.56 / 91900      | Explicit cuDNN             | Reproduces in the retained version sweep.                       |
| `2.14.1+cu126` |       12.6 | 9.20.0.48 / 92000      | Explicit cuDNN             | Reproduces in the retained version sweep.                       |
| `2.14.1+cu126` |       12.6 | 9.24.0.43 / 92400      | Explicit cuDNN             | Reproduces in the retained version sweep.                       |
| `2.14.1+cu126` |       12.6 | 9.26.1.1 / 92601       | Explicit cuDNN             | Reproduces in the retained version sweep.                       |
| `2.14.1+cu126` |       12.6 | 9.27.0.42 / 92700      | Explicit cuDNN             | Reproduces in the retained version sweep.                       |
| `2.11.0+cu130` |       13.0 | 9.19.0.56              | Explicit cuDNN             | Earlier H200 receipt reports reproduction.                      |
| `2.13.0+cu130` |       13.0 | 9.20.0.48              | Explicit cuDNN             | Earlier H200 receipt reports reproduction.                      |
| `2.14.1+cu130` |       13.0 | 9.24.0.43 / 92400      | Default and explicit cuDNN | Default selected cuDNN and reproduced in the recorded geometry. |

The full per-case measurements are in `evidence/receipts/versions/` and the
`backend/` directories. The six cu126 version-sweep cases retain normalized
driver/GPU/software identities. The earlier cu130 points are in the issue
reproducer result files and issue record; a separate driver receipt is not
bundled here for each of those earlier runs. Do not interpolate the table into
an untested range or state that every workload using these libraries is
affected.

## Backend dispatch

PyTorch's SDPA API may select among enabled implementations based on the input
and build. The recorded `2.14.1+cu130` default case selected cuDNN and failed;
the recorded `2.14.1+cu126` default case selected Flash for this geometry and
passed, while explicitly selecting cuDNN on that build failed. These are
geometry-specific dispatch observations, not universal wheel policy. Use the
receipt's backend-selection field and reproduce on the actual workload before
attributing a result.

Official references: [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [backend selector](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html).

## Framework probes

| Framework probe    | Recorded environment                                                                                             | Result                                                                          | What it does not show                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| JAX                | See [`jax-probe`](../evidence/receipts/frameworks/jax-probe/result.json) and `grad_jaxpr.txt` in source campaign | No relevant cuDNN attention custom call was detected in the recorded JAX trace. | Not a general claim that JAX cannot reach cuDNN or is unaffected in every configuration. |
| PaddlePaddle       | `3.3.1`, H200 probe                                                                                              | Tested gapped case matched its packed/FP64 sanity control.                      | One probe is not a complete version/backend matrix.                                      |
| Transformer Engine | `2.20.2`, PyTorch `2.14.1+cu130`, cuDNN runtime 92400, H200                                                      | Tested case matched the control; backend was not exposed by that TE version.    | Does not prove which internal backend ran or establish general non-impact.               |

These probes are included as bounded results because they help define what was
and was not tested. They are not endorsements or compatibility guarantees.

## Architecture evidence

- **Dynamic:** H200 only in the retained matrix.
- **Static binary inspection:** the matching instruction form was observed in
  selected cuDNN 9.20 cubins for SM80, SM86, SM90a, SM120 and SM121. This does
  not prove runtime reachability or reproduce the failure on those GPUs.
- **SM100:** a different address sequence was observed in the inspected path;
  it is neither classified as affected nor declared safe.
- Other GPUs, toolkit combinations, attention variants and releases remain
  unqualified.
