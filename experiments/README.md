# Experiment index

The experiment assets are split by question. Receipt identifiers are stable
directory names under `evidence/receipts/`; each result records its measured
stack and status. Reproduction code is in `reproduction/`.

| Family              | Question                                                                      | Entry point or receipt                                                              |
| ------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Core boundary       | Does a valid gapped BF16 `dO` that crosses 4 GiB change the backward result?  | `reproduction/b2_fixed.py`; `layout/same-batch-detector`                            |
| Boundary controls   | Does the below-boundary or packed layout show the same signal?                | `core/below-boundary`, `core/dense-control`, `core/packed-guard-control`            |
| Backend selection   | Does the selected or explicitly forced backend matter for the tested case?    | `reproduction/c6_default_backend_probe.py`; `backend/`                              |
| cuDNN versions      | Does the same geometry reproduce on the listed discrete versions?             | `versions/`                                                                         |
| Head widths         | Which widths were available and what norm ratio was recorded?                 | `reproduction/poc_width_variants.py`; `layout/widths-*`                             |
| Natural view        | Can ordinary concatenation backward deliver a gapped gradient view?           | `reproduction/poc_natural_trigger.py`; `layout/cat-backward-view`                   |
| Framework probes    | What did the limited JAX, Paddle and TE probes show?                          | `reproduction/framework_*.py`; `frameworks/`                                        |
| Sanitizers          | Which completed tool runs distinguished wrapped from control?                 | `sanitizers/` and short logs in `evidence/receipts/sanitizers/`                     |
| Native intervention | Does changing the identified byte arithmetic remove the captured discrepancy? | `data/native-prepass-comparison.csv`; `evidence/native/native-prepass-summary.json` |

## Scope notes

- `layout/cat-backward-view` is a small structural probe; it does not itself
  cross 4 GiB.
- `width-results.csv` records `kernel_unavailable` as unavailable, not passed.
- The valid same-batch detector compares the same inputs in both layouts. The
  older batched-versus-single-example comparison is not included as a validated
  result.
- Security-impact demonstration code (A0/A2/E-T1) is not included. The
  security assessment summarizes the boundaries without shipping those payload
  programs here.
- Synccheck did not complete. Its incomplete traces are not included and no
  verdict is drawn from them.
