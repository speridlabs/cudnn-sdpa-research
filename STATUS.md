# Research and disclosure status

Last reconciled: 2026-10-27.

## Repository

This research package is public at
https://github.com/speridlabs/cudnn-sdpa-research. It contains the synthetic
reproducers, sanitized measurement receipts, source figures, and a candidate
PyTorch-side mitigation. See the README for the run command and evidence index.

The repository does not select a license for Speridlabs-authored work. Public
visibility does not grant reuse rights; see LICENSE-STATUS.md.

## Reproduction and maintainer status

- The latest synthetic reproducer retained here was run on an H200 on
  2026-10-26 with PyTorch 2.14.1+cu126, CUDA build 12.6, and cuDNN runtime
  identifier 91002 / wheel 9.10.2.21.
- PyTorch issue #202035 is open. The issue requested a minimal reproducer and
  collect_env output on October 26. The repository now includes the runnable
  scripts, fresh H200 result, and environment records. No maintainer response
  has been received; the patch has not been submitted as a pull request.

## Vendor coordination

- NVIDIA PSIRT received the technical report and reproducer bundle on
  2026-10-25. No response has been received as of 2026-10-27.
- NVIDIA has not confirmed the affected range, fixed release, severity, CVE
  decision, or publication window.
- No CVE identifier is assigned.

The version matrix covers discrete tested releases and dynamic H200 results.
Static inspection of additional cubins is identified separately and does not
establish runtime impact on those architectures.
