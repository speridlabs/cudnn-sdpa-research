# Public repository contents

This repository contains the materials needed to inspect and reproduce the
reported numerical mismatch:

- Synthetic CUDA reproducer scripts and environment checks.
- Sanitized result receipts, CSV measurements, and source figures.
- The reviewed SASS excerpt and the PyTorch mitigation candidate.
- Documentation separating measured results from static analysis and open
  questions.

The repository does not include NVIDIA library binaries, full cubin dumps,
disassembler projects, or the separate security-impact payloads. No host
credentials, private addresses, or training-run artifacts are required by the
reproducer.

The PyTorch patch is a candidate mitigation and has not been submitted as a
pull request. NVIDIA has not confirmed the affected range or a fixed release.
See STATUS.md for the latest coordination state.

No license is selected for Speridlabs-authored material. Public visibility
allows readers to inspect and clone the repository; it does not grant reuse or
redistribution rights. See LICENSE-STATUS.md and the third-party license links
before distributing or modifying included material.
