# Reproduction guide

## Safety and resource requirements

This standalone test uses synthetic tensors; it does not read application data
or saved model state. The gapped BF16 allocation contains `4,295,000,064` bytes
(about `4.00003 GiB`); the wrapper requires at least about 6 GiB free to leave
a 2 GiB reserve. Run it only on a GPU and environment you own or are authorized
to use. Do not disrupt another workload.

## Recorded environment

The fresh receipt was captured on an NVIDIA H200 with Linux driver `595.91.07`,
PyTorch `2.14.1+cu126`, CUDA build `12.6`, cuDNN runtime identifier `91002`,
and `nvidia-cudnn-cu12` wheel `9.10.2.21`. The `collect_env` report also lists
system cuDNN 9.8 libraries under `/usr/lib`; the accompanying runtime identity
record shows the cuDNN 9.10.2.21 wheel paths loaded by the test process. Use the
runtime API and loaded-library record when attributing the result, not the
unqualified system-library list.

The reproducer was copied byte-for-byte from the retained issue attachment
source; provenance and source checksums are in
[`evidence/source-manifest.json`](../evidence/source-manifest.json).

## Run the primary case

Install an isolated PyTorch CUDA environment matching the requested values,
then run:

```bash
cd reproduction
python run_reproducer.py \
  --output /tmp/cudnn-sdpa-repro \
  --expect-torch '2.14.1+cu126' \
  --expect-cuda '12.6' \
  --expect-cudnn '9.10.2' \
  --expect-gpu H200
```

The launcher checks the installed versions, GPU model and available memory,
captures `torch.utils.collect_env`, then runs the same-batch comparison in a
fresh process. The lower-level scripts are:

- `run_reproducer.py` — environment checks and receipt output.
- `b2_fixed.py` — packed and gapped `dO` comparison on identical batch inputs.
- `synthetic_boundary.py` — fixed synthetic tensors and an independent CPU
  FP64 sanity reference.

## Expected result

The primary expected behavior is:

- Batch 0, whose reachable `dO` addresses remain below the boundary, matches
  between layouts.
- Batch 1, which crosses the boundary, differs for `dQ` and `dK` with relative
  L2 near `0.1285` and `0.1011`, respectively.
- `dV` matches between layouts.
- The packed result passes the CPU FP64 sanity threshold (about `0.0025`
  relative L2 or less in the retained run).

Exit code `0` means the expected signal was detected and the packed sanity
checks passed. Exit code `2` means the expected signal was absent or a sanity
check failed; record it as a negative result and preserve the environment
identity. Any other nonzero exit is an execution failure, not evidence for or
against the defect.

## Controls and related experiments

The retained receipt tree contains below-boundary, dense, packed-guard,
backend-selection, version, width, concatenation-view, framework and sanitizer
cases. Read [the evidence ledger](evidence-ledger.md) and each receipt before
making a claim from them. The version matrix has discrete tested points, not a
continuous release guarantee.

Do not use the old invalid B2 single-example comparison as a detector. It
changed the sample and loss normalization. `layout/same-batch-detector` is the
valid same-batch replacement.

## Rebuild the CSV data for design

After editing or adding normalized receipts, rebuild the machine-readable
tables with:

```bash
python tools/build_evidence_tables.py
```

The tables are source data for later figures and design work; this repository
does not include final branded raster charts. `data/README.md` documents the
columns and the transformations applied.
