# Security scope and classification

The confirmed primary result is a numerical correctness defect: a tested
backward path can use the wrong values when a large strided address crosses a
32-bit byte-offset boundary. The effect can be silent and can corrupt gradients.
The experiments do not establish a remotely exploitable vulnerability.

## CWE discussion

- **CWE-190 — Integer Overflow or Wraparound:** best fit for the demonstrated
  arithmetic mechanism.
- **CWE-197 — Numeric Truncation Error:** a possible secondary description of
  the lost high product bits; final mapping should be decided with the vendor
  and maintainers.
- **CWE-125 — Out-of-bounds Read:** not confirmed as a conventional
  out-of-allocation read. The core wrong-address reads remained inside the
  allocated backing. Synthetic same-process experiments recovered chosen bytes
  from another logical region of that backing; the security significance
  depends on the application boundary and available observability.
- **CWE-787 — Out-of-bounds Write:** not demonstrated as reachable through the
  documented public call path. A separate direct-ABI experiment changed guard
  cells, but frontend reachability is unresolved and it is not an established
  public-API write primitive.

No cross-process, cross-tenant, cross-VM, remote-read, code-execution or
network-impact result is claimed. No CVE identifier is assigned here.

## Gated security experiments

The corrected evidence summary is included in the ledger,
but raw scripts and payload receipts are not part of this local release
candidate. They require a separate disclosure-scope review before export:

| Experiment | Bounded result                                                                                                | Excluded inference                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| A0         | 128/128 chosen synthetic bytes recovered in same process/allocation; named memcheck run recorded zero errors. | Not arbitrary external memory or another process.                         |
| A2         | 256 bytes of stale synthetic content observed after same-process allocator reuse.                             | Not another process's freed VRAM; pointer identity was not recorded.      |
| A5         | Arithmetic add/subtract of already decoded `D` values reproduced a payload.                                   | Not NCCL, DDP, network traffic or all-reduce of model gradients.          |
| E-T1       | Direct ABI/guard experiment altered guard cells 4–188 bytes before the logical delta.                         | Frontend reachability remains unresolved; no public-API OOB-write claim.  |
| D0         | No fault observed in the tested setup.                                                                        | Does not prove general availability or a demand-zero mapping explanation. |

## Why the classification stays conservative

The issue is worth fixing even without a security label: long GPU jobs may
continue with wrong gradients and produce degraded results without an obvious
runtime signal. The public impact claim should stay anchored to this numerical
effect. A security classification requires a demonstrated boundary violation
and threat model, not only a surprising address calculation or a same-process
synthetic payload.
