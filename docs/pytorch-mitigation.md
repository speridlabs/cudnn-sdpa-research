# PyTorch-side mitigation candidate

The underlying address arithmetic belongs to cuDNN. The local PyTorch patch is
a defensive mitigation at the framework boundary; it does not modify or repair
the vendor kernel.

## Behavior

In the dense cuDNN SDPA backward path, the candidate checks the reachable byte
span of both saved output `O` and incoming gradient `dO`. If a strided layout
exceeds the supported byte span but its compact representation fits, the
candidate packs it. If even the compact representation is unsupported, it
fails explicitly. The normalized tensors must be used consistently for the
cuDNN descriptors, cache key, ragged offsets and execution arguments.

This is important because packing only `O` does not protect a separately gapped
`dO`, and a check after descriptor/cache construction can leave stale layout
identity in the execution plan. The patch is available as
[`0001-guard-cudnn-sdpa-backward-byte-spans.patch`](../patches/pytorch/0001-guard-cudnn-sdpa-backward-byte-spans.patch).

## Candidate provenance and validation

- Candidate commit: `a9675ccd04e279151116f724134fe64e13fd4a5f`.
- Base commit: `7ebdeade70eae42be89d92448a7bafd1f3f1a614`.
- Target: PyTorch issue [#202035](https://github.com/pytorch/pytorch/issues/202035).
- Current status: the issue is open, and I have not received a maintainer
  response. I have not submitted this patch as a pull request.
- Recorded focused checks: the patch branch reports lint passing on the five
  changed files, a compiled CPU metadata helper test, an H200 CUDA build, four
  CUDA regression tests, and safe-layout checks with and without the cache
  environment setting.
- The recorded safe-layout benchmark covers ordinary layouts. It does not
  replay the large defect-triggering layout in the patched build, run the full
  PyTorch suite, or establish performance for inputs that require packing.

I will update the patch against the agreed upstream base after hearing from the
maintainers. This mitigation is not an upstream acceptance or a permanent
substitute for a vendor fix.

## Tradeoffs

Packing adds memory traffic and temporary storage for affected layouts;
unsupported compact spans become a visible error. Ordinary layouts should not
be copied by the new guard. A vendor fix should widen every relevant address
product and intermediate and validate the descriptor limits; a final 64-bit
pointer addition alone cannot recover bits already discarded earlier.
