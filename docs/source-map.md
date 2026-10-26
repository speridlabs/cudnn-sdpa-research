# Source and provenance map

This repository combines four evidence streams while keeping their authority
and limitations visible:

| Evidence stream            | Source revision                                                                                       | Role in this package                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Receipt reconciliation     | `9496345f91397fa3722b08b320187dc60542f4ab`                                                            | Version/backend probes, normalized synthetic results and corrected security scope.            |
| Native kernel analysis     | `07d23257fd399d20d882a285d187dc7a162cba8b`                                                            | SM90 analysis and causal intervention metrics. Only selected non-sensitive values are copied. |
| Framework mitigation       | candidate `a9675ccd04e279151116f724134fe64e13fd4a5f`, base `7ebdeade70eae42be89d92448a7bafd1f3f1a614` | Fresh H200 reproducer, environment output and local mitigation patch candidate.               |
| Original reproducer source | `b44157a5c341410d46b3a234948a89357def2e84`                                                            | Historical source for reproducer scripts; not current narrative authority.                    |

Portable source identifiers, SHA-256 digests, destination digests and
transformations for copied files are in
[`evidence/source-manifest.json`](../evidence/source-manifest.json). The
manifest does not expose workstation or agent-workspace paths. The copied
result JSONs remove pointer/address-valued fields. The native summary contains
selected per-batch error metrics; it omits model inputs, input hashes, binary
identifiers and launch metadata. The standalone reproducer scripts are copied
from the retained package, with source revisions recorded in the manifest.

## Authority order

1. Use the latest receipt for the exact tested question.
2. Use the October 25 evidence reconciliation to correct earlier prose.
3. Use the October 24 native report for binary mechanism and intervention,
   while preserving its date and narrower scope.
4. Treat historical blog and disclosure drafts as narrative material only.

If an older dossier says a result is missing but a later exact receipt exists,
cite the later receipt. If a later receipt is missing, incomplete or
confounded, leave the claim unresolved.
