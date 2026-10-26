# Source and receipt provenance

The two experiment scripts are copied byte-for-byte from the retained
reproducer source at revision `b44157a5c341410d46b3a234948a89357def2e84`.

| File                    | SHA-256                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `b2_fixed.py`           | `86ee1ce458c2a683e71e5b1ad8e31ea5b74428e271521a46f6a2ca8d6051be03` |
| `synthetic_boundary.py` | `83051019d2a33bb2f5a54ab0f86e55c8169d04c741d9351d0cb6498209413cda` |

The primary JSON and environment records are copied from the retained H200
`b2-fixed` run. The additional JSON receipts are copied from their
corresponding version-sweep runs. Their filenames are included in the JSON
content so readers can match versions and metrics.

`run_reproducer.py` is a new wrapper for this attachment. It validates the
requested runtime/device, checks free memory, captures `collect_env`, then
invokes the unchanged experiment script in a fresh process.
