# Figure source data

## README figures

- `address-wraparound.png`: diagram of the intended byte offset and the earlier location selected after 32-bit wraparound.
- `controlled-backward.png`: comparison from the H200 BF16 reproducer with fixed inputs and no optimizer update. Values are for one selected weight gradient, not a global gradient norm.

## Additional figure source data

The CSVs in [`data/`](../data/README.md) and normalized receipts in `evidence/` supply values for additional branded figures.

Suggested figure inputs:

| Figure idea                    | Data source                                                           | Keep the caption precise                                                                                                                                                                              |
| ------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Same-batch gradient comparison | `data/gradient-metrics.csv`, rows where `comparison=packed_vs_gapped` | The test changes `dO` layout; batch 0 is below the boundary and batch 1 crosses it.                                                                                                                   |
| Native intervention            | `data/native-prepass-comparison.csv`                                  | Relative-L2 is against the captured FP64 reference; the repaired result is also compared bitwise with packed cuDNN. Fifteen strict FP64 residuals remain in the byte intervention and packed control. |
| Version points                 | `data/receipt-index.csv` plus `docs/version-matrix.md`                | Discrete H200 observations; not every intervening release.                                                                                                                                            |
| Sanitizer comparison           | `evidence/receipts/sanitizers/*.log`                                  | Initcheck has equal errors in both cases; synccheck is incomplete.                                                                                                                                    |

The original run comparison also changed sample order, so it does not isolate
the kernel change. Review figure data and labels against their source receipts
before publication.
