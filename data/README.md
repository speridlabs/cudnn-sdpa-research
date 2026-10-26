# Data for tables and future figures

The CSV files in this directory are generated from the normalized JSON
receipts under `evidence/receipts/` and the fresh run under
`evidence/fresh-h200-20261026/`.

Run `python tools/build_evidence_tables.py` from the repository root to rebuild:

- `gradient-metrics.csv` — one row per receipt, tensor and batch comparison.
- `receipt-index.csv` — experiment identity, normalized software/GPU/driver
  details where retained, status and layout geometry.
- `width-results.csv` — per-head-width norm ratios and availability outcomes.
- `native-prepass-comparison.csv` — selected per-batch native intervention
  metrics from `evidence/native/native-prepass-summary.json`.

Pointer/address fields are removed from normalized receipt copies before CSV
generation. The tool does not invent missing measurements; empty CSV cells
mean a field was not recorded in that receipt. Native intervention rows are
kept separate from the synthetic operator CSV because they came from a
different capture and answer a different comparison question.

The CSVs are raw numeric inputs for figure design. They do not include branded
figures or curves from the original run. That run also changed sample order, so
it does not isolate the kernel change by itself.
