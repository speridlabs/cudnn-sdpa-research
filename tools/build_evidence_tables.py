"""Build figure-ready CSV tables from the checked-in normalized receipts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypedDict


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "evidence" / "receipts"
FRESH = ROOT / "evidence" / "fresh-h200-20261026" / "result.json"
NATIVE = ROOT / "evidence" / "native" / "native-prepass-summary.json"
DATA = ROOT / "data"


class GradientRow(TypedDict):
    """A single tensor/batch comparison exported for plotting."""

    experiment_id: str
    comparison: str
    tensor: str
    batch: str
    relative_l2: str
    max_absolute: str
    finite: str
    torch: str
    cudnn_runtime: str
    gpu: str
    status: str


class ReceiptRow(TypedDict):
    """Identity and layout metadata for one experiment receipt."""

    experiment_id: str
    torch: str
    cuda_build: str
    cudnn_runtime: str
    cudnn_package: str
    gpu: str
    driver: str
    tested_utc: str
    status: str
    byte_span: str
    dtype: str
    shape: str
    dO_strides: str


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON receipt and require an object at its top level.

    Args:
        path: Receipt path relative to the repository.

    Returns:
        Parsed receipt object.

    Raises:
        ValueError: If the JSON top-level value is not an object.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Receipt must contain a JSON object: {path}")
    return value


def text(value: Any) -> str:
    """Return a stable CSV representation for a receipt value.

    Args:
        value: JSON-compatible receipt field.

    Returns:
        Empty text for null, compact JSON for containers, or `str(value)`.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def metric_rows(experiment_id: str, receipt: dict[str, Any]) -> list[GradientRow]:
    """Extract per-batch gradient metrics from known receipt schemas.

    Args:
        experiment_id: Stable path-like experiment identifier.
        receipt: Parsed normalized receipt.

    Returns:
        Rows for output, dQ, dK and dV comparisons. Unsupported schemas produce
        no metric rows rather than guessed values.
    """
    rows: list[GradientRow] = []
    base = {
        "torch": text(receipt.get("torch")),
        "cudnn_runtime": text(receipt.get("cudnn")),
        "gpu": text(receipt.get("gpu")),
        "status": text(receipt.get("status")),
    }
    regular = receipt.get("metrics")
    if isinstance(regular, dict):
        for tensor in ("output", "dq", "dk", "dv"):
            values = regular.get(tensor)
            if not isinstance(values, list):
                continue
            for batch, item in enumerate(values):
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "comparison": "backend_vs_cpu_fp64",
                        "tensor": tensor,
                        "batch": str(batch),
                        "relative_l2": text(item.get("relative_l2")),
                        "max_absolute": text(item.get("max_absolute")),
                        "finite": text(item.get("finite")),
                        **base,
                    }
                )
    paired = receipt.get("rel_l2_packed_vs_gapped_per_batch")
    if isinstance(paired, dict):
        for tensor, values in paired.items():
            if not isinstance(values, list):
                continue
            for batch, error in enumerate(values):
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "comparison": "packed_vs_gapped",
                        "tensor": text(tensor),
                        "batch": str(batch),
                        "relative_l2": text(error),
                        "max_absolute": "",
                        "finite": "",
                        **base,
                    }
                )
    return rows


def receipt_row(experiment_id: str, receipt: dict[str, Any]) -> ReceiptRow:
    """Extract stack identity and selected layout metadata.

    Args:
        experiment_id: Stable path-like receipt identifier.
        receipt: Parsed receipt JSON object.

    Returns:
        A row with stack, status and selected layout fields.
    """
    layout = receipt.get("layout")
    layout = layout if isinstance(layout, dict) else {}
    environment_path = ROOT / experiment_id / "environment.json"
    environment = load_json(environment_path) if environment_path.exists() else {}
    return {
        "experiment_id": experiment_id,
        "torch": text(receipt.get("torch")),
        "cuda_build": text(environment.get("cuda_build")),
        "cudnn_runtime": text(receipt.get("cudnn")),
        "cudnn_package": text(environment.get("cudnn_package")),
        "gpu": text(receipt.get("gpu")),
        "driver": text(environment.get("driver")),
        "tested_utc": text(
            environment.get("tested_utc", environment.get("tested_date"))
        ),
        "status": text(receipt.get("status")),
        "byte_span": text(
            layout.get("effective_do_span_bytes", layout.get("storage_bytes"))
        ),
        "dtype": text(layout.get("dtype")),
        "shape": text(layout.get("shape_bhsd")),
        "dO_strides": text(layout.get("do_stride")),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a deterministic CSV table with a stable column order.

    Args:
        path: Output path.
        rows: Rows with matching string-valued keys.

    Raises:
        ValueError: If rows have inconsistent columns.
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"CSV rows have inconsistent columns: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_tables() -> None:
    """Regenerate receipt index, gradient metrics and width tables."""
    receipt_paths = sorted(RECEIPTS.rglob("result.json"))
    if FRESH.exists():
        receipt_paths.append(FRESH)
    gradient_rows: list[GradientRow] = []
    receipt_rows: list[ReceiptRow] = []
    width_rows: list[dict[str, str]] = []
    for path in receipt_paths:
        receipt = load_json(path)
        experiment_id = path.parent.relative_to(ROOT).as_posix()
        if path == FRESH:
            experiment_id = "evidence/fresh-h200-20261026"
        gradient_rows.extend(metric_rows(experiment_id, receipt))
        receipt_rows.append(receipt_row(experiment_id, receipt))
        variants = receipt.get("results")
        if isinstance(variants, dict) and experiment_id.endswith(
            ("widths-cu126", "widths-cu130")
        ):
            for width, result in variants.items():
                if not isinstance(result, dict):
                    continue
                width_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "head_width": text(width),
                        "verdict": text(result.get("verdict", result.get("status"))),
                        "norm_ratio": text(result.get("ratio")),
                        "reachable_span_bytes": text(result.get("span_bytes")),
                        "torch": text(receipt.get("torch")),
                        "cudnn_runtime": text(receipt.get("cudnn")),
                        "gpu": text(receipt.get("gpu")),
                    }
                )
    write_csv(DATA / "gradient-metrics.csv", [dict(row) for row in gradient_rows])
    write_csv(DATA / "receipt-index.csv", [dict(row) for row in receipt_rows])
    write_csv(DATA / "width-results.csv", width_rows)
    build_native_table()
    print(
        f"Wrote {len(receipt_rows)} receipt rows, "
        f"{len(gradient_rows)} gradient rows, and native intervention data."
    )


def build_native_table() -> None:
    """Export selected native intervention metrics for later figure design.

    The source summary contains only reviewed numeric fields; it excludes model
    inputs, binary identifiers, launch metadata and private file paths.
    """
    summary = load_json(NATIVE)
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Native summary must contain a records list: {NATIVE}")
    rows: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Native summary records must be JSON objects")
        rows.append(
            {
                "experiment_id": text(record.get("experiment_id")),
                "source_receipt": text(record.get("source_receipt")),
                "batch": text(record.get("batch")),
                "status": text(record.get("status")),
                "relative_l2_vs_fp64": text(record.get("relative_l2_vs_fp64")),
                "mismatch_count_vs_strict_fp64": text(
                    record.get("mismatch_count_vs_strict_fp64")
                ),
                "bitwise_equal_to_packed_control": text(
                    record.get("bitwise_equal_to_packed_control")
                ),
                "gpu": text(record.get("gpu")),
                "torch": text(record.get("torch")),
            }
        )
    write_csv(DATA / "native-prepass-comparison.csv", rows)


if __name__ == "__main__":
    build_tables()
