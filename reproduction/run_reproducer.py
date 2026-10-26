"""Validate an environment, run the recorded SDPA reproducer, and save collect_env."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

BACKING_BYTES = 4_295_000_064


def runtime_cudnn_version() -> str:
    """Format PyTorch's cuDNN runtime version as major.minor.patch."""
    version = torch.backends.cudnn.version()
    if version is None:
        return "unavailable"
    return f"{version // 10000}.{(version // 100) % 100}.{version % 100}"


def release_prefix(version: str) -> tuple[str, str, str]:
    """Return the first three components of a runtime or package version."""
    parts = version.split(".")
    if len(parts) < 3:
        raise ValueError(f"version must have at least three components: {version}")
    return parts[0], parts[1], parts[2]


def main() -> int:
    """Check requested stack identities and invoke the preserved reproducer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--expect-torch", required=True)
    parser.add_argument("--expect-cuda", required=True)
    parser.add_argument("--expect-cudnn", required=True, help="runtime or package version")
    parser.add_argument("--expect-gpu", default="H200", help="required GPU name substring")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    actual = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": runtime_cudnn_version(),
        "gpu": torch.cuda.get_device_name(),
    }
    mismatches = []
    if actual["torch"] != args.expect_torch:
        mismatches.append(f"torch: expected {args.expect_torch}, found {actual['torch']}")
    if actual["cuda"] != args.expect_cuda:
        mismatches.append(f"CUDA: expected {args.expect_cuda}, found {actual['cuda']}")
    try:
        if release_prefix(actual["cudnn"]) != release_prefix(args.expect_cudnn):
            mismatches.append(f"cuDNN: expected {args.expect_cudnn}, found {actual['cudnn']}")
    except ValueError as error:
        parser.error(str(error))
    if args.expect_gpu.lower() not in actual["gpu"].lower():
        mismatches.append(f"GPU: expected name containing {args.expect_gpu}, found {actual['gpu']}")
    if mismatches:
        parser.error("environment mismatch: " + "; ".join(mismatches))

    free_bytes, _ = torch.cuda.mem_get_info()
    reserve_bytes = 2 * 1024**3
    if free_bytes < BACKING_BYTES + reserve_bytes:
        parser.error(
            f"need {BACKING_BYTES + reserve_bytes} free bytes for the test and reserve; "
            f"found {free_bytes}"
        )
    environment = subprocess.run(
        [sys.executable, "-m", "torch.utils.collect_env"],
        check=True,
        capture_output=True,
        text=True,
    )
    script = Path(__file__).with_name("b2_fixed.py")
    result = subprocess.run(
        [sys.executable, str(script), str(args.output), "--batch-stride", str(2**31 - 16384), "--seed", "1487"],
        check=False,
    )
    if not args.output.is_dir():
        return result.returncode
    (args.output / "collect_env.txt").write_text(environment.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
