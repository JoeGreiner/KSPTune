from __future__ import annotations

from pathlib import Path
from typing import Any

from .snapshot_analysis import analyze_snapshots

def analyze_matrix_properties(
    snapshot_directory: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    return analyze_snapshots(snapshot_directory, output_path, metadata_only=True)
