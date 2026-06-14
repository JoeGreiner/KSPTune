from __future__ import annotations

import csv
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SNAPSHOT_COLLECTION_COLUMNS = [
    "solve_index",
    "A",
    "b",
    "x0",
    "meta",
    "rows",
    "cols",
    "rhs_size",
    "mpi_size",
    "matrix_nullspace_attached",
    "transpose_nullspace_attached",
    "near_nullspace_attached",
    "nullspace_kind",
    "nullspace_field_index",
    "nullspace_block_size",
    "rhs_nullspace_component_removed",
]

DUMP_FILE_RE = re.compile(
    r"^(?P<prefix>.+)__solve_(?P<solve_index>[0-9]{6})__(?P<kind>A|b|x0)\.bin$"
)


def read_snapshot_metadata(metadata_file_path: str | Path) -> dict[str, str]:
    path = Path(metadata_file_path)
    if not path.exists():
        return {}

    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def metadata_int(metadata: dict[str, str], key: str, default: int = 0) -> int:
    value = metadata.get(key)
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def metadata_text(metadata: dict[str, str], key: str, default: str = "") -> str:
    value = metadata.get(key)
    if value in (None, ""):
        return default
    return value


def metadata_nullspace_kind(metadata: dict[str, str]) -> str:
    kind = metadata_text(metadata, "nullspace_kind", "")
    if kind:
        return kind
    return "unknown" if metadata_int(metadata, "nullspace", 0) else "none"


def infer_size_from_ownership_ranges(metadata: dict[str, str], key: str) -> int:
    raw = metadata.get(key, "")
    if not raw:
        return 0
    try:
        values = [int(item) for item in raw.split(",") if item != ""]
    except ValueError:
        return 0
    return values[-1] if values else 0


def infer_mpi_size_from_ownership_ranges(metadata: dict[str, str]) -> int:
    raw = metadata.get("row_ownership_ranges", "")
    if not raw:
        return metadata_int(metadata, "mpi_size", 1)
    values = [item for item in raw.split(",") if item != ""]
    return max(1, len(values) - 1)


def relative_path(path: Path | None, base_directory: Path) -> str:
    if path is None:
        return ""
    return os.path.relpath(path.resolve(), base_directory.resolve())


def resolve_snapshot_file_path(path_text: str, base_directory: Path) -> str:
    if path_text == "":
        return ""

    path = Path(path_text)
    if path.is_absolute():
        return str(path)

    snapshot_collectionrelative_path = base_directory / path
    if snapshot_collectionrelative_path.exists():
        return str(snapshot_collectionrelative_path.resolve())

    working_directoryrelative_path = Path.cwd() / path
    if working_directoryrelative_path.exists():
        return str(working_directoryrelative_path.resolve())

    return str(snapshot_collectionrelative_path)


def metadata_file_for_group(dump_directory: Path, prefix: str, solve_index: int) -> Path | None:
    metadata_file = dump_directory / f"{prefix}__solve_{solve_index:06d}__metadata.txt"
    return metadata_file if metadata_file.exists() else None


def find_snapshot_file_groups(dump_directory: str | Path) -> dict[tuple[Path, str, int], dict[str, Path]]:
    directory = Path(dump_directory)
    groups: dict[tuple[Path, str, int], dict[str, Path]] = {}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        match = DUMP_FILE_RE.match(path.name)
        if match is None:
            continue
        key = (path.parent.resolve(), match.group("prefix"), int(match.group("solve_index")))
        groups.setdefault(key, {})[match.group("kind")] = path
    return groups


def snapshot_collection_rows_from_directory(
    *,
    dump_directory: Path,
    base_directory: Path,
) -> list[dict[str, Any]]:
    if not dump_directory.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {dump_directory}")

    groups = find_snapshot_file_groups(dump_directory)
    if not groups:
        raise ValueError(f"No PETSc snapshot files found in {dump_directory}")

    incomplete: list[str] = []
    rows: list[dict[str, Any]] = []
    for (snapshot_directory, prefix, solve_index), files in sorted(groups.items(), key=lambda item: item[0]):
        missing = [kind for kind in ("A", "b", "x0") if kind not in files]
        if missing:
            incomplete.append(f"{prefix}__solve_{solve_index:06d}: missing {', '.join(missing)}")
            continue

        metadata_file = metadata_file_for_group(snapshot_directory, prefix, solve_index)
        metadata = read_snapshot_metadata(metadata_file) if metadata_file else {}
        row_count = metadata_int(metadata, "rows") or infer_size_from_ownership_ranges(
            metadata, "row_ownership_ranges"
        )
        column_count = metadata_int(metadata, "cols", row_count)
        right_hand_side_size = metadata_int(
            metadata, "rhs_size"
        ) or infer_size_from_ownership_ranges(metadata, "rhs_ownership_ranges")

        rows.append(
            {
                "solve_index": solve_index,
                "A": relative_path(files["A"], base_directory),
                "b": relative_path(files["b"], base_directory),
                "x0": relative_path(files["x0"], base_directory),
                "meta": relative_path(metadata_file, base_directory),
                "rows": row_count,
                "cols": column_count,
                "rhs_size": right_hand_side_size,
                "mpi_size": infer_mpi_size_from_ownership_ranges(metadata),
                "matrix_nullspace_attached": metadata_int(
                    metadata,
                    "matrix_nullspace_attached",
                    metadata_int(metadata, "nullspace", 0),
                ),
                "transpose_nullspace_attached": metadata_int(
                    metadata,
                    "transpose_nullspace_attached",
                    0,
                ),
                "near_nullspace_attached": metadata_int(metadata, "near_nullspace_attached", 0),
                "nullspace_kind": metadata_nullspace_kind(metadata),
                "nullspace_field_index": metadata_int(metadata, "nullspace_field_index", -1),
                "nullspace_block_size": metadata_int(metadata, "nullspace_block_size", 0),
                "rhs_nullspace_component_removed": metadata_text(
                    metadata,
                    "rhs_nullspace_component_removed",
                    "unknown",
                ),
            }
        )

    if incomplete:
        details = "\n".join(f"  - {item}" for item in incomplete)
        raise ValueError(f"Incomplete snapshot groups:\n{details}")

    return rows


def create_snapshot_collection_from_directory(
    dump_directory: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    directory = Path(dump_directory).resolve()

    collection_path = Path(output_path) if output_path is not None else directory / "snapshot_collection.csv"
    collection_path = collection_path.resolve()
    if collection_path.exists() and not overwrite:
        raise ValueError(f"Snapshot CSV already exists: {collection_path}")

    rows = snapshot_collection_rows_from_directory(
        dump_directory=directory,
        base_directory=collection_path.parent,
    )

    collection_path.parent.mkdir(parents=True, exist_ok=True)
    with collection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLLECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return collection_path


def load_snapshots_from_directory(dump_directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(dump_directory).resolve()
    scratch_collection_path = directory / "snapshot_collection.csv"
    rows = snapshot_collection_rows_from_directory(
        dump_directory=directory,
        base_directory=scratch_collection_path.parent,
    )
    return [normalize_snapshot_collection_row(row, scratch_collection_path.parent) for row in rows]


def normalize_snapshot_collection_row(row: dict[str, str], base_directory: Path) -> dict[str, Any]:
    matrix_path = row.get("A") or row.get("matrix_file_path") or row.get("mat_file") or ""
    right_hand_side_path = row.get("b") or row.get("right_hand_side_file_path") or row.get("rhs_file") or ""
    initial_guess_path = row.get("x0") or row.get("initial_guess_file_path") or ""
    metadata_path = row.get("meta") or row.get("metadata_file_path") or row.get("meta_file") or ""

    matrix_nullspace_attached = int(
        float(row.get("matrix_nullspace_attached") or row.get("nullspace") or 0)
    )
    transpose_nullspace_attached = int(float(row.get("transpose_nullspace_attached") or 0))
    near_nullspace_attached = int(float(row.get("near_nullspace_attached") or 0))
    nullspace_attached = (
        matrix_nullspace_attached
        or transpose_nullspace_attached
        or near_nullspace_attached
    )
    nullspace_kind = row.get("nullspace_kind") or (
        "unknown" if int(float(row.get("nullspace") or 0)) else "none"
    )

    return {
        "solve_index": int(row.get("solve_index") or 0),
        "matrix_file_path": resolve_snapshot_file_path(matrix_path, base_directory),
        "right_hand_side_file_path": resolve_snapshot_file_path(right_hand_side_path, base_directory),
        "initial_guess_file_path": resolve_snapshot_file_path(initial_guess_path, base_directory),
        "metadata_file_path": resolve_snapshot_file_path(metadata_path, base_directory),
        "rows": int(float(row.get("rows") or 0)),
        "cols": int(float(row.get("cols") or 0)),
        "right_hand_side_size": int(float(row.get("rhs_size") or row.get("right_hand_side_size") or 0)),
        "mpi_size": int(float(row.get("mpi_size") or 1)),
        "matrix_nullspace_attached": matrix_nullspace_attached,
        "transpose_nullspace_attached": transpose_nullspace_attached,
        "near_nullspace_attached": near_nullspace_attached,
        "nullspace_kind": nullspace_kind,
        "nullspace_field_index": int(float(row.get("nullspace_field_index") or -1)),
        "nullspace_block_size": int(float(row.get("nullspace_block_size") or 0)),
        "rhs_nullspace_component_removed": row.get("rhs_nullspace_component_removed") or "unknown",
        "nullspace": int(bool(nullspace_attached)),
        "raw": dict(row),
    }


def load_snapshot_collection(snapshot_collection_path: str | Path) -> list[dict[str, Any]]:
    path = Path(snapshot_collection_path).resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Snapshot CSV has no header: {path}")
        return [normalize_snapshot_collection_row(row, path.parent) for row in reader]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    print(f"[snapshot_collections] hashing matrix file: {path}", flush=True)
    chunk_size = 16 * 1024 * 1024
    total_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            total_bytes += len(chunk)
            digest.update(chunk)
    print(
        f"[snapshot_collections] done hashing: {path} ({total_bytes / (1024 * 1024):.2f} MiB)",
        flush=True,
    )
    return digest.hexdigest()


def canonical_matrix_paths(rows: list[dict[str, Any]]) -> dict[str, str]:
    matrix_groups: dict[tuple[int, int, int], list[str]] = {}
    for row in rows:
        matrix_path_text = str(row.get("A") or "")
        if not matrix_path_text:
            continue
        matrix_path = Path(matrix_path_text)
        try:
            file_size = matrix_path.stat().st_size
        except OSError:
            continue
        group_key = (int(row.get("rows") or 0), int(row.get("cols") or 0), file_size)
        matrix_groups.setdefault(group_key, []).append(matrix_path_text)

    candidate_groups: list[tuple[int, int, int, list[str]]] = []
    for (rows_count, cols_count, file_size), paths in matrix_groups.items():
        unique_paths = list(dict.fromkeys(paths))
        if len(unique_paths) >= 2:
            candidate_groups.append((rows_count, cols_count, file_size, unique_paths))

    print(
        f"[snapshot_collections] found {len(candidate_groups)} matrix groups with >=2 candidate files for deduplication",
        flush=True,
    )
    if not candidate_groups:
        print("[snapshot_collections] deduplication complete: no duplicate matrices detected", flush=True)
        return {}

    paths_to_hash = list(
        dict.fromkeys(
            path_text
            for _, _, _, paths in candidate_groups
            for path_text in paths
        )
    )
    print(
        f"[snapshot_collections] hashing {len(paths_to_hash)} unique candidate files in parallel",
        flush=True,
    )

    path_hashes: dict[str, str] = {}
    max_workers = min(len(paths_to_hash), max(1, os.cpu_count() or 1), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path_text, matrix_hash in zip(paths_to_hash, executor.map(file_sha256, map(Path, paths_to_hash))):
            path_hashes[path_text] = matrix_hash

    canonical_paths: dict[str, str] = {}
    for rows_count, cols_count, file_size, paths in candidate_groups:
        print(
            "[snapshot_collections] processing matrix group: "
            f"{rows_count}x{cols_count}, size {file_size} bytes, {len(paths)} candidate files",
            flush=True,
        )

        first_path_by_hash: dict[tuple[int, int, int, str], str] = {}
        for path_text in paths:
            matrix_hash = path_hashes[path_text]
            dedupe_key = (rows_count, cols_count, file_size, matrix_hash)
            canonical_paths[path_text] = first_path_by_hash.setdefault(dedupe_key, path_text)

    if not canonical_paths:
        print("[snapshot_collections] deduplication complete: no duplicate matrices detected", flush=True)
    return canonical_paths


def deduplicate_matrix_rows(rows: list[dict[str, Any]]) -> None:
    canonical_paths = canonical_matrix_paths(rows)
    for row in rows:
        matrix_path = row.get("A")
        if matrix_path in canonical_paths:
            row["A"] = canonical_paths[matrix_path]


def write_resolved_snapshot_collection(
    snapshot_collection_path: str | Path,
    output_path: str | Path,
    *,
    deduplicate_matrices: bool = True,
) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshot_collection(snapshot_collection_path)
    rows = [
        {
            "solve_index": snapshot["solve_index"],
            "A": snapshot["matrix_file_path"],
            "b": snapshot["right_hand_side_file_path"],
            "x0": snapshot["initial_guess_file_path"],
            "meta": snapshot["metadata_file_path"],
            "rows": snapshot["rows"],
            "cols": snapshot["cols"],
            "rhs_size": snapshot["right_hand_side_size"],
            "mpi_size": snapshot["mpi_size"],
            "matrix_nullspace_attached": snapshot["matrix_nullspace_attached"],
            "transpose_nullspace_attached": snapshot["transpose_nullspace_attached"],
            "near_nullspace_attached": snapshot["near_nullspace_attached"],
            "nullspace_kind": snapshot["nullspace_kind"],
            "nullspace_field_index": snapshot["nullspace_field_index"],
            "nullspace_block_size": snapshot["nullspace_block_size"],
            "rhs_nullspace_component_removed": snapshot["rhs_nullspace_component_removed"],
        }
        for snapshot in snapshots
    ]
    if deduplicate_matrices:
        deduplicate_matrix_rows(rows)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLLECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def validate_snapshot_collection(snapshot_collection_path: str | Path) -> list[str]:
    errors: list[str] = []
    path = Path(snapshot_collection_path)
    if not path.exists():
        return [f"Snapshot CSV does not exist: {path}"]

    try:
        snapshots = load_snapshot_collection(path)
    except Exception as exc:  # noqa: BLE001
        return [str(exc)]

    for snapshot in snapshots:
        solve_index = snapshot["solve_index"]

        for key in ("matrix_file_path", "right_hand_side_file_path", "initial_guess_file_path"):
            file_path = Path(snapshot[key])
            if not file_path.exists():
                errors.append(f"solve_index {solve_index}: missing {key} {file_path}")
        metadata_file_path = snapshot.get("metadata_file_path", "")
        if metadata_file_path and not Path(metadata_file_path).exists():
            errors.append(f"solve_index {solve_index}: missing metadata_file_path {metadata_file_path}")

    if not snapshots:
        errors.append(f"Snapshot CSV is empty: {path}")
    return errors


def summarize_snapshot_collection(snapshot_collection_path: str | Path) -> dict[str, Any]:
    snapshots = load_snapshot_collection(snapshot_collection_path)
    return summarize_snapshots(snapshots)


def summarize_snapshot_directory(snapshot_directory: str | Path) -> dict[str, Any]:
    snapshots = load_snapshots_from_directory(snapshot_directory)
    return summarize_snapshots(snapshots)


def summarize_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    solve_indices = [snapshot["solve_index"] for snapshot in snapshots]
    return {
        "snapshot_count": len(snapshots),
        "solve_index_min": min(solve_indices) if solve_indices else None,
        "solve_index_max": max(solve_indices) if solve_indices else None,
        "mpi_sizes": sorted({snapshot["mpi_size"] for snapshot in snapshots}),
        "matrix_shapes": sorted({(snapshot["rows"], snapshot["cols"]) for snapshot in snapshots}),
        "has_nullspace": any(bool(snapshot["nullspace"]) for snapshot in snapshots),
        "nullspace_kinds": sorted({snapshot["nullspace_kind"] for snapshot in snapshots}),
    }
