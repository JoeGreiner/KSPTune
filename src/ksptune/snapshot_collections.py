from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .file_io import atomic_text_file

SNAPSHOT_COLLECTION_COLUMNS = [
    "snapshot_id",
    "solve_index",
    "matrix_key",
    "A",
    "matrix_metadata",
    "b",
    "x0",
    "meta",
    "matrix_write",
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

SNAPSHOT_PATH_COLUMNS = {
    "A": "matrix_file_path",
    "matrix_metadata": "matrix_metadata_file_path",
    "b": "right_hand_side_file_path",
    "x0": "initial_guess_file_path",
    "meta": "metadata_file_path",
}

SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def matrix_hash_path(matrix_path: Path) -> Path:
    return matrix_path.with_name(f"{matrix_path.name}.sha256")


def matrix_file_signature(matrix_path: Path) -> dict[str, int]:
    stat = matrix_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def read_matrix_hash(matrix_path: Path) -> str | None:
    hash_path = matrix_hash_path(matrix_path)
    if not hash_path.exists():
        return None
    try:
        cached = json.loads(hash_path.read_text(encoding="utf-8"))
        if not isinstance(cached, dict) or cached.get("signature") != matrix_file_signature(
            matrix_path
        ):
            return None
        matrix_hash = cached.get("sha256")
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(matrix_hash, str) or not SHA256_HEX_RE.fullmatch(matrix_hash):
        return None
    print(f"[snapshot_collections] using cached matrix hash: {hash_path}", flush=True)
    return matrix_hash.lower()


def write_matrix_hash(matrix_path: Path, matrix_hash: str, signature: dict[str, int]) -> None:
    hash_path = matrix_hash_path(matrix_path)
    try:
        with atomic_text_file(hash_path) as handle:
            json.dump({"sha256": matrix_hash, "signature": signature}, handle)
            handle.write("\n")
    except OSError as exc:
        print(
            f"[snapshot_collections] warning: could not write matrix hash cache {hash_path}: {exc}",
            flush=True,
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


def relative_path(path: Path | None, base_directory: Path) -> str:
    if path is None:
        return ""
    return os.path.relpath(path.resolve(), base_directory.resolve())


def resolve_snapshot_file_path(path_text: str, base_directory: Path) -> str:
    return str((base_directory / path_text).resolve()) if path_text else ""


def snapshot_collection_rows_from_directory(
    *,
    dump_directory: Path,
    base_directory: Path,
) -> list[dict[str, Any]]:
    if not dump_directory.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {dump_directory}")
    metadata_files = sorted(path for path in dump_directory.glob("s_*.txt") if path.is_file())
    if not metadata_files:
        raise ValueError(f"No snapshot metadata (s_*.txt) found in {dump_directory}")

    incomplete: list[str] = []
    rows: list[dict[str, Any]] = []
    for metadata_file in metadata_files:
        metadata = read_snapshot_metadata(metadata_file)
        required_paths = {key: metadata.get(key, "") for key in ("A", "matrix_metadata", "b", "x0")}
        missing_keys = [key for key, value in required_paths.items() if not value]
        if missing_keys:
            incomplete.append(f"{metadata_file.name}: missing {', '.join(missing_keys)}")
            continue

        referenced_paths = {key: metadata_file.parent / value for key, value in required_paths.items()}
        missing_files = [f"{key}={path}" for key, path in referenced_paths.items() if not path.is_file()]
        if missing_files:
            incomplete.append(f"{metadata_file.name}: missing referenced files: {', '.join(missing_files)}")
            continue

        matrix_metadata = read_snapshot_metadata(referenced_paths["matrix_metadata"])
        combined_metadata = {**matrix_metadata, **metadata}
        rows.append(
            {
                "snapshot_id": metadata.get("snapshot_id") or metadata_file.stem.removeprefix("s_"),
                "solve_index": metadata_int(metadata, "solve_index"),
                "matrix_key": metadata_text(combined_metadata, "matrix_key"),
                **{key: relative_path(path, base_directory) for key, path in referenced_paths.items()},
                "meta": relative_path(metadata_file, base_directory),
                "matrix_write": metadata_text(metadata, "matrix_write"),
                "rows": metadata_int(combined_metadata, "rows"),
                "cols": metadata_int(combined_metadata, "cols"),
                "rhs_size": metadata_int(combined_metadata, "rhs_size"),
                "mpi_size": metadata_int(combined_metadata, "mpi_size", 1),
                "matrix_nullspace_attached": metadata_int(combined_metadata, "matrix_nullspace_attached"),
                "transpose_nullspace_attached": metadata_int(combined_metadata, "transpose_nullspace_attached"),
                "near_nullspace_attached": metadata_int(combined_metadata, "near_nullspace_attached"),
                "nullspace_kind": metadata_text(combined_metadata, "nullspace_kind", "none"),
                "nullspace_field_index": metadata_int(combined_metadata, "nullspace_field_index", -1),
                "nullspace_block_size": metadata_int(combined_metadata, "nullspace_block_size"),
                "rhs_nullspace_component_removed": metadata_text(
                    combined_metadata, "rhs_nullspace_component_removed", "unknown"
                ),
            }
        )

    if incomplete:
        details = "\n".join(f"  - {item}" for item in incomplete)
        raise ValueError(f"Incomplete snapshot metadata:\n{details}")
    return sorted(rows, key=lambda row: int(row["solve_index"]))


def create_snapshot_collection_from_directory(
    dump_directory: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    directory = Path(dump_directory).resolve()

    collection_path = (
        Path(output_path) if output_path is not None else directory / "snapshot_collection.csv"
    )
    collection_path = collection_path.resolve()
    if collection_path.exists() and not overwrite:
        raise ValueError(f"Snapshot CSV already exists: {collection_path}")

    return write_snapshot_collection(
        load_snapshots_from_directory(directory),
        collection_path,
        relative_paths=True,
    )


def load_snapshots_from_directory(dump_directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(dump_directory).resolve()
    rows = snapshot_collection_rows_from_directory(
        dump_directory=directory,
        base_directory=directory,
    )
    return [normalize_snapshot_collection_row(row, directory) for row in rows]


def normalize_snapshot_collection_row(row: dict[str, Any], base_directory: Path) -> dict[str, Any]:
    missing = [key for key in ("solve_index", *SNAPSHOT_PATH_COLUMNS) if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Snapshot row missing required fields: {', '.join(missing)}")

    metadata_path = Path(resolve_snapshot_file_path(row["meta"], base_directory))
    snapshot_id = (
        row.get("snapshot_id")
        or read_snapshot_metadata(metadata_path).get("snapshot_id")
        or metadata_path.stem.removeprefix("s_")
    )
    matrix_nullspace_attached = int(float(row.get("matrix_nullspace_attached") or 0))
    transpose_nullspace_attached = int(float(row.get("transpose_nullspace_attached") or 0))
    near_nullspace_attached = int(float(row.get("near_nullspace_attached") or 0))
    nullspace_attached = (
        matrix_nullspace_attached or transpose_nullspace_attached or near_nullspace_attached
    )
    nullspace_kind = row.get("nullspace_kind") or ("unknown" if nullspace_attached else "none")

    return {
        "snapshot_id": snapshot_id,
        "solve_index": int(row.get("solve_index") or 0),
        "matrix_key": row.get("matrix_key") or "",
        **{
            key: resolve_snapshot_file_path(row[column], base_directory)
            for column, key in SNAPSHOT_PATH_COLUMNS.items()
        },
        "matrix_write": row.get("matrix_write") or "",
        "rows": int(float(row.get("rows") or 0)),
        "cols": int(float(row.get("cols") or 0)),
        "right_hand_side_size": int(float(row.get("rhs_size") or 0)),
        "mpi_size": int(float(row.get("mpi_size") or 1)),
        "matrix_nullspace_attached": matrix_nullspace_attached,
        "transpose_nullspace_attached": transpose_nullspace_attached,
        "near_nullspace_attached": near_nullspace_attached,
        "nullspace_kind": nullspace_kind,
        "nullspace_field_index": int(float(row["nullspace_field_index"]))
        if row.get("nullspace_field_index") not in (None, "")
        else -1,
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
    cached_hash = read_matrix_hash(path)
    if cached_hash is not None:
        return cached_hash

    digest = hashlib.sha256()
    signature = matrix_file_signature(path)
    print(f"[snapshot_collections] hashing matrix file: {path}", flush=True)
    chunk_size = 16 * 1024 * 1024
    total_bytes = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            total_bytes += len(chunk)
            digest.update(chunk)
    if matrix_file_signature(path) != signature:
        raise ValueError(f"Matrix file changed while hashing: {path}")
    print(
        f"[snapshot_collections] done hashing: {path} ({total_bytes / (1024 * 1024):.2f} MiB)",
        flush=True,
    )
    matrix_hash = digest.hexdigest()
    write_matrix_hash(path, matrix_hash, signature)
    return matrix_hash


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
        print(
            "[snapshot_collections] deduplication complete: no duplicate matrices detected",
            flush=True,
        )
        return {}

    paths_to_hash = list(
        dict.fromkeys(path_text for _, _, _, paths in candidate_groups for path_text in paths)
    )
    print(
        f"[snapshot_collections] hashing {len(paths_to_hash)} unique candidate files in parallel",
        flush=True,
    )

    path_hashes: dict[str, str] = {}
    max_workers = min(len(paths_to_hash), max(1, os.cpu_count() or 1), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path_text, matrix_hash in zip(
            paths_to_hash, executor.map(file_sha256, map(Path, paths_to_hash))
        ):
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
        print(
            "[snapshot_collections] deduplication complete: no duplicate matrices detected",
            flush=True,
        )
    return canonical_paths


def deduplicate_matrix_rows(rows: list[dict[str, Any]]) -> None:
    canonical_paths = canonical_matrix_paths(rows)
    for row in rows:
        matrix_path = row.get("A")
        if matrix_path in canonical_paths:
            row["A"] = canonical_paths[matrix_path]


def write_snapshot_collection(
    snapshots: list[dict[str, Any]],
    output_path: str | Path,
    *,
    relative_paths: bool = False,
    deduplicate_matrices: bool = False,
) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    aliases = {**SNAPSHOT_PATH_COLUMNS, "rhs_size": "right_hand_side_size"}
    rows = [
        {column: snapshot[aliases.get(column, column)] for column in SNAPSHOT_COLLECTION_COLUMNS}
        for snapshot in snapshots
    ]
    if deduplicate_matrices:
        deduplicate_matrix_rows(rows)
    if relative_paths:
        for row in rows:
            for column in SNAPSHOT_PATH_COLUMNS:
                row[column] = relative_path(
                    Path(row[column]) if row[column] else None, output.parent
                )
    with atomic_text_file(output) as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLLECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_resolved_snapshot_collection(
    snapshot_collection_path: str | Path,
    output_path: str | Path,
    *,
    deduplicate_matrices: bool = True,
) -> Path:
    return write_snapshot_collection(
        load_snapshot_collection(snapshot_collection_path),
        output_path,
        deduplicate_matrices=deduplicate_matrices,
    )


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

        for key in (
            "matrix_file_path",
            "matrix_metadata_file_path",
            "right_hand_side_file_path",
            "initial_guess_file_path",
        ):
            file_path_text = snapshot.get(key, "")
            if file_path_text and not Path(file_path_text).exists():
                file_path = Path(file_path_text)
                errors.append(f"solve_index {solve_index}: missing {key} {file_path}")
        metadata_file_path = snapshot.get("metadata_file_path", "")
        if metadata_file_path and not Path(metadata_file_path).exists():
            errors.append(
                f"solve_index {solve_index}: missing metadata_file_path {metadata_file_path}"
            )

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
