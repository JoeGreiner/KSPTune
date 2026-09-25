from __future__ import annotations

from pathlib import Path

import pytest

from ksptune.snapshot_collections import (
    create_snapshot_collection_from_directory,
    matrix_hash_path,
    load_snapshot_collection,
    summarize_snapshot_collection,
    validate_snapshot_collection,
    write_resolved_snapshot_collection,
)


def write_snapshot(
    directory: Path,
    solve_index: int,
    *,
    matrix_key: str | None = None,
    matrix_bytes: bytes = b"A",
    with_matrix_metadata: bool = True,
) -> None:
    if matrix_key is None:
        matrix_key = f"matrix_{solve_index:06d}"
    snapshot_prefix = "rmtest_6x5x4"
    snapshot_id = f"{snapshot_prefix}_{solve_index:06d}"
    matrix_file = directory / f"A_{matrix_key}.bin"
    matrix_metadata_file = directory / f"A_{matrix_key}.txt"
    right_hand_side_file = directory / f"b_{snapshot_id}.bin"
    initial_guess_file = directory / f"x0_{snapshot_id}.bin"
    solve_metadata_file = directory / f"s_{snapshot_id}.txt"

    if not matrix_file.exists():
        matrix_file.write_bytes(matrix_bytes)
    right_hand_side_file.write_bytes(b"b")
    initial_guess_file.write_bytes(b"x0")

    if with_matrix_metadata and not matrix_metadata_file.exists():
        matrix_metadata_file.write_text(
            "\n".join(
                [
                    f"matrix_key={matrix_key}",
                    f"matrix_id={matrix_key}",
                    f"A={matrix_file.name}",
                    "rows=12",
                    "cols=12",
                    "mpi_size=2",
                    "matrix_nullspace_attached=1",
                    "transpose_nullspace_attached=0",
                    "near_nullspace_attached=0",
                    "nullspace_kind=constant",
                    "nullspace_field_index=-1",
                    "nullspace_block_size=0",
                    "row_ownership_ranges=0,6,12",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    solve_metadata_file.write_text(
        "\n".join(
            [
                f"solve_index={solve_index}",
                f"snapshot_id={snapshot_id}",
                f"matrix_key={matrix_key}",
                f"matrix_id={matrix_key}",
                f"matrix_write={'written' if solve_index == 0 else 'reused'}",
                f"A={matrix_file.name}",
                f"matrix_metadata={matrix_metadata_file.name}",
                f"b={right_hand_side_file.name}",
                f"x0={initial_guess_file.name}",
                "rhs_size=12",
                "rhs_ownership_ranges=0,6,12",
                "x0_ownership_ranges=0,6,12",
                "rhs_nullspace_component_removed=yes",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_create_snapshot_collection_reuses_matrix_metadata(tmp_path: Path) -> None:
    write_snapshot(tmp_path, 0, matrix_key="emi_static_operator")
    write_snapshot(tmp_path, 5, matrix_key="emi_static_operator")

    collection_path = create_snapshot_collection_from_directory(tmp_path)

    errors = validate_snapshot_collection(collection_path)
    assert errors == []

    snapshots = load_snapshot_collection(collection_path)
    assert [snapshot["solve_index"] for snapshot in snapshots] == [0, 5]
    assert snapshots[0]["matrix_key"] == "emi_static_operator"
    assert snapshots[0]["matrix_file_path"] == snapshots[1]["matrix_file_path"]
    assert snapshots[0]["matrix_metadata_file_path"] == snapshots[1]["matrix_metadata_file_path"]
    assert snapshots[0]["right_hand_side_file_path"] != snapshots[1]["right_hand_side_file_path"]
    assert snapshots[0]["rows"] == 12
    assert snapshots[0]["nullspace_kind"] == "constant"
    assert snapshots[1]["matrix_write"] == "reused"


def test_create_snapshot_collection_requires_matrix_metadata(
    tmp_path: Path,
) -> None:
    write_snapshot(tmp_path, 0, with_matrix_metadata=False)

    with pytest.raises(ValueError, match="missing referenced files: matrix_metadata="):
        create_snapshot_collection_from_directory(tmp_path)


def test_create_snapshot_collection_from_directory(tmp_path: Path) -> None:
    write_snapshot(tmp_path, 0)
    write_snapshot(tmp_path, 5)

    collection_path = create_snapshot_collection_from_directory(tmp_path)

    assert collection_path.name == "snapshot_collection.csv"
    errors = validate_snapshot_collection(collection_path)
    assert errors == []

    snapshots = load_snapshot_collection(collection_path)
    assert [snapshot["solve_index"] for snapshot in snapshots] == [0, 5]
    assert snapshots[0]["rows"] == 12
    assert snapshots[0]["right_hand_side_size"] == 12
    assert snapshots[0]["nullspace"] == 1

    summary = summarize_snapshot_collection(collection_path)
    assert summary["snapshot_count"] == 2
    assert summary["mpi_sizes"] == [2]


def test_create_snapshot_collection_fails_on_missing_vector(tmp_path: Path) -> None:
    write_snapshot(tmp_path, 0)
    (tmp_path / "x0_rmtest_6x5x4_000000.bin").unlink()

    with pytest.raises(ValueError, match="missing referenced files: x0="):
        create_snapshot_collection_from_directory(tmp_path)


def test_create_snapshot_collection_ignores_subdirectories(tmp_path: Path) -> None:
    nested_directory = tmp_path / "run_a"
    nested_directory.mkdir()
    write_snapshot(nested_directory, 0)

    with pytest.raises(ValueError, match=r"No snapshot metadata \(s_\*\.txt\) found"):
        create_snapshot_collection_from_directory(tmp_path)


def test_write_resolved_snapshot_collection_uses_absolute_paths(tmp_path: Path) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_snapshot(dump_directory, 0)
    collection_path = create_snapshot_collection_from_directory(dump_directory)
    resolved_collection_path = tmp_path / "run/snapshot_collection.resolved.csv"

    write_resolved_snapshot_collection(collection_path, resolved_collection_path)

    resolved_snapshots = load_snapshot_collection(resolved_collection_path)
    assert Path(resolved_snapshots[0]["matrix_file_path"]).is_absolute()
    assert Path(resolved_snapshots[0]["right_hand_side_file_path"]).is_absolute()
    assert Path(resolved_snapshots[0]["initial_guess_file_path"]).is_absolute()


def test_write_resolved_snapshot_collection_deduplicates_identical_matrices(
    tmp_path: Path,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_snapshot(dump_directory, 0, matrix_bytes=b"same matrix")
    write_snapshot(dump_directory, 1, matrix_bytes=b"same matrix")
    collection_path = create_snapshot_collection_from_directory(dump_directory)

    resolved_collection_path = tmp_path / "run/snapshot_collection.resolved.csv"
    write_resolved_snapshot_collection(collection_path, resolved_collection_path)

    resolved_snapshots = load_snapshot_collection(resolved_collection_path)
    assert resolved_snapshots[0]["matrix_file_path"] == resolved_snapshots[1]["matrix_file_path"]
    assert resolved_snapshots[0]["right_hand_side_file_path"] != resolved_snapshots[1]["right_hand_side_file_path"]


def test_write_resolved_snapshot_collection_reuses_matrix_hash_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_snapshot(dump_directory, 0, matrix_bytes=b"same matrix")
    write_snapshot(dump_directory, 1, matrix_bytes=b"same matrix")
    collection_path = create_snapshot_collection_from_directory(dump_directory)

    first_resolved_collection_path = tmp_path / "run_a/snapshot_collection.resolved.csv"
    write_resolved_snapshot_collection(collection_path, first_resolved_collection_path)
    first_output = capsys.readouterr().out

    first_matrix_path = dump_directory / "A_matrix_000000.bin"
    second_matrix_path = dump_directory / "A_matrix_000001.bin"
    assert matrix_hash_path(first_matrix_path).exists()
    assert matrix_hash_path(second_matrix_path).exists()
    assert "hashing matrix file" in first_output

    second_resolved_collection_path = tmp_path / "run_b/snapshot_collection.resolved.csv"
    write_resolved_snapshot_collection(collection_path, second_resolved_collection_path)
    second_output = capsys.readouterr().out

    assert "using cached matrix hash" in second_output
    assert "hashing matrix file" not in second_output


def test_write_resolved_snapshot_collection_keeps_different_matrices(
    tmp_path: Path,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_snapshot(dump_directory, 0, matrix_bytes=b"matrix one")
    write_snapshot(dump_directory, 1, matrix_bytes=b"matrix two")
    collection_path = create_snapshot_collection_from_directory(dump_directory)

    resolved_collection_path = tmp_path / "run/snapshot_collection.resolved.csv"
    write_resolved_snapshot_collection(collection_path, resolved_collection_path)

    resolved_snapshots = load_snapshot_collection(resolved_collection_path)
    assert resolved_snapshots[0]["matrix_file_path"] != resolved_snapshots[1]["matrix_file_path"]


def test_write_resolved_snapshot_collection_can_skip_matrix_deduplication(
    tmp_path: Path,
) -> None:
    dump_directory = tmp_path / "dumps"
    dump_directory.mkdir()
    write_snapshot(dump_directory, 0, matrix_bytes=b"same matrix")
    write_snapshot(dump_directory, 1, matrix_bytes=b"same matrix")
    collection_path = create_snapshot_collection_from_directory(dump_directory)

    resolved_collection_path = tmp_path / "run/snapshot_collection.resolved.csv"
    write_resolved_snapshot_collection(
        collection_path,
        resolved_collection_path,
        deduplicate_matrices=False,
    )

    resolved_snapshots = load_snapshot_collection(resolved_collection_path)
    assert resolved_snapshots[0]["matrix_file_path"] != resolved_snapshots[1]["matrix_file_path"]
