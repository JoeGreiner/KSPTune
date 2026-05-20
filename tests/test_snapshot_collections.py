from __future__ import annotations

from pathlib import Path

import pytest

from ksptune.snapshot_collections import (
    create_snapshot_collection_from_directory,
    load_snapshot_collection,
    summarize_snapshot_collection,
    validate_snapshot_collection,
    write_resolved_snapshot_collection,
)


def write_snapshot(directory: Path, solve_index: int, *, with_metadata: bool = True) -> None:
    prefix = "rmtest_6x5x4_trimmed_veri_nosmall"
    file_stem = f"{prefix}__solve_{solve_index:06d}"
    (directory / f"{file_stem}__A.bin").write_bytes(b"A")
    (directory / f"{file_stem}__b.bin").write_bytes(b"b")
    (directory / f"{file_stem}__x0.bin").write_bytes(b"x0")
    if with_metadata:
        (directory / f"{file_stem}__metadata.txt").write_text(
            "\n".join(
                [
                    f"solve_index={solve_index}",
                    f"snapshot_file_prefix={prefix}",
                    f"A={file_stem}__A.bin",
                    f"b={file_stem}__b.bin",
                    f"x0={file_stem}__x0.bin",
                    "rows=12",
                    "cols=12",
                    "rhs_size=12",
                    "mpi_size=2",
                    "nullspace=1",
                    "row_ownership_ranges=0,6,12",
                    "rhs_ownership_ranges=0,6,12",
                    "x0_ownership_ranges=0,6,12",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


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


def test_create_snapshot_collection_fails_on_incomplete_group(tmp_path: Path) -> None:
    (tmp_path / "rmtest__solve_000000__A.bin").write_bytes(b"A")
    (tmp_path / "rmtest__solve_000000__b.bin").write_bytes(b"b")

    with pytest.raises(ValueError, match="Incomplete snapshot groups"):
        create_snapshot_collection_from_directory(tmp_path)


def test_create_snapshot_collection_without_metadata_uses_zero_unknowns(tmp_path: Path) -> None:
    write_snapshot(tmp_path, 2, with_metadata=False)

    collection_path = create_snapshot_collection_from_directory(tmp_path)
    snapshots = load_snapshot_collection(collection_path)

    assert snapshots[0]["rows"] == 0
    assert snapshots[0]["cols"] == 0
    assert snapshots[0]["right_hand_side_size"] == 0
    assert snapshots[0]["mpi_size"] == 1


def test_stale_working_directoryrelative_paths_are_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dump_directory = tmp_path / "benchmark/snapshots/ksp"
    dump_directory.mkdir(parents=True)
    write_snapshot(dump_directory, 0)
    collection_path = dump_directory / "snapshot_collection.csv"
    stale_prefix = Path("benchmark/snapshots/ksp")
    file_stem = "rmtest_6x5x4_trimmed_veri_nosmall__solve_000000"
    collection_path.write_text(
        "\n".join(
            [
                "solve_index,A,b,x0,meta,rows,cols,rhs_size,mpi_size,nullspace",
                ",".join(
                    [
                        "0",
                        str(stale_prefix / f"{file_stem}__A.bin"),
                        str(stale_prefix / f"{file_stem}__b.bin"),
                        str(stale_prefix / f"{file_stem}__x0.bin"),
                        str(stale_prefix / f"{file_stem}__metadata.txt"),
                        "12",
                        "12",
                        "12",
                        "2",
                        "1",
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = load_snapshot_collection(collection_path)

    assert snapshots[0]["matrix_file_path"] == str((dump_directory / f"{file_stem}__A.bin").resolve())
    assert validate_snapshot_collection(collection_path) == []


def test_create_snapshot_collection_scans_subdirectories(tmp_path: Path) -> None:
    first_directory = tmp_path / "run_a"
    second_directory = tmp_path / "run_b"
    first_directory.mkdir()
    second_directory.mkdir()
    write_snapshot(first_directory, 0)
    write_snapshot(second_directory, 0)

    collection_path = create_snapshot_collection_from_directory(tmp_path)

    assert validate_snapshot_collection(collection_path) == []
    assert len(load_snapshot_collection(collection_path)) == 2


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
