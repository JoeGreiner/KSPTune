from __future__ import annotations

import pytest

from ksptune.nullspaces import (
    nullspace_action_replay_options,
    nullspace_replay_options,
    parse_nullspace,
    parse_nullspace_actions,
)


def test_parse_none_nullspace() -> None:
    assert parse_nullspace("none") == {
        "mode": "none",
        "source": "none",
        "label": "none",
        "replay_options": ["-replay_nullspace", "none"],
    }


def test_parse_from_metadata_nullspace() -> None:
    assert parse_nullspace(None) == {
        "mode": "from-metadata",
        "source": "from-metadata",
        "label": "from-metadata",
        "replay_options": ["-replay_nullspace", "from-metadata"],
    }


def test_parse_constant_nullspace() -> None:
    assert nullspace_replay_options("constant") == ["-replay_nullspace", "constant"]


def test_parse_field_nullspace() -> None:
    assert nullspace_replay_options("field:1,block_size=3") == [
        "-replay_nullspace",
        "field",
        "-replay_field_nullspace",
        "1",
        "-replay_field_nullspace_block_size",
        "3",
    ]


def test_parse_field_nullspace_rejects_invalid_field() -> None:
    with pytest.raises(ValueError, match="field index must be smaller"):
        parse_nullspace("field:2,block_size=2")


def test_parse_field_nullspace_rejects_targets() -> None:
    with pytest.raises(ValueError, match="Unsupported field nullspace option"):
        parse_nullspace("field:1,targets=global")


def test_parse_default_nullspace_actions() -> None:
    assert parse_nullspace_actions(None, nullspace="from-metadata")["label"] == "metadata"
    assert parse_nullspace_actions(None, nullspace="constant")["label"] == "matrix,transpose"


def test_parse_explicit_nullspace_actions_with_aliases() -> None:
    assert nullspace_action_replay_options("right,left,near,remove-right-component") == [
        "-replay_nullspace_actions",
        "matrix,transpose,near,remove-rhs",
    ]


def test_parse_nullspace_actions_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="Nullspace action must be one of"):
        parse_nullspace_actions("matrix,bad")
