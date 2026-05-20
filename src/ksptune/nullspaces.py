from __future__ import annotations

from typing import Any

NULLSPACE_ACTIONS = {"matrix", "transpose", "near", "remove-rhs"}
NULLSPACE_ACTION_ALIASES = {
    "right": "matrix",
    "left": "transpose",
    "remove-right-component": "remove-rhs",
}


def parse_key_value_items(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid nullspace item: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid nullspace item: {item}")
        values[key] = value
    return values


def parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Nullspace {name} must be an integer: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"Nullspace {name} must be positive: {value}")
    return parsed


def parse_nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Nullspace {name} must be an integer: {value}") from exc
    if parsed < 0:
        raise ValueError(f"Nullspace {name} must be non-negative: {value}")
    return parsed


def parse_nullspace(nullspace: str | None) -> dict[str, Any]:
    text = (nullspace or "from-metadata").strip()
    if text == "" or text == "from-metadata":
        return {
            "mode": "from-metadata",
            "source": "from-metadata",
            "label": "from-metadata",
            "replay_options": ["-replay_nullspace", "from-metadata"],
        }
    if text == "none":
        return {
            "mode": "none",
            "source": "none",
            "label": "none",
            "replay_options": ["-replay_nullspace", "none"],
        }
    if text == "constant":
        return {
            "mode": "constant",
            "source": "explicit",
            "label": "constant",
            "replay_options": ["-replay_nullspace", "constant"],
        }

    parts = [part.strip() for part in text.split(",") if part.strip()]
    head = parts[0] if parts else ""
    options = parse_key_value_items(parts[1:])

    if head.startswith("field:"):
        field_index = parse_nonnegative_int(head.removeprefix("field:"), "field index")
        block_size = parse_positive_int(options.pop("block_size", "2"), "block size")
        if field_index >= block_size:
            raise ValueError(
                f"Nullspace field index must be smaller than block size: {field_index} >= {block_size}"
            )
        if options:
            raise ValueError("Unsupported field nullspace option(s): " + ", ".join(sorted(options)))
        return {
            "mode": "field",
            "source": "explicit",
            "label": f"field:{field_index}",
            "field_index": field_index,
            "block_size": block_size,
            "replay_options": [
                "-replay_nullspace",
                "field",
                "-replay_field_nullspace",
                str(field_index),
                "-replay_field_nullspace_block_size",
                str(block_size),
            ],
        }

    raise ValueError("Nullspace must be one of from-metadata, none, constant, or field:<index>.")

def nullspace_replay_options(nullspace: str | None) -> list[str]:
    return list(parse_nullspace(nullspace)["replay_options"])


def parse_nullspace_actions(
    actions: str | None,
    *,
    nullspace: str | None = None,
) -> dict[str, Any]:
    nullspace_configuration = parse_nullspace(nullspace)
    text = (actions or "default").strip()
    if text == "" or text == "default":
        label = (
            "metadata"
            if nullspace_configuration["mode"] == "from-metadata"
            else "matrix,transpose"
        )
        return {
            "mode": "default",
            "label": label,
            "replay_options": [],
        }
    if text == "metadata":
        return {
            "mode": "metadata",
            "label": "metadata",
            "replay_options": ["-replay_nullspace_actions", "metadata"],
        }
    if text == "none":
        return {
            "mode": "none",
            "label": "none",
            "replay_options": ["-replay_nullspace_actions", "none"],
        }
    if text == "all":
        return {
            "mode": "explicit",
            "label": "matrix,transpose,near,remove-rhs",
            "replay_options": ["-replay_nullspace_actions", "all"],
        }

    normalized: list[str] = []
    for raw_item in [item.strip() for item in text.split(",") if item.strip()]:
        item = NULLSPACE_ACTION_ALIASES.get(raw_item, raw_item)
        if item not in NULLSPACE_ACTIONS:
            raise ValueError(
                "Nullspace action must be one of "
                + ", ".join(sorted(NULLSPACE_ACTIONS | set(NULLSPACE_ACTION_ALIASES)))
                + f": {raw_item}"
            )
        if item not in normalized:
            normalized.append(item)

    label = ",".join(normalized) if normalized else "none"
    return {
        "mode": "explicit",
        "label": label,
        "replay_options": ["-replay_nullspace_actions", label],
    }


def nullspace_action_replay_options(
    actions: str | None,
    *,
    nullspace: str | None = None,
) -> list[str]:
    return list(parse_nullspace_actions(actions, nullspace=nullspace)["replay_options"])
