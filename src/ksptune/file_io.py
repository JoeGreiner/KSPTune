from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO


@contextmanager
def atomic_text_file(path: str | Path) -> Iterator[TextIO]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text_atomic(path: str | Path, text: str) -> None:
    with atomic_text_file(path) as handle:
        handle.write(text)
