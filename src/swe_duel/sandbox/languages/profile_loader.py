"""Load per-repo adapter profiles from ``profiles/<repo_name>.yaml``.

Adding a new repository for an already-supported language only requires
dropping a YAML file into this directory — no edits to the adapter classes.
Python repos need no profile (the adapter is repo-independent).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"  # sibling package dir


@lru_cache(maxsize=64)
def load_repo_profile(repo_name: str | None) -> dict[str, Any]:
    """Return the YAML profile dict for ``repo_name``, or ``{}`` if missing."""
    name = (repo_name or "").strip()
    if not name:
        return {}
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def profiles_dir() -> Path:
    """Return the on-disk profile directory (for tests / docs)."""
    return _PROFILES_DIR
