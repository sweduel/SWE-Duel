"""Adapter registry — resolve a language (+ optional repo profile) to an adapter."""

from __future__ import annotations

from swe_duel.sandbox.languages.base import LanguageAdapter
from swe_duel.sandbox.languages.c_adapter import CAdapter
from swe_duel.sandbox.languages.go_adapter import GoAdapter
from swe_duel.sandbox.languages.java_adapter import JavaAdapter
from swe_duel.sandbox.languages.node_adapter import NodeAdapter
from swe_duel.sandbox.languages.python_adapter import PythonAdapter

# Languages whose adapter is repo-independent are singletons. Java/C/Node/Go
# load a per-repo YAML profile from ``profiles/`` and are constructed on demand.
_ADAPTERS: dict[str, LanguageAdapter] = {
    "python": PythonAdapter(),
}


def get_adapter(language: str | None, repo_name: str | None = None) -> LanguageAdapter:
    """Return the LanguageAdapter for a language string (default: python).

    ``repo_name`` selects a per-repo YAML profile (under
    ``src/swe_duel/sandbox/languages/profiles/<repo_name>.yaml``) for Go/Java/C/Node.
    """
    key = (language or "python").strip().lower()
    repo = (repo_name or "").strip()
    if key in ("javascript", "typescript"):
        key = "node"
    if key == "java":
        return JavaAdapter(repo_name=repo)
    if key == "c":
        return CAdapter(repo_name=repo)
    if key == "node":
        return NodeAdapter(repo_name=repo)
    if key == "go":
        return GoAdapter(repo_name=repo)
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        raise ValueError(
            f"unsupported repo language {language!r}; "
            f"known: {sorted(set(_ADAPTERS) | {'node', 'c', 'java', 'go'})}"
        )
    return adapter
