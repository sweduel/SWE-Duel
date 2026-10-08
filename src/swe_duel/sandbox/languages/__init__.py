"""Per-language (and per-repo) adapters for test execution, parsing, linting,
and complexity.

The arena was originally Python/pytest only. Adding Go (`jwt`, `chi`), Node
(`helmet`, `expressjs`), Java (`java-html-sanitizer`, `java-jwt`), and C
(`cjson`, `libexpat`) repos means every language-coupled step — how injected
tests are placed and run, how test output is parsed into counts, how the
existing-test command is built, how lint runs, and how feature-test complexity
is measured — must vary by `RepoConfig.language`, and for some languages by
`RepoConfig.name` too (build tool, source layout, test runner).

A `LanguageAdapter` owns all of that. `get_adapter(language, repo_name)`
returns the right one; for Java/C/Node/Go it constructs a per-repo instance
whose profile (loaded from ``profiles/<repo_name>.yaml``) selects Maven vs
Gradle, glob vs cmake, tsx vs mocha, import paths, etc. Adding a new repo for
an already-supported language means dropping a YAML file into ``profiles/`` —
no edits to the adapter code. Each adapter also exposes `prompt_hints()` so
the Red/Blue prompt builder can interpolate repo-specific values without
duplicating repo knowledge.

Conventions for injected tests (feature/bug tests the harness runs against
agent-modified code):

  - python: the test file is written verbatim at its given filename in the repo
    root and run with `python -m pytest <file>`.
  - go: the test is written as a standalone package under `_swe-duel/` (which Go
    ignores for `./...` because the dir name starts with `_`) and run with
    `go test -buildvcs=false -json ./_swe-duel/`.
  - node (tsx/node:test, helmet): the test is written under `_swe-duel/*.test.ts`
    and run with `npx tsx --test <file>`. Counts come from the TAP summary.
  - node (mocha, expressjs): the test is written under `_swe-duel/*.test.js`
    and run with `npx mocha --reporter tap <file>`. Counts come from the TAP plan.
  - java (Maven / Gradle): the JUnit 4 class is placed in the module's test
    source set and run with the build tool's single-class filter.
  - c (glob / cmake): the injected C program is compiled against the library
    (or linked after a cmake rebuild) and run.
"""

from __future__ import annotations

from swe_duel.sandbox.languages.base import LanguageAdapter
from swe_duel.sandbox.languages.c_adapter import CAdapter
from swe_duel.sandbox.languages.go_adapter import GoAdapter
from swe_duel.sandbox.languages.java_adapter import JavaAdapter
from swe_duel.sandbox.languages.node_adapter import NodeAdapter
from swe_duel.sandbox.languages.python_adapter import PythonAdapter
from swe_duel.sandbox.languages.registry import get_adapter

__all__ = [
    "LanguageAdapter",
    "PythonAdapter",
    "GoAdapter",
    "NodeAdapter",
    "JavaAdapter",
    "CAdapter",
    "get_adapter",
]
