"""Shared test fixtures and per-test result dumping.

Every test can call ``record("key", value)`` to stash procedure outputs.
After the session, each test file's results are written to
``./tmp/<test_file_stem>.json`` split into ``unit`` and ``integration``
buckets (integration = any test marked ``@pytest.mark.integration``).

Under pytest-xdist each worker writes its own ``…_gwN.json`` so concurrent
workers do not clobber each other.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest

import swe_duel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
# Fixtures (mock repo, gate-regression records, sample workspaces) ship inside
# the swe-duel package (src/swe_duel/validation/fixtures) so pip-installed users get
# them too; resolve them off the installed package, not the repo layout.
PACKAGE_ROOT = Path(swe_duel.__file__).resolve().parent
FIXTURES_DIR = PACKAGE_ROOT / "validation" / "fixtures"
MOCK_REPO_DIR = FIXTURES_DIR / "mock_repo"
DOCKER_DATA_DIR = PACKAGE_ROOT / "docker"
TMP_DIR = PROJECT_ROOT / "tmp"


def load_integration_models():
    """Load all models from config/models.yaml for integration-test parameterization.

    The arena-wide completion cap (arena.yaml ``agent_model.max_tokens``) is
    stamped onto every entry exactly like ``scripts/_common.py::setup`` does,
    so integration tests exercise the production request shape.

    Import lazily so unit tests do not require pydantic/yaml fully wired.
    """
    from swe_duel.config import load_arena_config, load_models_config

    models = load_models_config(
        CONFIG_DIR, max_tokens=load_arena_config(CONFIG_DIR).agent_model.max_tokens
    )
    return list(models.values())


def integration_model_id(mc) -> str:
    return mc.model_id.replace("/", "_")


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def mock_repo_dir() -> Path:
    return MOCK_REPO_DIR


@pytest.fixture
def docker_data_dir() -> Path:
    return DOCKER_DATA_DIR


# ── Result recording ───────────────────────────────────────


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj):
        try:
            return _jsonable(asdict(obj))
        except TypeError:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj))
        except TypeError:
            pass
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


_PAYLOAD_KEY = pytest.StashKey[dict]()
_RESULTS_KEY = pytest.StashKey[dict]()


@pytest.fixture
def record(request: pytest.FixtureRequest):
    """Stash arbitrary payload under ``request.node`` for later JSON dump."""
    payload: dict[str, Any] = {}
    request.node.stash[_PAYLOAD_KEY] = payload

    def _record(key: str, value: Any) -> None:
        payload[key] = _jsonable(value)

    return _record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):  # type: ignore[override]
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    if report.when != "call":
        return
    session = item.session
    if _RESULTS_KEY not in session.stash:
        session.stash[_RESULTS_KEY] = {}
    results = session.stash[_RESULTS_KEY]

    payload = item.stash[_PAYLOAD_KEY] if _PAYLOAD_KEY in item.stash else None
    is_integration = any(m.name == "integration" for m in item.iter_markers())
    bucket = "integration" if is_integration else "unit"

    file_stem = Path(str(report.fspath)).stem
    file_results = results.setdefault(file_stem, {"unit": [], "integration": []})
    file_results[bucket].append({
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "duration_s": report.duration,
        "longrepr": str(report.longrepr) if report.longrepr else None,
        "payload": payload if payload is not None else {},
    })


def _xdist_worker_suffix() -> str:
    """``_gw0``/``_gw1`` under xdist; empty for a serial run."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    return f"_{worker}" if worker else ""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _RESULTS_KEY not in session.stash:
        return
    results: dict = session.stash[_RESULTS_KEY]
    if not results:
        return
    TMP_DIR.mkdir(exist_ok=True)
    suffix = _xdist_worker_suffix()
    for file_stem, buckets in results.items():
        for bucket, entries in buckets.items():
            if not entries:
                continue
            out_path = TMP_DIR / f"{file_stem}_{bucket}{suffix}.json"
            out_path.write_text(json.dumps({"results": entries}, indent=2, default=str))
