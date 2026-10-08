"""ArtifactLogger defense index / cache lookup tests."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    DefenseResult,
    TurnScore,
)


def _defense(
    challenge_id: str,
    blue_model: str,
    *,
    defense_id: str | None = None,
    blue_harness_id: str = "mini-swe-agent",
    timestamp: datetime | None = None,
    blue_composite: float = 1.0,
) -> DefenseResult:
    return DefenseResult(
        defense_id=defense_id or str(uuid.uuid4()),
        challenge_id=challenge_id,
        blue_model_id=blue_model,
        blue_fix=BlueFix(
            review_findings=[],
            fix_explanation="x",
            fix_diff="",
            modified_file_contents={},
            agent_trajectory=AgentTrajectory(
                steps=[],
                total_steps=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                model_id=blue_model,
                duration_seconds=0.0,
            ),
        ),
        score=TurnScore(
            s_regression=1.0,
            s_feature=1.0,
            s_bugfix=blue_composite,
            blue_composite=blue_composite,
            red_composite=1.0 - blue_composite,
            test_details={},
        ),
        duration_seconds=0.01,
        cost_usd=0.0,
        timestamp=timestamp or datetime.now(timezone.utc),
        blue_harness_id=blue_harness_id,
    )


def test_find_defense_miss(tmp_path: Path) -> None:
    al = ArtifactLogger(data_dir=tmp_path / "data")
    assert al.find_defense("c1", "blue") is None
    assert not al.has_defense("c1", "blue")


def test_log_defense_updates_index(tmp_path: Path) -> None:
    al = ArtifactLogger(data_dir=tmp_path / "data")
    d = _defense("c1", "blue")
    al.log_defense(d)
    assert al.has_defense("c1", "blue")
    found = al.find_defense("c1", "blue")
    assert found is not None
    assert found.defense_id == d.defense_id


def test_find_defense_picks_most_recent(tmp_path: Path) -> None:
    al = ArtifactLogger(data_dir=tmp_path / "data")
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    older = _defense("c1", "blue", defense_id="old", timestamp=t0)
    newer = _defense(
        "c1", "blue", defense_id="new", timestamp=t0 + timedelta(hours=1)
    )
    al.log_defense(older)
    al.log_defense(newer)
    found = al.find_defense("c1", "blue")
    assert found is not None
    assert found.defense_id == "new"


def test_find_defense_respects_harness(tmp_path: Path) -> None:
    al = ArtifactLogger(data_dir=tmp_path / "data")
    al.log_defense(_defense("c1", "blue", blue_harness_id="openhands"))
    assert not al.has_defense("c1", "blue", blue_harness_id="mini-swe-agent")
    assert al.has_defense("c1", "blue", blue_harness_id="openhands")
    hit = al.find_defense("c1", "blue", blue_harness_id="openhands")
    assert hit is not None
    assert hit.blue_harness_id == "openhands"


def test_index_bootstrap_from_existing_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    al_write = ArtifactLogger(data_dir=data)
    al_write.log_defense(_defense("c1", "blue", defense_id="pre"))

    # Fresh logger instance must discover the on-disk defense.
    al = ArtifactLogger(data_dir=data)
    assert al.has_defense("c1", "blue")
    found = al.find_defense("c1", "blue")
    assert found is not None
    assert found.defense_id == "pre"


def test_legacy_defense_defaults_harness(tmp_path: Path) -> None:
    """Files without blue_harness_id match mini-swe-agent (legacy)."""
    al = ArtifactLogger(data_dir=tmp_path / "data")
    d = _defense("c1", "blue", defense_id="legacy")
    path = al.log_defense(d)
    payload = json.loads(path.read_text())
    del payload["blue_harness_id"]
    path.write_text(json.dumps(payload))
    al.invalidate_defense_index()
    assert al.has_defense("c1", "blue", blue_harness_id="mini-swe-agent")
    assert not al.has_defense("c1", "blue", blue_harness_id="openhands")
