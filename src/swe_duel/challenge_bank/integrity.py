"""Challenge freshness/staleness checks.

A challenge is *fresh* iff its `repo_commit_sha` matches the current
`RepoConfig.commit` for that repo. Anything else is stale and should be
pruned before Blue evaluation.
"""

from __future__ import annotations

from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import RepoConfig
from swe_duel.models import ChallengeRecord


def verify_challenge_freshness(
    record: ChallengeRecord, repo_config: RepoConfig
) -> bool:
    """Return True iff record's repo commit matches the current config."""
    return record.repo_commit_sha == repo_config.commit


def prune_stale_challenges(
    store: ChallengeStore,
    repo_configs: dict[str, RepoConfig],
) -> int:
    """Delete challenges whose commit no longer matches config. Returns count pruned."""
    pruned = 0
    for record in store.query():
        repo_cfg = repo_configs.get(record.repo_name)
        if repo_cfg is None:
            # No config for this repo anymore — treat as stale.
            store.delete(record.challenge_id)
            pruned += 1
            continue
        if not verify_challenge_freshness(record, repo_cfg):
            store.delete(record.challenge_id)
            pruned += 1
    return pruned
