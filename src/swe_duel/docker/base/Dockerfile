FROM python:3.12-slim

# Pin pytest to the 8.x line. pytest 9 removed several long-deprecated private
# APIs (e.g. `_pytest.monkeypatch.notset`) that pinned target repos like
# flask 3.1.1 still reference in their test suites, so an unpinned `pytest>=8.0`
# silently pulls pytest 9 and makes every existing test error at collection —
# which the validation gates correctly report as a failure. Keep us on 8.x so
# the gates and agents see the repo's intended test environment.
RUN pip install --no-cache-dir \
    "pytest>=8.0,<9" \
    pytest-json-report>=1.5 \
    ruff>=0.4 \
    mypy>=1.10

# ── OpenHands agent-server ───────────────────────────────────
# Bake the OpenHands agent-server into the image so the `openhands` harness can
# run INSIDE this same image (DockerWorkspace launches it as
# `<image> --host 0.0.0.0 --port 8000`). The otel pins resolve a known
# upstream dependency conflict (see docker/openhands-constraints.txt).
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
RUN pip install --no-cache-dir \
    -c /tmp/openhands-constraints.txt \
    openhands-sdk==1.28.1 openhands-tools==1.28.1 \
    openhands-workspace==1.28.1 openhands-agent-server==1.28.1

# ── Codex + Claude Code CLIs (for the `codex` / `claude-code` harnesses) ──
# Both are npm packages; install-cli-agents.sh drops a portable Node 22 into
# /usr/local if the image has no node, then npm-installs the CLIs onto PATH.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh \
    && rm -f /tmp/install-cli-agents.sh

# Dual-mode entrypoint: `--`-prefixed args → agent-server (OpenHands harness);
# anything else (sleep infinity / bash -c … / codex exec / claude -p) runs
# verbatim for mini-swe-agent, Codex, Claude Code, and the validation gates.
COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]

WORKDIR /workspace
