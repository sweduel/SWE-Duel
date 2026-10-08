# express is a plain JavaScript (no TypeScript) project that uses mocha as its
# test runner, with supertest/connect as test deps. We install deps at build
# time so the mocha suite runs offline.
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned express source into /workspace.
COPY repos/expressjs /workspace/
WORKDIR /workspace

# Install all dependencies (including devDependencies) so mocha + supertest
# are available offline at run time. Express ships no package-lock.json, so
# `npm install` (not `npm ci`) is the right tool.
RUN npm install

# The sandbox executor copies /workspace to a host-owned temp dir and mounts it
# back as root, so git refuses to operate on it ("dubious ownership"). Mark any
# workspace as safe so git-using checks run under the executor's copy model.
RUN git config --system --add safe.directory '*'

# ── OpenHands agent-server (for the `openhands` harness) ─────
# node:22-bookworm-slim ships Python 3.11; the agent-server needs >=3.12, so
# install-agent-server.sh provisions a standalone 3.12 venv at /opt/oh.
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
COPY docker/install-agent-server.sh /tmp/install-agent-server.sh
RUN chmod +x /tmp/install-agent-server.sh && /tmp/install-agent-server.sh
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh

# Dual-mode entrypoint: `--`-prefixed args → agent-server (OpenHands harness);
# anything else (sleep infinity / bash -c …) runs verbatim for mini-swe-agent
# and the validation-gate executor.
COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]
