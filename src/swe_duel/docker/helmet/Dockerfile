# helmet is a Node/TypeScript project. It uses Node's built-in test runner
# driven through `tsx`, with supertest/connect as test deps. We install deps at
# build time so tests run quickly, but containers now have network access if
# additional packages are needed.
FROM node:22-bookworm-slim

# node-zopfli (a transitive devDependency) is a native addon that needs a
# C/C++ toolchain and python3 to compile via node-gyp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        make \
        g++ \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned helmet source into /workspace.
COPY repos/helmet /workspace/
WORKDIR /workspace

# Install all dependencies (including devDependencies) so the test runner,
# tsx, and supertest are available offline at run time.
RUN npm ci

# The sandbox executor copies /workspace to a host-owned temp dir and mounts it
# back as root, so git refuses to operate on it ("dubious ownership"). helmet's
# source-files test shells out to `git ls-files`; mark any workspace as safe so
# that check runs under the executor's copy-and-mount model.
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
