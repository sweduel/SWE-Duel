# golang-jwt/jwt is a pure-Go module with no external dependencies, so
# `go test ./...` runs entirely from the standard library. The sandbox
# executor runs containers with `network_mode=none`; pre-fetching modules and
# pre-compiling the test binaries at build time keeps the suite fully offline.
FROM golang:1.23-bookworm

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned jwt source into /workspace.
COPY repos/jwt /workspace/
WORKDIR /workspace

# Warm the build/module cache so the first offline `go test ./...` does not
# need to download or recompile the standard test toolchain.
RUN go mod download && go build ./... && go test -count=1 ./... >/dev/null 2>&1 || true

# ── OpenHands agent-server (for the `openhands` harness) ─────
# golang:1.23-bookworm ships Python 3.11; the agent-server needs >=3.12, so
# install-agent-server.sh provisions a standalone 3.12 venv at /opt/oh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
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
