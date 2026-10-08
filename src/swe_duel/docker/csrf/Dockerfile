# gorilla/csrf is a pure-Go module (single dep: gorilla/securecookie).
# The suite is httptest-only and runs fully offline.
FROM golang:1.23-bookworm

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
COPY repos/csrf /workspace/
WORKDIR /workspace

# Warm the build/module cache so the first offline `go test ./...` is fast.
RUN go mod download && go build ./... && go test -count=1 -buildvcs=false ./... >/dev/null 2>&1 || true

# ── OpenHands agent-server (for the `openhands` harness) ─────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
COPY docker/install-agent-server.sh /tmp/install-agent-server.sh
RUN chmod +x /tmp/install-agent-server.sh && /tmp/install-agent-server.sh
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh

COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]
