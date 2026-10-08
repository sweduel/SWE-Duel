# auth0/node-jsonwebtoken is plain JavaScript with mocha. Install deps at
# build time so the suite runs offline.
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
COPY repos/node-jsonwebtoken /workspace/
WORKDIR /workspace

# No package-lock.json ships, so `npm install` (not `npm ci`). Pulls mocha +
# chai + sinon into node_modules for offline use.
RUN npm install

# Mark any workspace as safe so git-using checks pass under the executor's
# host-owned bind mount ("dubious ownership").
RUN git config --system --add safe.directory '*'

# ── OpenHands agent-server (for the `openhands` harness) ─────
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
COPY docker/install-agent-server.sh /tmp/install-agent-server.sh
RUN chmod +x /tmp/install-agent-server.sh && /tmp/install-agent-server.sh
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh

COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]
