# jwtk/jjwt is a Maven multi-module Java project (api/impl/extensions/tdjar).
# Pre-populate ~/.m2 so subsequent `mvn test` invocations are fast offline.
FROM maven:3.9-eclipse-temurin-17

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
COPY repos/jjwt /workspace/
WORKDIR /workspace

# Resolve plugins/deps + run the full suite once to warm the local .m2 cache.
RUN mvn -B dependency:resolve dependency:resolve-plugins -q \
    && mvn -B test -q

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
