# OWASP Java HTML Sanitizer is a Maven multi-module Java project. Containers
# now have network access, but we still pre-populate ~/.m2 so subsequent
# `mvn test` invocations are fast.
FROM maven:3.9-eclipse-temurin-17

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned Java sanitizer source into /workspace.
COPY repos/java-html-sanitizer /workspace/
WORKDIR /workspace

# Resolve all plugins/dependencies and compile the world so subsequent
# `mvn test` invocations are fast.
RUN mvn -B dependency:resolve dependency:resolve-plugins -q \
    && mvn -B test -q

# ── OpenHands agent-server (for the `openhands` harness) ─────
# This image ships no system Python; install-agent-server.sh uses uv to
# provision a standalone CPython 3.12 venv at /opt/oh for the agent-server.
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
