# auth0/java-jwt is a Gradle (not Maven) Java project. The `:java-jwt` module
# lives under lib/ (see settings.gradle). The repo ships a Gradle 6.9.2
# wrapper, which only supports JDK <= 15, and the build targets Java 8
# (testInJavaVersions = [8, 11, 17, 21]); JDK 11 is the newest the bundled
# Gradle accepts, so we use eclipse-temurin:11 here. We pre-resolve
# dependencies and compile + test the world at build time so subsequent
# `./gradlew :java-jwt:test` invocations are fast; the Gradle cache
# (~/.gradle) is baked in to keep offline runs quick.
FROM eclipse-temurin:11-jdk-jammy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

ENV GRADLE_OPTS="-Dorg.gradle.daemon=false -Dorg.gradle.console=plain"

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned java-jwt source into /workspace.
COPY repos/java-jwt /workspace/
WORKDIR /workspace

# Resolve all dependencies and compile + test the world once to warm the
# Gradle/dependencies caches. Allow a non-zero exit in case flaky network
# deps slip in — the real gate run will surface genuine failures.
RUN ./gradlew :java-jwt:test --console=plain >/dev/null 2>&1 || true

# ── OpenHands agent-server (for the `openhands` harness) ─────
# This image ships no system Python; install-agent-server.sh uses uv to
# provision a standalone CPython 3.12 venv at /opt/oh for the agent-server.
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
