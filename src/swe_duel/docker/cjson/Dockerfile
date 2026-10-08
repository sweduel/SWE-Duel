# cJSON is a single-file ANSI-C library with a zero-dependency Makefile. The
# sandbox runs containers with network disabled, so we build and run the bundled
# `cJSON_test` suite at image-build time to warm the toolchain and prove the
# sources compile cleanly. At run time `make test` rebuilds (incrementally) and
# executes the tests again fully offline.
FROM gcc:14-bookworm

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned cjson source into /workspace.
COPY repos/cjson /workspace/
WORKDIR /workspace

# Compile and run the bundled C test suite once to warm caches and validate.
RUN make test

# ── OpenHands agent-server (for the `openhands` harness) ─────
# gcc:14-bookworm ships Python 3.11; the agent-server needs >=3.12, so
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
