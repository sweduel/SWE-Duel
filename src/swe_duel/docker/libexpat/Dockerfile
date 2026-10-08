# libexpat is a CMake-driven ANSI-C library whose sources live under expat/lib/.
# The image generates expat_config.h (a cmake output header that xmlparse.c /
# xmlrole.c / xmltok.c `#include`) into expat/lib/ at build time, then builds
# and runs the bundled `runtests` suite once to validate the toolchain. At run
# time the existing-test gate rebuilds from (possibly agent-modified) source
# and re-runs ctest fully offline.
FROM gcc:14-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake make \
    && rm -rf /var/lib/apt/lists/*

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned libexpat source into /workspace.
COPY repos/libexpat /workspace/
WORKDIR /workspace

# Generate expat_config.h into expat/lib/ (next to the sources that include
# it) and build+run the bundled test suite once to warm caches and validate.
RUN cmake -S expat -B /tmp/exb \
        -DEXPAT_BUILD_TESTS=ON \
        -DEXPAT_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/exb \
    && cp /tmp/exb/expat_config.h /workspace/expat/lib/expat_config.h \
    && (cd /tmp/exb && ctest --output-on-failure) \
    && rm -rf /tmp/exb

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
