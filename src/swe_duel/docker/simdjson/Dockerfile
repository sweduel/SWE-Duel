# simdjson is a CMake-driven C++17 JSON library. gcc:14-bookworm ships g++ so
# we can build DEVELOPER_MODE (library + unit suite) at image-build time and
# re-run it fully offline for gate_existing_tests. Injected SWE-Duel tests compile
# with g++ -std=c++17 against the static lib (see languages/profiles/simdjson.yaml).
FROM gcc:14-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake make \
    && rm -rf /var/lib/apt/lists/*

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
COPY repos/simdjson /workspace/
WORKDIR /workspace

# simdjson's CMake configure fetches the CPM module plus a set of CPM source
# packages (simdjson-data, cxxopts, competition libs) at configure time. Bake
# both into a persistent CPM source cache and guard the vendored
# cmake/CPM.cmake bootstrap so a cached module is reused instead of
# re-downloaded — the sandbox runs every gate/scoring command with
# `--network none`, and the unpatched bootstrap's unconditional
# file(DOWNLOAD) fails offline. The guard is behavior-preserving when online
# (the cached module is the hash-verified one the warm configure below pulls).
ENV CPM_SOURCE_CACHE=/opt/cpm-cache
RUN sed -i 's|^file(DOWNLOAD|if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")\nfile(DOWNLOAD|' cmake/CPM.cmake \
    && sed -i 's|^include(${CPM_DOWNLOAD_LOCATION})|endif()\ninclude(${CPM_DOWNLOAD_LOCATION})|' cmake/CPM.cmake

# Configure + build a small acceptance-target subset once to warm toolchain
# caches and populate the CPM source cache. Cap -j2 so builds on high-nproc
# hosts do not OOM under resource-limited later gate runs either (the gate
# re-configures).
RUN cmake -S . -B /tmp/simdjson-b \
        -DSIMDJSON_DEVELOPER_MODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/simdjson-b -j2 \
        --target simdjson unicode_tests minify_tests padded_string_tests prettify_tests \
    && ctest --test-dir /tmp/simdjson-b \
        -R '^(unicode_tests|minify_tests|padded_string_tests|prettify_tests)$' \
        --output-on-failure \
    && rm -rf /tmp/simdjson-b

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
