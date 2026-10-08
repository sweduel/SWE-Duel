FROM swe-duel-base:latest

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned sqlalchemy source into /workspace.
COPY repos/sqlalchemy /workspace/
WORKDIR /workspace

# Pure-Python install: swe-duel-base is slim and has no gcc/cython toolchain.
# DISABLE_SQLALCHEMY_CEXT skips optional C extensions so `pip install -e`
# succeeds. The `asyncio` optional-dep pulls greenlet (needed by sqlalchemy's
# testing plugin). `tests` lives under [dependency-groups] not
# [project.optional-dependencies] in this release, so install pytest-xdist
# (and keep swe-duel-base's pytest 8.x) separately.
ENV DISABLE_SQLALCHEMY_CEXT=1
RUN pip install --no-cache-dir -e ".[asyncio]" \
    && pip install --no-cache-dir "pytest-xdist>=3.0"
