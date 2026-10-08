FROM swe-duel-base:latest

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned flask source into /workspace so tests can be collected.
COPY repos/flask /workspace/
WORKDIR /workspace
RUN pip install --no-cache-dir -e /workspace/
