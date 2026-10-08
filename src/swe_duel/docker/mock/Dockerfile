FROM swe-duel-base:latest

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
COPY tests/fixtures/mock_repo /workspace/
RUN pip install --no-cache-dir -e /workspace/
