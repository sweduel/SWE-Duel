FROM swe-duel-base:latest

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
# Copy the pinned jinja source into /workspace so tests can be collected.
COPY repos/jinja /workspace/
WORKDIR /workspace

# Install jinja (editable) plus its test deps so the full pytest suite runs
# offline: MarkupSafe (runtime dep) and trio (test dep, pinned to the repo's
# requirements/tests.txt version).
RUN pip install --no-cache-dir -e /workspace/ \
    && pip install --no-cache-dir \
        "MarkupSafe>=2.0" \
        "trio==0.27.0"
