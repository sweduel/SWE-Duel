#!/usr/bin/env bash
# Install the OpenHands agent-server into an isolated venv inside a repo image
# whose base is NOT swe-duel-base (the Node/Go/C/Java images).
#
# Those base images ship Python 3.11 or no Python at all, but openhands-sdk
# requires Python >=3.12. We use `uv` (a single static binary) to provision a
# standalone CPython 3.12 and build the venv at /opt/oh — this works uniformly
# across Debian bookworm/trixie regardless of the distro Python. The venv is
# kept separate from the language toolchain so the heavy OpenHands dependency
# tree can't interfere with it. The `agent-server` console script is symlinked
# onto PATH so the shared swe-duel-entrypoint.sh can launch it in OpenHands mode.
#
# Requires: curl + ca-certificates, network at build time, and the shared
# docker/openhands-constraints.txt copied to /tmp/openhands-constraints.txt.
set -euo pipefail

export UV_INSTALL_DIR=/usr/local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh

# Standalone CPython 3.12 + a venv that uses it.
uv python install 3.12
uv venv --python 3.12 /opt/oh

VENV_PY=/opt/oh/bin/python
uv pip install --python "$VENV_PY" --no-cache \
    -c /tmp/openhands-constraints.txt \
    openhands-sdk==1.28.1 openhands-tools==1.28.1 \
    openhands-workspace==1.28.1 openhands-agent-server==1.28.1

ln -sf /opt/oh/bin/agent-server /usr/local/bin/agent-server
