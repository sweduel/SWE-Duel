#!/usr/bin/env bash
# SWE-Duel dual-mode container entrypoint.
#
# The same repo image is used several ways:
#   1. OpenHands DockerWorkspace launches it as `<image> --host 0.0.0.0 --port 8000`
#      and expects the OpenHands agent-server HTTP API on that port.
#   2. mini-swe-agent's DockerEnvironment, the Codex / Claude Code harnesses, and
#      the validation-gate DockerExecutor launch it as
#      `<image> sleep infinity` / `<image> bash -c "<cmd>"` and just need a
#      normal shell command run. Codex (`codex exec`) and Claude Code
#      (`claude -p`) are then docker-exec'd into that container.
#
# We dispatch on the first argument: a leading `--` (or no args) means "run the
# agent-server"; anything else is executed verbatim. This keeps a single image
# valid for every harness and for the gates, so the agent and the gates share
# the exact same environment.
set -e

if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
    # No args, or first arg starts with '-': run the OpenHands agent-server.
    exec agent-server "$@"
fi

exec "$@"
