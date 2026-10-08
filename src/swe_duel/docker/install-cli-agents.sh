#!/usr/bin/env bash
# Install Codex CLI + Claude Code CLI into a repo image so the `codex` and
# `claude-code` harnesses can run *inside* the same container the validation
# gates use.
#
# Both CLIs ship as npm packages. Images that already have Node (node:*-bookworm
# variants) just run `npm install -g`. Everyone else gets a portable Node 22
# toolchain dropped into /usr/local (no apt repo needed — works on Debian
# bookworm/trixie, Ubuntu jammy, slim variants).
#
# Requires: curl + ca-certificates (+ xz-utils for the node tarball), network
# at build time.
set -euo pipefail

# Some slim bases forget xz-utils / ca-certificates; best-effort top-up when
# apt-get is available so the Node tarball extraction succeeds.
if ! command -v xz >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    apt-get update \
        && apt-get install -y --no-install-recommends xz-utils ca-certificates curl \
        && rm -rf /var/lib/apt/lists/* || true
fi

NODE_VERSION="${SWE_DUEL_NODE_VERSION:-22.14.0}"
CODEX_PKG="${SWE_DUEL_CODEX_PKG:-@openai/codex@0.144.5}"
CLAUDE_PKG="${SWE_DUEL_CLAUDE_PKG:-@anthropic-ai/claude-code@2.1.212}"

have_node() {
    command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1
}

if ! have_node; then
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64) node_arch="x64" ;;
        aarch64|arm64) node_arch="arm64" ;;
        *)
            echo "install-cli-agents.sh: unsupported arch: $arch" >&2
            exit 1
            ;;
    esac
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
        -o "$tmp/node.tar.xz"
    tar -xJf "$tmp/node.tar.xz" -C /usr/local --strip-components=1
    # Drop the trap's EXIT so later npm install failures keep the node install.
    trap - EXIT
    rm -rf "$tmp"
fi

# Global install so `codex` / `claude` land on PATH for docker exec.
npm install -g --omit=dev "$CODEX_PKG" "$CLAUDE_PKG"

# Sanity-check both binaries are resolvable (prints versions to build log).
codex --version || true
claude --version || true
