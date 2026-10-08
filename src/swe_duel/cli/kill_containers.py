"""Kill every running ``swe-duel-`` container (manual cleanup escape hatch).

Reaps only containers whose name contains the ``swe-duel-`` prefix — co-tenant
``minisweagent-*`` / ``agent-server-*`` containers are never touched.
"""

from __future__ import annotations

import docker


def main() -> int:
    client = docker.from_env()
    prefix = "swe-duel-"
    for container in client.containers.list():
        if prefix in (container.name or ""):
            print(f"Killing container: {container.name} (ID: {container.short_id})")
            container.kill()
            print(f"Container {container.name} killed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
