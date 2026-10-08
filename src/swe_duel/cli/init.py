"""``swe-duel init`` — scaffold a working directory for running the arena.

Copies the bundled default configuration (arena.yaml, models.yaml,
repos/*.yaml) into ``<dir>/config/`` so the user owns an editable copy, and
creates ``<dir>/data/``. Everything else (Dockerfiles, fixtures, prompt
templates) is read in place from the installed package.

Typical end-user flow::

    mkdir my-arena && cd my-arena
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install swe-duel
    swe-duel init                      # scaffolds ./config + ./data
    $EDITOR config/models.yaml     # curate participant models
    export SWE_DUEL_OPENROUTER_API_KEY=...
    swe-duel setup repos               # clone target repos into ./repos
    swe-duel setup docker              # build the per-repo Docker images
    swe-duel doctor                    # validate the environment
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import swe_duel.config_defaults as _defaults


def _defaults_dir() -> Path:
    """On-disk location of the bundled config templates (package data)."""
    return Path(_defaults.__file__).resolve().parent


def scaffold(root: Path, *, force: bool = False) -> tuple[list[Path], list[Path]]:
    """Copy default configs into ``root/config``; create ``root/data``.

    Returns (created, skipped) file lists. Existing files are never
    overwritten unless ``force`` is set.
    """
    src = _defaults_dir()
    created: list[Path] = []
    skipped: list[Path] = []

    targets: list[tuple[Path, Path]] = [
        (src / "arena.yaml", root / "config" / "arena.yaml"),
        (src / "models.yaml", root / "config" / "models.yaml"),
    ]
    targets += [
        (rc, root / "config" / "repos" / rc.name)
        for rc in sorted((src / "repos").glob("*.yaml"))
    ]

    for from_path, to_path in targets:
        if to_path.exists() and not force:
            skipped.append(to_path)
            continue
        to_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(from_path, to_path)
        created.append(to_path)

    (root / "data").mkdir(parents=True, exist_ok=True)
    return created, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swe-duel-init",
        description=(
            "Scaffold a working directory: editable copies of the bundled "
            "arena/models config under ./config, plus the ./data output root."
        ),
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="target working directory (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing config files (default: keep them)",
    )
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    created, skipped = scaffold(root, force=args.force)

    for path in created:
        print(f"created: {path}")
    for path in skipped:
        print(f"kept existing: {path} (use --force to overwrite)")
    print(f"output directory ready: {root / 'data'}")
    print()
    print("Next steps:")
    print(f"  1. edit  {root / 'config' / 'models.yaml'}   (curate participant models)")
    print("  2. export SWE_DUEL_OPENROUTER_API_KEY=<key>")
    print("  3. swe-duel setup repos    (clone target repos into ./repos)")
    print("  4. swe-duel setup docker   (build per-repo Docker images)")
    print("  5. swe-duel doctor         (validate the environment)")
    print("  6. swe-duel-generate / swe-duel-tournament / swe-duel-match ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
