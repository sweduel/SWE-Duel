# Adding a New Repository to SWE-Duel

This guide walks a human developer through adding a **new target repository**
to the Adversarial Coding Agent Arena. SWE-Duel supports five languages today
(Python, Go, Node/JavaScript, Java, C); adding a repo for one of those
languages is a ~6-step config change. Adding a **new language** is a much
larger change (new `LanguageAdapter`, new prompt guide, new parser) and is
**out of scope for this guide** — it is hardcoded for now, and the five
languages are sufficient for the project.

> Prerequisites: Docker daemon running, the project's committed `venv/`
> activated (`./venv/bin/python`), and `make install` already run once.

---

## TL;DR — the six files you will touch

| Step | File | What you add |
|------|------|--------------|
| 1 | `config/repos/<name>.yaml` | repo metadata + test command + docker image name |
| 2 | `src/swe_duel/docker/<name>/Dockerfile` | image that bakes the toolchain + deps + source |
| 3 | `src/swe_duel/sandbox/languages/profiles/<name>.yaml` | per-repo adapter profile (Java/C/Node/Go only; Python skips) |
| 4 | `tests/test_repo_containers.py` | a container test class that builds + runs the suite |
| 5 | `tests/test_config.py` | update the repo-count assertion |

There is **no build script to edit**: `swe-duel setup docker --only <name>`
discovers `src/swe_duel/docker/<name>/Dockerfile` from the repo config, stages the
canonical build context (docker/ + mock repo + the clone), and builds
`swe-duel-<name>:latest`. Then run `make setup-repos`, `swe-duel setup docker --only
<name>`, and the new test. The rest of this guide explains each step in
detail, with a concrete running example:
**adding a second Python repo** (the simplest case) and then the per-language
twists for Go / Node / Java / C.

---

## Step 0 — Pick the repo and pin a commit

SWE-Duel pins every repo to an exact tag or commit so challenge generation is
reproducible. Find a release tag that:

- builds and tests **fully offline** (no network calls during the test run),
- has a test suite that runs in **under ~5 minutes** in a container with
  `--network none`, and
- does not require external services (databases, brokers, live HTTP servers)
  that the sandboxed gate container cannot provide.

> Avoid repos whose test suites spin up real servers or make network calls
> even with a "not network" marker — those tests hang inside the gate's
> resource-limited container and there is no clean way to skip them
> per-repo. (The Python **encode/httpx** library was rejected for this reason:
> large parts of its suite start uvicorn servers and hang.)

Get the tag:

```bash
git ls-remote --tags https://github.com/<org>/<repo> | tail
```

Use a tag name as the `commit` value (e.g. `"v5.3.0"`, `"3.1.6"`,
`"R_2_8_2"`). `swe-duel setup repos` clones with
`git clone --depth=1 --branch <commit>`, so the value must be a tag or branch
name that `git clone --branch` accepts.

---

## Step 1 — Create `config/repos/<name>.yaml`

One file per repo, under `config/repos/`. The `name` **must equal** the
directory name the repo clones into under `repos/` (i.e. the `<name>` in the
filename) — `swe-duel setup repos` derives the clone target from `name`,
not the filename, but keeping them identical avoids confusion and is what
every existing repo does.

### Minimal Python repo

Copy `config/repos/jinja.yaml` as a template:

```yaml
repo:
  name: jinja
  url: https://github.com/pallets/jinja
  commit: "3.1.6"
  language: python
  # Pure Python, runs fully offline against baked-in test resources.
  test_command: "python -m pytest tests/ -x -q --tb=short"
  docker_image: swe-duel-jinja
```

### Full field reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Repo slug; clones into `repos/<name>/`. Must match the YAML filename stem. |
| `url` | yes | `https://github.com/<org>/<repo>` clone URL. |
| `commit` | yes | Tag or branch name to pin (quoted string). |
| `language` | yes | One of `python`, `go`, `node` (or `javascript`/`typescript`, both map to the Node adapter), `java`, `c`. |
| `test_command` | yes | The shell command that runs the repo's own test suite. The validation gates execute this verbatim inside the Docker image. It must run **fully offline**. |
| `docker_image` | yes | Image name, conventionally `swe-duel-<name>`. `swe-duel setup docker` builds `<docker_image>:latest`. |
| `preserve_paths` | no | List of workspace subpaths (relative to `/workspace`) whose **image-baked** contents must show through the bind mount via an anonymous volume. Use for Node repos so `node_modules` (installed at image build time) is visible to the agent even though the host clone has none. See `helmet.yaml` / `expressjs.yaml`. |
| `login_shell` | no | `true` (default) uses `bash -lc`; `false` uses `bash -c`. Set `false` for Go repos: the golang image puts the toolchain on PATH via `/usr/local/go/bin`, and a login shell re-sources `/etc/profile` and resets PATH, dropping `go`. See `jwt.yaml` / `chi.yaml`. |
| `exclude_paths` | no | List of repo-relative glob patterns (or prefix strings) to exclude from the challenge diff/snapshot, in addition to the global `_DEFAULT_EXCLUDE`. Use for repo-specific build artifacts an in-tree build may generate. See `libexpat.yaml` and the "Keeping the diff clean" section below. |

---

## Step 2 — Create `docker/<name>/Dockerfile`

Every repo image does three things: (1) install the language toolchain, (2)
copy the pinned source into `/workspace`, (3) pre-build/warm deps so the
test suite runs offline. The build context is **always the project root**
(not `docker/`), because the Dockerfiles COPY shared files from `docker/`.

Pick the template that matches the repo's language and build system.

### Python — extends `swe-duel-base:latest`

`swe-duel-base` already has Python 3.12, pytest (pinned `>=8,<9` — do not change
this; pytest 9 breaks repos that use removed private pytest APIs), ruff,
mypy, and the OpenHands agent-server. You only add the source + the repo's
test deps.

```dockerfile
FROM swe-duel-base:latest

# Build context root is staged by `swe-duel setup docker` (docker/ + tests/fixtures/ + repos/).
COPY repos/<name> /workspace/
WORKDIR /workspace

# Install the package (editable) + any test-only deps the suite needs.
# Pin test deps to the versions the repo's requirements file declares so the
# offline suite behaves as upstream intends.
RUN pip install --no-cache-dir -e /workspace/ \
    && pip install --no-cache-dir \
        "<test-dep-1>==<version>" \
        "<test-dep-2>==<version>"
```

> Look at the repo's `requirements/tests.txt` / `pyproject.toml`
> `[project.optional-dependencies]` to find the test deps. Install the
> package with the extras it needs for its own tests, e.g.
> `-e ".[brotli,cli,http2]"`.

### Go — `golang:<ver>-bookworm` + agent-server block

```dockerfile
FROM golang:1.23-bookworm

COPY repos/<name> /workspace/
WORKDIR /workspace

# Warm the module/build cache so the first offline `go test ./...` is fast.
RUN go mod download && go build ./... && go test -count=1 ./... >/dev/null 2>&1 || true

# ── OpenHands agent-server (for the `openhands` harness) ─────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
COPY docker/install-agent-server.sh /tmp/install-agent-server.sh
RUN chmod +x /tmp/install-agent-server.sh && /tmp/install-agent-server.sh
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh

# Dual-mode entrypoint (shared, verbatim for every non-Python image):
COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]
```

The OpenHands agent-server block + CLI-agents block + entrypoint block are
**identical across every non-Python Dockerfile** — copy them verbatim from
`docker/chi/Dockerfile` or `docker/jwt/Dockerfile`. They install a standalone
Python 3.12 venv (the agent-server requires >=3.12; the golang/gcc/node images
ship 3.11 or none), the Codex + Claude Code CLIs (via portable Node if needed),
and wire up the dual-mode entrypoint that lets the same image serve every
harness (mini-swe-agent, OpenHands, Codex, Claude Code) plus the validation
gates.

### Node — `node:<ver>-bookworm-slim` + `npm install`

```dockerfile
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY repos/<name> /workspace/
WORKDIR /workspace

# Use `npm ci` if the repo ships package-lock.json (reproducible), else
# `npm install`. Install ALL deps (including devDependencies) so the test
# runner + its helpers are available offline.
RUN npm install   # or: RUN npm ci

# Git refuses to operate on the sandbox's host-owned copy ("dubious
# ownership"); mark any workspace safe so git-using checks pass.
RUN git config --system --add safe.directory '*'

# ── OpenHands agent-server block + entrypoint (verbatim) ─────
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
COPY docker/install-agent-server.sh /tmp/install-agent-server.sh
RUN chmod +x /tmp/install-agent-server.sh && /tmp/install-agent-server.sh
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh
COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]
```

For Node repos, also set `preserve_paths: [node_modules]` in the YAML so the
image's installed `node_modules` shows through the bind mount.

### Java — Maven or Gradle

Maven (`maven:<ver>-eclipse-temurin-<jdk>`) and Gradle
(`eclipse-temurin-<jdk>-jammy`) both work. **Check the JDK version the
repo's build tool requires**: e.g. java-jwt ships a Gradle 6.9.2 wrapper
that only supports JDK <= 15, so its image uses `eclipse-temurin:11-jdk-jammy`
even though the build targets Java 8. Using JDK 17 there fails at settings
file compile time with `Unsupported class file major version 61`.

```dockerfile
# Gradle example (Maven example: see docker/java-html-sanitizer/Dockerfile)
FROM eclipse-temurin:11-jdk-jammy

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV GRADLE_OPTS="-Dorg.gradle.daemon=false -Dorg.gradle.console=plain"

COPY repos/<name> /workspace/
WORKDIR /workspace

# Resolve deps + compile + test once to warm caches. `|| true` so a flaky
# network dep during build doesn't fail the image build — the gate run will
# surface genuine failures.
RUN ./gradlew :<module>:test --console=plain >/dev/null 2>&1 || true

# ── OpenHands agent-server block + entrypoint (verbatim) ─────
COPY docker/openhands-constraints.txt /tmp/openhands-constraints.txt
COPY docker/install-agent-server.sh /tmp/install-agent-server.sh
RUN chmod +x /tmp/install-agent-server.sh && /tmp/install-agent-server.sh
COPY docker/install-cli-agents.sh /tmp/install-cli-agents.sh
RUN chmod +x /tmp/install-cli-agents.sh && /tmp/install-cli-agents.sh
COPY docker/swe-duel-entrypoint.sh /usr/local/bin/swe-duel-entrypoint.sh
RUN chmod +x /usr/local/bin/swe-duel-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/swe-duel-entrypoint.sh"]
```

### C — `gcc:<ver>-bookworm` (+ `cmake make` if the build uses CMake)

For a simple Makefile repo (cJSON-style), `make test` at build time warms
the toolchain. For a CMake repo (libexpat-style), run the cmake configure +
build + test once at build time and copy any generated config headers next
to the sources so a plain `gcc -I<dir>` compile works offline at gate time:

```dockerfile
FROM gcc:14-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends cmake make \
    && rm -rf /var/lib/apt/lists/*

COPY repos/<name> /workspace/
WORKDIR /workspace

# Configure + build + run the bundled test suite once; copy generated
# config headers next to the sources (the injected-test compile path
# expects them there).
RUN cmake -S <source-root> -B /tmp/exb -DEXPAT_BUILD_TESTS=ON ... \
    && cmake --build /tmp/exb \
    && cp /tmp/exb/expat_config.h /workspace/<source-root>/lib/expat_config.h \
    && (cd /tmp/exb && ctest --output-on-failure) \
    && rm -rf /tmp/exb

# ── OpenHands agent-server block + entrypoint (verbatim) ─────
...
```

### Verifying the Dockerfile

```bash
docker build -t swe-duel-<name>:latest -f docker/<name>/Dockerfile .
```

Must succeed. Then sanity-check the suite runs offline (the gate container
has no network):

```bash
./venv/bin/python -c "
from swe_duel.sandbox.docker_executor import DockerExecutor
ex = DockerExecutor(docker_image='swe-duel-<name>:latest', timeout_s=600, memory_mb=2048)
r = ex.execute(file_overrides={}, command='<the test_command from the yaml>')
print('rc=', r.return_code, 'timed_out=', r.timed_out)
print(r.stdout[-800:])
"
```

If this hangs or times out, the suite is making network calls or starting
servers — pick a different repo or a different `test_command` subset that
runs fully offline.

---

## Step 3 — Build the image (no script edit needed)

`swe-duel setup docker` auto-discovers `src/swe_duel/docker/<name>/Dockerfile` from
`config/repos/<name>.yaml` — there is no build script to edit. Staging
(`swe_duel.sandbox.image_build.stage_docker_context`) copies the packaged
`docker/` tree, the bundled mock repo, and the fresh `repos/<name>/` clone
into a temp context root, so the Dockerfile can `COPY repos/<name>` /
`docker/...` exactly like every existing repo:

```bash
./venv/bin/swe-duel setup repos --only <name>    # clone first
./venv/bin/swe-duel setup docker --only <name>   # build swe-duel-<name>:latest
```

---

## Step 4 — Drop a per-repo adapter profile (Python can skip)

Language adapters live under `src/swe_duel/sandbox/languages/` (one module per
language). Per-repo variation — import path, test runner, build tool,
compile flags — is **not** hardcoded in those modules. Instead, every
non-Python repo gets a YAML profile at:

```
src/swe_duel/sandbox/languages/profiles/<name>.yaml
```

`get_adapter(language, repo_name)` loads `profiles/<repo_name>.yaml` (if
present) and constructs the language adapter with that profile. **The
filename stem must equal the repo's `name`.** For Python there is nothing
per-repo — skip to Step 5. You should **never** need to edit
`languages/*.py` just to add a repo of an already-supported language.

Copy the closest existing profile as a template, then fill in the fields
for your language:

### Go — `profiles/<name>.yaml`

```yaml
language: go
# From the repo's go.mod (`module github.com/...`). Surfaced in Red prompts
# so the agent does not have to guess the import path.
import_path: github.com/<org>/<module-path>
```

See `profiles/jwt.yaml` / `profiles/chi.yaml`.

### Node — `profiles/<name>.yaml`

Two test-runner modes:
- `node_test` (TypeScript via `npx tsx --test`, used by helmet): set
  `test_runner: node_test`, `ext: test.ts`.
- `mocha` (plain JavaScript via `npx mocha`, used by expressjs): set
  `test_runner: mocha`, `ext: test.js`, and `require:` to the repo's mocha
  bootstrap file if it has one (passed to `mocha --require <value>`).

```yaml
language: node
test_runner: mocha          # or node_test
ext: test.js                # or test.ts
require: test/support/env   # optional, mocha only
source_import_example: "const express = require('../lib/express');"  # optional prompt hint
```

> If the repo uses a different test runner (jest, vitest, …), the Node
> adapter does not support it — that counts as "adding a new language" and
> is out of scope.

See `profiles/helmet.yaml` / `profiles/expressjs.yaml`.

### Java — `profiles/<name>.yaml`

Two build tools: `maven` and `gradle`. Provide the module path, the test
source directory (where injected JUnit classes land), and the top-level
Java package prefix (for the prompt).

```yaml
language: java
build_tool: maven           # or gradle
module: owasp-java-html-sanitizer   # maven -pl path, or gradle :module path
test_source_dir: owasp-java-html-sanitizer/src/test/java
import_root: org.owasp.html
```

When no profile is found, the Java adapter falls back to Maven/owasp defaults
so bare `get_adapter("java")` (unit tests) keeps working. The Gradle branch
adds `--rerun-tasks` to the existing-test command so a baked "UP-TO-DATE"
cache does not mask regressions.

See `profiles/java-html-sanitizer.yaml` / `profiles/java-jwt.yaml`.

### C — `profiles/<name>.yaml`

Two build methods:
- `glob` (cJSON-style, single dir of `*.c` at the repo root): the adapter
  compiles the injected test against `find <source_dir> -name '*.c'`.
- `cmake` (libexpat-style, multi-source library with platform-conditional
  files): the adapter rebuilds the project's static lib via cmake, then
  links the injected test against it.

```yaml
# glob (cJSON-style)
language: c
build_method: glob
source_dir: "."
include_flags: "-I."
exclude_glob: test*.c
maxdepth: 1
header: cJSON.h

# cmake (libexpat-style)
language: c
build_method: cmake
cmake_source_root: expat
cmake_target: expat
cmake_extra_args: "-DEXPAT_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DEXPAT_BUILD_TESTS=OFF"
cmake_build_dir: /tmp/exb
include_flags: "-Iexpat/lib"
link_flags: "-L/tmp/exb -lexpat -lm"
header: expat.h
# Optional free-form extras merged into adapter.prompt_hints():
prompt_hints:
  include_directive: '#include "expat.h"'
  source_dir: expat/lib

# cmake + C++ (simdjson-style). The C adapter honours these optional knobs
# so injected harness tests compile with g++ rather than gcc -std=c99:
#   compiler: g++
#   std: c++17
#   test_ext: cpp          # injected file ends in .cpp
```

Use `glob` if the library is one or a few `.c` files in a single directory
with no generated/config headers. Use `cmake` if the sources need a config
header (like `expat_config.h`) that only cmake generates, or if a flat
`gcc *.c` compile fails with undefined-reference errors from
platform-conditional source files (libexpat's randomness sources). For a
C++ library that is close enough to "C-ish" (header + static lib), keep
`language: c` and set `compiler`/`std`/`test_ext` so the injected-test
compile path uses g++.

On hosts with very many CPUs, **always cap cmake build parallelism**
(`-j2`) in `test_command`: the gate container applies a hard memory limit
and an unrestricted `-j$(nproc)` parallel C++ build will OOM-kill the
compiler. See `profiles/simdjson.yaml` / `config/repos/simdjson.yaml`.

See `profiles/cjson.yaml` / `profiles/libexpat.yaml` / `profiles/simdjson.yaml`.

### Verifying the profile

```bash
./venv/bin/python -c "
from swe_duel.sandbox.languages import get_adapter
a = get_adapter('<language>', '<name>')
print(a.name, a.prompt_hints())
extra, cmd = a.prepare_injected('feature_tests.<ext>', '<minimal test>', ['<a target file>'])
print('files:', list(extra)); print('command:', cmd)
"
```

The command should look runnable inside the image. If it references paths
that do not exist in the image, fix the profile before proceeding.

---

## Step 5 — Add a container test class

Open `tests/test_repo_containers.py` (it is marked `integration` at module
level, so these run only when you select integration tests or run `make test`
without `-m "not integration"`). Add a class modeled on the existing ones —
it builds the image (if missing), runs the repo's `test_command` inside it,
and asserts the suite passes offline.

```python
# ── <name>: <Language/toolchain> ────────────────────────────


class Test<Name>Container:
    def test_<name>_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "<name>.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "<name>", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "<name> test suite timed out"
        assert result.return_code == 0, f"<name> tests failed:\n{result.stdout[-3000:]}"

        # Assert the runner's success marker appears in the output.
        # pytest: "N passed"; mocha: "N passing"; go: "ok <pkg>";
        # ctest: "100% tests passed"; gradle: "BUILD SUCCESSFUL".
```

For repos with a compile step worth proving (Go, C), add a second
`test_<name>_compiles_offline` method that runs a pure compile command
(`go build ./...`, a standalone `gcc` of the library + a trivial test
program, etc.) — see `TestJwtContainer.test_jwt_compiles_offline` and
`TestLibexpatContainer.test_libexpat_compiles_offline` for templates.

Also update the module docstring's repo list at the top of the file.

---

## Step 6 — Update `tests/test_config.py`

`TestLoadAllRepoConfigs.test_load_all_repo_configs` asserts the total repo
count and that each repo key is present. Update both:

```python
        # flask, jinja, sqlalchemy, <name> (python); helmet, expressjs (node); ...
        assert len(repos) == 12            # <- was 11, +1
        ...
        assert "<name>" in repos           # <- add this line
```

And in `test_non_python_repo_configs`, add an assertions block for the new
repo if it is non-Python (language, docker_image, url, a substring of
test_command).

---

## Keeping the challenge diff clean — `exclude_paths`

The validation gates compute a unified diff of the agent's workspace against
the pristine reference. If an exploring agent runs an in-tree build
(`./configure && make`, `./gradlew`, etc.), it generates build artifacts in
the workspace that are **not source edits** and must not appear in the diff.
A polluted diff balloons to megabytes and fails `gate_diff_valid` with
`"pr_diff is not a parseable unified diff"`.

The global `_DEFAULT_EXCLUDE` in `src/swe_duel/sandbox/workspace.py` already
covers the common build-output directories for every supported language:

```
_swe-duel/  .git/  __pycache__/  .pytest_cache/  .mypy_cache/  .ruff_cache/
node_modules/  vendor/  dist/  build/  .next/  .turbo/  coverage/  .cache/
target/  .m2/  .gradle/  TestSweDuelFeature.java  TestSweDuelBug.java
conversations/  bash_events/
```

You usually do **not** need to touch this. Only add `exclude_paths` to the
repo YAML if the repo's in-tree build writes artifacts somewhere the global
list does not cover (e.g. libexpat's autotools build writes
`expat/autom4te.cache/`, `expat/Makefile`, `expat/conftools/*`,
`expat/*/*.o`, libtool wrapper scripts, etc. under `expat/`).

### When to add `exclude_paths`

1. Run the new repo through one Red generation, then inspect the failed
   `gate_failures.json` under `data/workspaces/<model>/<name>/red/*/working/_swe-duel/`.
   If `gate_diff_valid` failed with "not a parseable unified diff", the
   workspace contains build artifacts.
2. List what the agent's build created that is not in the reference:
   ```bash
   cd data/workspaces/<model>/<name>/red/<id>
   python3 -c "
   from pathlib import Path
   ref={str(p.relative_to(Path('reference'))) for p in Path('reference').rglob('*') if p.is_file()}
   work={str(p.relative_to(Path('working'))) for p in Path('working').rglob('*') if p.is_file()}
   for r in sorted(work-ref):
       if not r.startswith('_swe-duel/'): print(' +', r)
   " | head -40
   ```
3. Add glob patterns (matched against the full repo-relative path via
   `fnmatch`) for each generated artifact. Be precise — do not exclude
   real source files. Non-glob patterns keep the old prefix/substring
   semantics.
4. Re-verify the diff is valid and contains only real source edits:
   ```bash
   ./venv/bin/python -c "
   from pathlib import Path
   from swe_duel.config import load_repo_config
   from swe_duel.sandbox import diff_utils
   from swe_duel.sandbox.workspace import _DEFAULT_EXCLUDE
   rc = load_repo_config(Path('config/repos/<name>.yaml'))
   excludes = _DEFAULT_EXCLUDE + list(rc.exclude_paths)
   ref = Path('data/workspaces/.../<id>/reference')
   work = Path('data/workspaces/.../<id>/working')
   d = diff_utils.generate_tree_diff(ref, work, exclude=excludes)
   print('len=', len(d), 'valid=', diff_utils.validate_diff_format(d))
   from unidiff import PatchSet
   print('files:', [f.path for f in PatchSet(d)])
   "
   ```

The diff should be a few KB and list only the source files the agent
intentionally edited.

---

## Final verification checklist

Run these in order from the project root:

```bash
# 1. Clone the new repo at the pinned commit.
make setup-repos                          # = swe-duel setup repos

# 2. Build the new image (and all others).
make build-docker                         # = swe-duel setup docker

# 3. Unit tests stay green (config count, adapter profile wiring).
./venv/bin/pytest tests/test_config.py tests/test_languages.py -v

# 4. The new container test passes (builds image if missing, runs suite offline).
./venv/bin/pytest "tests/test_repo_containers.py::Test<Name>Container" -v

# 5. Lint + typecheck stay clean (must not introduce new errors).
./venv/bin/ruff check .
./venv/bin/mypy src

# 6. Full unit suite (no Docker) — no regressions.
./venv/bin/pytest tests/ -m "not integration" -v

# 7. (Optional, once you have generated challenges in data/) gate regression
#    for this repo — red-feature / red-bug / red-self-review / blue-scoring.
./venv/bin/pytest tests/test_validation_gates_regression.py -k <name> -v
```

If steps 1–6 pass, the new repo is wired in. Update `AGENTS.md` (the
`make build-docker` bullet lists every image) and `.claude/CLAUDE.md` (the
directory tree lists every `config/repos/*.yaml` and `docker/*/Dockerfile`)
so the docs stay accurate. After the first successful Red generation, the
challenge is automatically picked up by
`tests/test_validation_gates_regression.py` once you add the repo name to
`_EXTENSION_REPOS` / `_ALL_REPOS` in that file.

---

## Reference: existing repos at a glance

| Repo | Language | Build system | `test_command` (abridged) | Notable config |
|------|----------|--------------|---------------------------|----------------|
| flask | python | pip (swe-duel-base) | `python -m pytest tests/ -x -q` | — |
| jinja | python | pip (swe-duel-base) | `python -m pytest tests/ -x -q` | — |
| sqlalchemy | python | pip (swe-duel-base) | `python -m pytest test/sql/ test/engine/ test/orm/ …` | pure-Python install (`DISABLE_SQLALCHEMY_CEXT=1`); SQLite offline; skips memory/timing-intensive |
| jwt | go | golang image | `go test -buildvcs=false ./...` | `login_shell: false` |
| chi | go | golang image | `go test -buildvcs=false ./...` | `login_shell: false` |
| csrf | go | golang image | `go test -buildvcs=false ./...` | `login_shell: false`; httptest-only suite |
| helmet | node | node image + tsx | `npx tsx --test $(ls test/*.test.ts \| ...)` | `preserve_paths: [node_modules]` |
| expressjs | node (js) | node image + mocha | `npx mocha --require test/support/env --reporter spec --check-leaks test/` | `preserve_paths: [node_modules]`; adapter rewrites `--reporter` to `tap` for the gate |
| node-jsonwebtoken | node (js) | node image + mocha | `npx mocha --reporter spec test/` | `preserve_paths: [node_modules]` |
| java-html-sanitizer | java | maven | `mvn test -B` | — |
| java-jwt | java | gradle | `./gradlew :java-jwt:test --console=plain` | image uses JDK 11 (Gradle 6.9.2 wrapper) |
| jjwt | java | maven multi-module | `mvn -B test` | native suite is Groovy/JUnit; injected tests land in `impl/src/test/java` |
| cjson | c | gcc + makefile | `make test` | — |
| libexpat | c | gcc + cmake | `cmake -S expat -B /tmp/exb ... && ctest` | `exclude_paths:` for autotools in-tree build artifacts |
| simdjson | c (C++17) | gcc/g++ + cmake | `cmake … -j2 --target unicode_tests … && ctest -R …` | profile sets `compiler: g++` / `std: c++17` / `test_ext: cpp`; cap `-j2` |

Use the closest match as your template.
