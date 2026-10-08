"""Build + run target repos' native test suites in Docker.

helmet (Node/TypeScript), expressjs (Node/mocha), node-jsonwebtoken
(Node/mocha), jwt (Go), chi (Go), csrf (Go), java-html-sanitizer
(Java/Maven), java-jwt (Java/Gradle), jjwt (Java/Maven), cjson
(C/Makefile), libexpat (C/CMake), simdjson (C++/CMake), plus the Python
repos that need their own image warmup (jinja, sqlalchemy) ship native
test runners. These tests verify the per-repo Docker image:

  1. builds successfully (toolchain + dependencies baked in), and
  2. can compile and run that repo's own test suite **fully offline**
     (``--network none``) — exactly how DockerExecutor runs commands.

The suites are invoked with the same ``test_command`` declared in
``config/repos/<name>.yaml`` so the config and the image stay in lock-step.

These tests are slow (image build can take minutes) and need Docker + the
cloned repo, so they are marked ``integration`` and skip gracefully when
prerequisites are missing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from swe_duel.config import load_repo_config
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.image_build import build_repo_image, image_exists, staged_build_context

pytestmark = pytest.mark.integration


def _build_image(project_root: Path, repo_name: str, image_tag: str) -> subprocess.CompletedProcess:
    """Build swe-duel-<repo> from the staged canonical context (same path `swe-duel setup docker` uses)."""
    with staged_build_context({repo_name: project_root / "repos" / repo_name}) as ctx:
        return build_repo_image(repo_name, ctx, capture=True)


def _ensure_image(project_root: Path, repo_name: str, image_tag: str, record) -> None:
    """Build the image if it does not already exist; skip the test if the build fails."""
    repo_dir = project_root / "repos" / repo_name
    if not repo_dir.exists():
        pytest.skip(f"repos/{repo_name} not cloned — run: swe-duel setup repos --only {repo_name}")
    if image_exists(image_tag):
        record("image_prebuilt", True)
        return
    record("image_prebuilt", False)
    build = _build_image(project_root, repo_name, image_tag)
    record("build_return_code", build.returncode)
    record("build_stderr_tail", build.stderr[-2000:] if build.stderr else "")
    if build.returncode != 0:
        pytest.skip(f"docker build for {image_tag} failed:\n{build.stderr[-2000:]}")


def _er(result) -> dict:
    return {
        "return_code": result.return_code,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-2000:],
        "timed_out": result.timed_out,
        "duration_ms": result.duration_ms,
    }


# ── helmet: Node/TypeScript ─────────────────────────────────


class TestHelmetContainer:
    def test_helmet_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "helmet.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "helmet", image_tag, record)

        # Generous timeout: tsx transpiles every test file on the fly.
        executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "helmet test suite timed out"
        assert result.return_code == 0, f"helmet tests failed:\n{result.stdout[-3000:]}"

        # Node's test runner prints a TAP summary like "# pass 124" / "# fail 0".
        pass_match = re.search(r"#\s*pass\s+(\d+)", result.stdout)
        fail_match = re.search(r"#\s*fail\s+(\d+)", result.stdout)
        record("pass_count", int(pass_match.group(1)) if pass_match else None)
        record("fail_count", int(fail_match.group(1)) if fail_match else None)
        assert pass_match, f"could not find pass count in output:\n{result.stdout[-2000:]}"
        assert int(pass_match.group(1)) > 0
        assert fail_match and int(fail_match.group(1)) == 0


# ── jwt: Go ─────────────────────────────────────────────────


class TestJwtContainer:
    def test_jwt_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "jwt.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "jwt", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "jwt test suite timed out"
        assert result.return_code == 0, f"jwt tests failed:\n{result.stdout[-3000:]}"

        # `go test ./...` prints "ok <pkg>" per package and "FAIL" on any failure.
        record("ok_lines", [ln for ln in result.stdout.splitlines() if ln.startswith("ok")])
        assert "FAIL" not in result.stdout, f"go test reported a failure:\n{result.stdout[-2000:]}"
        assert "ok  \tgithub.com/golang-jwt/jwt/v5" in result.stdout

    def test_jwt_compiles_offline(self, project_root: Path, config_dir: Path, record):
        """A pure compile step proves the Go toolchain + sources are wired up."""
        rc = load_repo_config(config_dir / "repos" / "jwt.yaml")
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "jwt", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=180, memory_mb=1024)
        result = executor.execute(file_overrides={}, command="go build -buildvcs=false ./...")
        record("command", "go build -buildvcs=false ./...")
        record("result", _er(result))
        assert not result.timed_out
        assert result.return_code == 0, f"go build failed:\n{result.stderr[-2000:]}"


# ── java-html-sanitizer: Java/Maven ───────────────────────


class TestJavaHtmlSanitizerContainer:
    def test_java_html_sanitizer_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "java-html-sanitizer.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "java-html-sanitizer", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "java-html-sanitizer test suite timed out"
        assert result.return_code == 0, f"java-html-sanitizer tests failed:\n{result.stdout[-4000:]}"

        assert "BUILD SUCCESS" in result.stdout, f"expected BUILD SUCCESS:\n{result.stdout[-3000:]}"
        assert "BUILD FAILURE" not in result.stdout
        # Surefire prints "Tests run: N, Failures: 0, Errors: 0" in the Maven output.
        tests_match = re.search(
            r"Tests run:\s+(\d+),\s*Failures:\s+(\d+),\s*Errors:\s+(\d+)",
            result.stdout,
        )
        record("tests_run", int(tests_match.group(1)) if tests_match else None)
        record("failures", int(tests_match.group(2)) if tests_match else None)
        record("errors", int(tests_match.group(3)) if tests_match else None)
        assert tests_match, f"could not find surefire summary:\n{result.stdout[-3000:]}"
        assert int(tests_match.group(1)) > 0
        assert int(tests_match.group(2)) == 0
        assert int(tests_match.group(3)) == 0


# ── cjson: C/Makefile ─────────────────────────────────────


class TestCjsonContainer:
    def test_cjson_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "cjson.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "cjson", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=120, memory_mb=512)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "cjson test suite timed out"
        assert result.return_code == 0, f"cjson tests failed:\n{result.stdout[-3000:]}"

        # cJSON_test prints the library version and a set of JSON samples on
        # success; the binary exits non-zero on any failure, so a zero return
        # code plus the version marker is sufficient confirmation.
        assert "Version:" in result.stdout, f"no version marker in output:\n{result.stdout[-2000:]}"
        assert "FAIL" not in result.stdout.upper()


# ── jinja: Python/pytest ──────────────────────────────────


class TestJinjaContainer:
    def test_jinja_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "jinja.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "jinja", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "jinja test suite timed out"
        assert result.return_code == 0, f"jinja tests failed:\n{result.stdout[-3000:]}"

        match = re.search(r"(\d+) passed", result.stdout)
        record("passed_count", int(match.group(1)) if match else None)
        assert match, f"could not find pass count in output:\n{result.stdout[-2000:]}"
        assert int(match.group(1)) > 0


# ── expressjs: Node/mocha ─────────────────────────────────


class TestExpressjsContainer:
    def test_expressjs_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "expressjs.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "expressjs", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "expressjs test suite timed out"
        assert result.return_code == 0, f"expressjs tests failed:\n{result.stdout[-3000:]}"

        # mocha spec reporter prints "N passing (Ys)" / "N failing".
        passing_match = re.search(r"(\d+)\s+passing", result.stdout)
        failing_match = re.search(r"(\d+)\s+failing", result.stdout)
        record("passing", int(passing_match.group(1)) if passing_match else None)
        record("failing", int(failing_match.group(1)) if failing_match else None)
        assert passing_match, f"could not find passing count:\n{result.stdout[-2000:]}"
        assert int(passing_match.group(1)) > 0
        assert (not failing_match) or int(failing_match.group(1)) == 0


# ── chi: Go ───────────────────────────────────────────────


class TestChiContainer:
    def test_chi_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "chi.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "chi", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "chi test suite timed out"
        assert result.return_code == 0, f"chi tests failed:\n{result.stdout[-3000:]}"
        assert "FAIL" not in result.stdout, f"go test reported a failure:\n{result.stdout[-2000:]}"
        record("ok_lines", [ln for ln in result.stdout.splitlines() if ln.startswith("ok")])
        assert any("go-chi/chi" in ln for ln in result.stdout.splitlines() if ln.startswith("ok"))

    def test_chi_compiles_offline(self, project_root: Path, config_dir: Path, record):
        """A pure compile step proves the Go toolchain + sources are wired up."""
        rc = load_repo_config(config_dir / "repos" / "chi.yaml")
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "chi", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=180, memory_mb=1024)
        result = executor.execute(file_overrides={}, command="go build -buildvcs=false ./...")
        record("command", "go build -buildvcs=false ./...")
        record("result", _er(result))
        assert not result.timed_out
        assert result.return_code == 0, f"go build failed:\n{result.stderr[-2000:]}"


# ── libexpat: C/CMake ─────────────────────────────────────


class TestLibexpatContainer:
    def test_libexpat_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "libexpat.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "libexpat", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "libexpat test suite timed out"
        assert result.return_code == 0, f"libexpat tests failed:\n{result.stdout[-4000:]}"
        # ctest prints "100% tests passed, 0 tests failed out of N".
        pct_match = re.search(r"(\d+)%\s+tests passed", result.stdout)
        total_match = re.search(r"out of\s+(\d+)", result.stdout)
        record("pass_pct", int(pct_match.group(1)) if pct_match else None)
        record("total", int(total_match.group(1)) if total_match else None)
        assert pct_match, f"could not find ctest summary:\n{result.stdout[-3000:]}"
        assert int(pct_match.group(1)) == 100

    def test_libexpat_compiles_offline(self, project_root: Path, config_dir: Path, record):
        """A cmake build of the static lib + a trivial injected C test (linked
        against it) proves the C adapter's per-repo cmake compile path works
        for libexpat's multi-source, platform-conditional layout."""
        rc = load_repo_config(config_dir / "repos" / "libexpat.yaml")
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "libexpat", image_tag, record)

        from swe_duel.sandbox.languages import get_adapter

        adapter = get_adapter("c", "libexpat")
        test_code = (
            '#include "expat.h"\n'
            "#include <assert.h>\n"
            "int main(void) {\n"
            "  XML_Parser p = XML_ParserCreate(NULL);\n"
            "  assert(p != NULL);\n"
            "  XML_ParserFree(p);\n"
            "  return 0;\n"
            "}\n"
        )
        extra, command = adapter.prepare_injected(
            "feature_tests.c", test_code, ["expat/lib/xmlparse.c"]
        )
        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, extra_files=extra, command=command)
        record("command", command)
        record("result", _er(result))
        assert not result.timed_out
        assert result.return_code == 0, (
            f"libexpat injected-test compile/run failed:\n{result.stderr[-2000:]}\n"
            f"stdout: {result.stdout[-1000:]}"
        )


# ── java-jwt: Java/Gradle ─────────────────────────────────


class TestJavaJwtContainer:
    def test_java_jwt_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "java-jwt.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "java-jwt", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "java-jwt test suite timed out"
        assert result.return_code == 0, f"java-jwt tests failed:\n{result.stdout[-4000:]}"
        assert "BUILD SUCCESSFUL" in result.stdout, f"expected BUILD SUCCESSFUL:\n{result.stdout[-3000:]}"
        assert "BUILD FAILED" not in result.stdout

    def test_java_jwt_compiles_offline(self, project_root: Path, config_dir: Path, record):
        """A compile-only Gradle task proves the toolchain + sources are wired."""
        rc = load_repo_config(config_dir / "repos" / "java-jwt.yaml")
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "java-jwt", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=2048)
        result = executor.execute(
            file_overrides={},
            command="./gradlew :java-jwt:compileJava --console=plain",
        )
        record("command", "./gradlew :java-jwt:compileJava --console=plain")
        record("result", _er(result))
        assert not result.timed_out
        assert result.return_code == 0, f"gradle compile failed:\n{result.stderr[-2000:]}"
        assert "BUILD SUCCESSFUL" in result.stdout


# ── sqlalchemy: Python/pytest ─────────────────────────────


class TestSqlalchemyContainer:
    def test_sqlalchemy_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "sqlalchemy.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "sqlalchemy", image_tag, record)

        # Core sql/engine/orm subset is large but runs fully offline on SQLite.
        executor = DockerExecutor(docker_image=image_tag, timeout_s=900, memory_mb=2048)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "sqlalchemy test suite timed out"
        assert result.return_code == 0, f"sqlalchemy tests failed:\n{result.stdout[-3000:]}"
        match = re.search(r"(\d+) passed", result.stdout)
        record("passed_count", int(match.group(1)) if match else None)
        assert match, f"could not find pass count in output:\n{result.stdout[-2000:]}"
        assert int(match.group(1)) > 0


# ── csrf: Go (gorilla) ────────────────────────────────────


class TestCsrfContainer:
    def test_csrf_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "csrf.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "csrf", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "csrf test suite timed out"
        assert result.return_code == 0, f"csrf tests failed:\n{result.stdout[-3000:]}"
        assert "FAIL" not in result.stdout, f"go test reported a failure:\n{result.stdout[-2000:]}"
        record("ok_lines", [ln for ln in result.stdout.splitlines() if ln.startswith("ok")])
        assert any("gorilla/csrf" in ln for ln in result.stdout.splitlines() if ln.startswith("ok"))

    def test_csrf_compiles_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "csrf.yaml")
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "csrf", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=180, memory_mb=1024)
        result = executor.execute(file_overrides={}, command="go build -buildvcs=false ./...")
        record("command", "go build -buildvcs=false ./...")
        record("result", _er(result))
        assert not result.timed_out
        assert result.return_code == 0, f"go build failed:\n{result.stderr[-2000:]}"


# ── node-jsonwebtoken: Node/mocha ─────────────────────────


class TestNodeJsonwebtokenContainer:
    def test_node_jsonwebtoken_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "node-jsonwebtoken.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "node-jsonwebtoken", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=300, memory_mb=1024)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "node-jsonwebtoken test suite timed out"
        assert result.return_code == 0, f"node-jsonwebtoken tests failed:\n{result.stdout[-3000:]}"
        passing_match = re.search(r"(\d+)\s+passing", result.stdout)
        failing_match = re.search(r"(\d+)\s+failing", result.stdout)
        record("passing", int(passing_match.group(1)) if passing_match else None)
        record("failing", int(failing_match.group(1)) if failing_match else None)
        assert passing_match, f"could not find passing count:\n{result.stdout[-2000:]}"
        assert int(passing_match.group(1)) > 0
        assert (not failing_match) or int(failing_match.group(1)) == 0


# ── jjwt: Java/Maven ──────────────────────────────────────


class TestJjwtContainer:
    def test_jjwt_native_tests_pass_offline(self, project_root: Path, config_dir: Path, record):
        rc = load_repo_config(config_dir / "repos" / "jjwt.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "jjwt", image_tag, record)

        executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "jjwt test suite timed out"
        assert result.return_code == 0, f"jjwt tests failed:\n{result.stdout[-4000:]}"
        assert "BUILD SUCCESS" in result.stdout, f"expected BUILD SUCCESS:\n{result.stdout[-3000:]}"
        assert "BUILD FAILURE" not in result.stdout
        tests_match = re.search(
            r"Tests run:\s+(\d+),\s*Failures:\s+(\d+),\s*Errors:\s+(\d+)",
            result.stdout,
        )
        record("tests_run", int(tests_match.group(1)) if tests_match else None)
        record("failures", int(tests_match.group(2)) if tests_match else None)
        record("errors", int(tests_match.group(3)) if tests_match else None)
        assert tests_match, f"could not find surefire summary:\n{result.stdout[-3000:]}"
        assert int(tests_match.group(1)) > 0
        assert int(tests_match.group(2)) == 0
        assert int(tests_match.group(3)) == 0


# ── simdjson: C++/CMake ───────────────────────────────────


class TestSimdjsonContainer:
    def test_simdjson_native_tests_pass_offline(
        self, project_root: Path, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / "simdjson.yaml")
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "simdjson", image_tag, record)

        # Acceptance subset is moderate; combine long timeout with 2 GB+
        # so cmake + -j2 does not OOM.
        executor = DockerExecutor(docker_image=image_tag, timeout_s=900, memory_mb=4096)
        result = executor.execute(file_overrides={}, command=rc.test_command)
        record("command", rc.test_command)
        record("result", _er(result))

        assert not result.timed_out, "simdjson test suite timed out"
        assert result.return_code == 0, f"simdjson tests failed:\n{result.stdout[-4000:]}"
        pct_match = re.search(r"(\d+)%\s+tests passed", result.stdout)
        total_match = re.search(r"out of\s+(\d+)", result.stdout)
        record("pass_pct", int(pct_match.group(1)) if pct_match else None)
        record("total", int(total_match.group(1)) if total_match else None)
        assert pct_match, f"could not find ctest summary:\n{result.stdout[-3000:]}"
        assert int(pct_match.group(1)) == 100
        assert total_match and int(total_match.group(1)) >= 1

    def test_simdjson_compiles_offline(self, project_root: Path, config_dir: Path, record):
        """cmake static lib + trivial injected C++ test via the C adapter."""
        rc = load_repo_config(config_dir / "repos" / "simdjson.yaml")
        image_tag = f"{rc.docker_image}:latest"
        _ensure_image(project_root, "simdjson", image_tag, record)

        from swe_duel.sandbox.languages import get_adapter

        adapter = get_adapter("c", "simdjson")
        test_code = (
            '#include "simdjson.h"\n'
            "#include <cassert>\n"
            "#include <string>\n"
            "int main() {\n"
            '  std::string json("{\\"a\\":1}");\n'
            "  simdjson::padded_string j(json);\n"
            "  simdjson::ondemand::parser parser;\n"
            "  auto doc = parser.iterate(j);\n"
            "  int64_t a = 0;\n"
            "  auto err = doc[\"a\"].get(a);\n"
            "  assert(err == simdjson::SUCCESS);\n"
            "  assert(a == 1);\n"
            "  return 0;\n"
            "}\n"
        )
        extra, command = adapter.prepare_injected(
            "feature_tests.cpp", test_code, ["src/simdjson.cpp"]
        )
        executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
        result = executor.execute(file_overrides={}, extra_files=extra, command=command)
        record("command", command)
        record("result", _er(result))
        assert not result.timed_out
        assert result.return_code == 0, (
            f"simdjson injected-test compile/run failed:\n{result.stderr[-2000:]}\n"
            f"stdout: {result.stdout[-1000:]}"
        )
