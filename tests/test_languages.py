"""Tests for language adapters — especially per-language sanitisation."""

from __future__ import annotations

import pytest

from swe_duel.sandbox.languages import get_adapter

@pytest.mark.parametrize(
    "rel",
    [
        "pom.xml",
        "owasp-java-html-sanitizer/pom.xml",
    ],
)
def test_java_adapter_strips_pom_xml(rel: str) -> None:
    adapter = get_adapter("java")
    overrides = {
        rel: "<broken>",
        "owasp-java-html-sanitizer/src/main/java/org/owasp/html/X.java": "class X {}",
    }
    filtered = adapter.filter_file_overrides(overrides)
    assert rel not in filtered
    assert "owasp-java-html-sanitizer/src/main/java/org/owasp/html/X.java" in filtered


def test_java_adapter_keeps_non_pom_files() -> None:
    adapter = get_adapter("java")
    overrides = {
        "src/main/java/Foo.java": "class Foo {}",
        "src/test/java/FooTest.java": "class FooTest {}",
    }
    assert adapter.filter_file_overrides(overrides) == overrides


def test_java_injected_command_runs_clean_and_uses_source_class_name() -> None:
    """Injected-test command must `clean` (avoid stale-target incremental
    recompile) and place the file at the test's *declared* public class name."""
    adapter = get_adapter("java")
    test_code = (
        "import org.junit.Test;\n"
        "public class TestSweDuelFeature {\n"
        "  @Test public void testFoo() { assert true; }\n}\n"
    )
    extra_files, command = adapter.prepare_injected(
        "test_swe_duel_feature.py", test_code, ["X.java"]
    )
    # File named after the declared public class, NOT a mangled label.
    assert any(p.endswith("/TestSweDuelFeature.java") for p in extra_files)
    assert not any("Testsweduelfeature" in p for p in extra_files)
    # `clean` forces a from-source compile.
    assert " clean " in f" {command} "
    assert "-Dtest=TestSweDuelFeature" in command


def test_java_existing_command_runs_clean() -> None:
    adapter = get_adapter("java")
    cmd = adapter.build_existing_command("mvn test -B", [])
    assert "clean" in cmd
    assert "surefire.excludes" in cmd


def test_java_class_name_falls_back_to_filename() -> None:
    """With no public class in the source, derive a valid name from the file."""
    adapter = get_adapter("java")
    extra_files, command = adapter.prepare_injected(
        "TestSweDuelBug.java", "// nothing declared yet\n", None
    )
    assert any(p.endswith("/TestSweDuelBug.java") for p in extra_files)
    assert "-Dtest=TestSweDuelBug" in command


def test_python_adapter_passes_through_overrides() -> None:
    adapter = get_adapter("python")
    overrides = {
        "pyproject.toml": "[broken",
        "src/foo.py": "x = 1",
    }
    assert adapter.filter_file_overrides(overrides) == overrides


# ── per-repo adapter selection ──────────────────────────────


def test_get_adapter_unknown_language_raises() -> None:
    with pytest.raises(ValueError):
        get_adapter("rust")


def test_repo_profiles_load_from_yaml_directory() -> None:
    """Per-repo adapter knobs come from profiles/*.yaml, not hard-coded dicts."""
    from swe_duel.sandbox.languages.profile_loader import load_repo_profile, profiles_dir

    assert (profiles_dir() / "jwt.yaml").is_file()
    assert load_repo_profile("jwt")["import_path"] == "github.com/golang-jwt/jwt/v5"
    assert load_repo_profile("expressjs")["test_runner"] == "mocha"
    assert load_repo_profile("java-jwt")["build_tool"] == "gradle"
    assert load_repo_profile("libexpat")["build_method"] == "cmake"
    assert load_repo_profile("csrf")["import_path"] == "github.com/gorilla/csrf"
    assert load_repo_profile("node-jsonwebtoken")["test_runner"] == "mocha"
    assert load_repo_profile("jjwt")["build_tool"] == "maven"
    assert load_repo_profile("jjwt")["module"] == "impl"
    assert load_repo_profile("simdjson")["build_method"] == "cmake"
    assert load_repo_profile("simdjson")["compiler"] == "g++"
    assert load_repo_profile("does-not-exist") == {}


def test_java_adapter_defaults_to_maven_when_no_repo() -> None:
    """get_adapter('java') with no repo keeps the owasp Maven defaults."""
    adapter = get_adapter("java")
    assert adapter.name == "java"
    extra, command = adapter.prepare_injected(
        "test_swe_duel_feature.py",
        "public class TestSweDuelFeature { @org.junit.Test public void t() {} }",
        None,
    )
    assert any("owasp-java-html-sanitizer/src/test/java/TestSweDuelFeature.java" == p for p in extra)
    assert "mvn " in command
    assert "-pl owasp-java-html-sanitizer" in command
    assert not adapter.is_gradle


def test_java_adapter_gradle_for_java_jwt() -> None:
    adapter = get_adapter("java", "java-jwt")
    assert adapter.is_gradle
    extra, command = adapter.prepare_injected(
        "test_swe_duel_feature.py",
        "public class TestSweDuelFeature { @org.junit.Test public void t() {} }",
        None,
    )
    # The test lands in the Gradle module's test source set under lib/.
    assert any(p == "lib/src/test/java/TestSweDuelFeature.java" for p in extra)
    assert "./gradlew :java-jwt:test" in command
    assert "--tests TestSweDuelFeature" in command
    assert "--console=plain" in command


def test_java_adapter_gradle_strips_build_gradle() -> None:
    adapter = get_adapter("java", "java-jwt")
    overrides = {
        "build.gradle": "// broken",
        "settings.gradle": "// broken",
        "lib/src/main/java/com/auth0/jwt/X.java": "class X {}",
    }
    filtered = adapter.filter_file_overrides(overrides)
    assert "build.gradle" not in filtered
    assert "settings.gradle" not in filtered
    assert "lib/src/main/java/com/auth0/jwt/X.java" in filtered


def test_java_adapter_gradle_existing_command_reruns() -> None:
    adapter = get_adapter("java", "java-jwt")
    cmd = adapter.build_existing_command("./gradlew :java-jwt:test", [])
    assert "--console=plain" in cmd
    assert "--rerun-tasks" in cmd


def test_java_adapter_prompt_hints_per_repo() -> None:
    maven = get_adapter("java", "java-html-sanitizer")
    gradle = get_adapter("java", "java-jwt")
    jjwt = get_adapter("java", "jjwt")
    assert maven.prompt_hints()["build_tool"] == "maven"
    assert maven.prompt_hints()["import_root"] == "org.owasp.html"
    assert gradle.prompt_hints()["build_tool"] == "gradle"
    assert gradle.prompt_hints()["import_root"] == "com.auth0.jwt"
    assert gradle.prompt_hints()["module"] == ":java-jwt"
    assert jjwt.prompt_hints()["build_tool"] == "maven"
    assert jjwt.prompt_hints()["import_root"] == "io.jsonwebtoken"
    assert jjwt.prompt_hints()["module"] == "impl"
    extra, command = jjwt.prepare_injected(
        "test_swe_duel_feature.py",
        "public class TestSweDuelFeature { @org.junit.Test public void t() {} }",
        None,
    )
    assert "impl/src/test/java/TestSweDuelFeature.java" in extra
    assert "-pl impl" in command


def test_c_adapter_cjson_root_glob() -> None:
    adapter = get_adapter("c", "cjson")
    extra, command = adapter.prepare_injected("feature_tests.c", "int main(){return 0;}", None)
    assert "_swe-duel/feature_tests.c" in extra
    # cJSON: glob root-level *.c excluding test*.c, with -I.
    assert "-I." in command
    assert "find . -maxdepth 1" in command
    assert "! -name test*.c" in command
    assert adapter.prompt_hints()["header"] == "cJSON.h"


def test_c_adapter_libexpat_uses_cmake() -> None:
    adapter = get_adapter("c", "libexpat")
    extra, command = adapter.prepare_injected("feature_tests.c", "int main(){return 0;}", None)
    assert "_swe-duel/feature_tests.c" in extra
    # libexpat: rebuild the static lib via cmake, then link the test against it.
    assert "cmake -S expat -B /tmp/exb" in command
    assert "--target expat" in command
    assert "-Iexpat/lib" in command
    assert "-L/tmp/exb -lexpat -lm" in command
    assert adapter.prompt_hints()["header"] == "expat.h"
    assert adapter.prompt_hints()["include_directive"] == '#include "expat.h"'


def test_c_adapter_simdjson_uses_gpp_and_cpp() -> None:
    """simdjson is C++17: injected tests are .cpp compiled with g++."""
    adapter = get_adapter("c", "simdjson")
    extra, command = adapter.prepare_injected(
        "feature_tests.cpp", "int main(){return 0;}", None
    )
    assert "_swe-duel/feature_tests.cpp" in extra
    assert "cmake -S . -B /tmp/simdjson-b" in command
    assert "--target simdjson" in command
    assert "g++ -std=c++17" in command
    assert "-Iinclude" in command
    assert "-lsimdjson" in command
    assert adapter.prompt_hints()["header"] == "simdjson.h"


def test_node_adapter_helmet_uses_tsx_node_test() -> None:
    adapter = get_adapter("node", "helmet")
    extra, command = adapter.prepare_injected("feature_tests.test.ts", "test('x', () => {})", None)
    assert any(p.endswith(".test.ts") for p in extra)
    assert "npx tsx --test" in command
    assert not adapter.is_mocha
    assert adapter.lint_command(["src/x.ts"]) == "npx tsc --noEmit -p tsconfig.json"


def test_node_adapter_express_uses_mocha() -> None:
    adapter = get_adapter("node", "expressjs")
    assert adapter.is_mocha
    extra, command = adapter.prepare_injected("feature_tests.test.js", "it('x', () => {})", None)
    assert any(p.endswith(".test.js") for p in extra)
    assert "npx mocha" in command
    assert "--reporter tap" in command
    assert "--require test/support/env" in command
    # express is plain JS with no tsconfig → lint skipped.
    assert adapter.lint_command(["index.js"]) is None


def test_node_adapter_mocha_existing_command_normalizes_reporter() -> None:
    """The repo test_command may use mocha's spec reporter, but the adapter
    parses TAP — build_existing_command must rewrite --reporter to tap so the
    existing-test gate output is parseable."""
    adapter = get_adapter("node", "expressjs")
    cmd = adapter.build_existing_command(
        "npx mocha --require test/support/env --reporter spec --check-leaks test/", []
    )
    assert "--reporter tap" in cmd
    assert "--reporter spec" not in cmd
    # A command with no --reporter gets one appended.
    cmd2 = adapter.build_existing_command("npx mocha --require test/support/env test/", [])
    assert "--reporter tap" in cmd2


def test_node_adapter_helmet_existing_command_passthrough() -> None:
    """The tsx/node:test adapter has no reporter to normalize."""
    adapter = get_adapter("node", "helmet")
    base = "npx tsx --test $(ls test/*.test.ts | grep -v project-setups)"
    assert adapter.build_existing_command(base, []) == base


def test_go_adapter_prompt_hints_per_repo() -> None:
    assert get_adapter("go", "jwt").prompt_hints()["import_path"] == "github.com/golang-jwt/jwt/v5"
    assert get_adapter("go", "chi").prompt_hints()["import_path"] == "github.com/go-chi/chi/v5"
    # Unknown go repo → empty hint (agent reads go.mod).
    assert get_adapter("go", "unknown").prompt_hints()["import_path"] == ""


# ── mocha / gradle output parsing ──────────────────────────


def test_parse_mocha_tap_pass() -> None:
    from swe_duel.models import ExecutionResult

    adapter = get_adapter("node", "expressjs")
    # mocha's TAP writes `ok N <name>` (no dash after the number).
    stdout = "1..3\nok 1 Suite a\nok 2 Suite b\nok 3 Suite c\n"
    er = ExecutionResult(return_code=0, stdout=stdout, stderr="", timed_out=False, duration_ms=10)
    result = adapter.parse(stdout, er)
    assert result.passed
    assert result.total == 3
    assert result.passed_count == 3
    assert result.failed_count == 0


def test_parse_mocha_tap_fail() -> None:
    from swe_duel.models import ExecutionResult

    adapter = get_adapter("node", "expressjs")
    # Standard TAP `not ok N - name` form (mocha accepts both).
    stdout = (
        "1..2\nok 1 - Suite a\nnot ok 2 - Suite b\n  ---\n  message: boom\n  ---\n"
    )
    er = ExecutionResult(return_code=1, stdout=stdout, stderr="", timed_out=False, duration_ms=10)
    result = adapter.parse(stdout, er)
    assert not result.passed
    assert result.total == 2
    assert result.failed_count == 1
    assert "Suite b" in result.failure_messages[0]


def test_parse_gradle_test_pass() -> None:
    from swe_duel.models import ExecutionResult

    adapter = get_adapter("java", "java-jwt")
    stdout = (
        "> Task :java-jwt:test\n"
        "5 tests completed\n"
        "BUILD SUCCESSFUL\n"
    )
    er = ExecutionResult(return_code=0, stdout=stdout, stderr="", timed_out=False, duration_ms=10)
    result = adapter.parse(stdout, er)
    assert result.passed
    assert result.total == 5
    assert result.passed_count == 5
    assert result.failed_count == 0


def test_parse_gradle_test_fail() -> None:
    from swe_duel.models import ExecutionResult

    adapter = get_adapter("java", "java-jwt")
    stdout = (
        "> Task :java-jwt:test\n"
        "com.auth0.jwt.JWTDecoderTest > testDecode FAILED\n"
        "3 tests completed, 1 failed\n"
        "BUILD FAILED\n"
    )
    er = ExecutionResult(return_code=1, stdout=stdout, stderr="", timed_out=False, duration_ms=10)
    result = adapter.parse(stdout, er)
    assert not result.passed
    assert result.total == 3
    assert result.failed_count == 1
    assert any("JWTDecoderTest>testDecode" in m for m in result.failure_messages)
