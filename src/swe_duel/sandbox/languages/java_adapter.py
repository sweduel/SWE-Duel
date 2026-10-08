"""Java (Maven / Gradle) language adapter.

Per-repo build profiles live in ``profiles/<repo>.yaml``. When no profile is
present, defaults match the original Maven/owasp layout so ``get_adapter("java")``
keeps unit-test back-compat.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from swe_duel.models import ExecutionResult, TestExecutionResult
from swe_duel.sandbox.languages.base import LanguageAdapter, infer_collection_error
from swe_duel.sandbox.languages.profile_loader import load_repo_profile


class JavaAdapter(LanguageAdapter):
    """Adapter for Java projects built with either Maven or Gradle.

    Injected tests are standalone JUnit 4 classes placed in the module's
    ``src/test/java`` directory (default package). The build tool is selected
    per repo via its YAML profile:

      - ``java-html-sanitizer`` (Maven, module ``owasp-java-html-sanitizer``):
        ``mvn -B -pl <module> -am clean -Dtest=<Class> test``.
      - ``java-jwt`` (Gradle, module ``:java-jwt`` whose project dir is
        ``lib``): ``./gradlew :java-jwt:test --console=plain --tests <Class>``.

    SWE-Duel injected test classes (``TestSweDuel*``) are excluded from the repo's
    normal test run so they do not leak into ``gate_existing_tests``: Maven via
    ``-Dsurefire.excludes``, Gradle by virtue of the sandbox seeding each
    ``execute()`` from a fresh image copy that contains no SWE-Duel files (and
    ``filter_file_overrides`` drops any ``TestSweDuel*`` the agent tries to inject
    via file overrides).
    """

    name = "java"

    # Red agents often break Maven/Gradle by editing pom.xml/build.gradle while
    # trying to fix dependency issues. Red agents also occasionally write SWE-Duel
    # test classes directly into src/test/java instead of _swe-duel/. Drop build
    # files and any SWE-Duel test file so the existing-test gate runs only the
    # upstream suite.
    _BUILD_FILE_RE = re.compile(r"(^|/)(pom\.xml|build\.gradle|settings\.gradle)$")
    _SWE_DUEL_TEST_RE = re.compile(r"(^|/)TestSweDuel[A-Za-z0-9_]*\.java$")

    # Public top-level class/interface/enum declared in a Java source. Java
    # requires the file to be named exactly after it, so this is the source of
    # truth for the on-disk filename (NOT the logical test_filename label).
    _PUBLIC_CLASS_RE = re.compile(
        r"\bpublic\s+(?:final\s+|abstract\s+)*(?:class|interface|enum)\s+([A-Za-z_]\w*)"
    )

    def __init__(self, repo_name: str | None = None) -> None:
        self.repo_name = (repo_name or "").strip()
        profile = load_repo_profile(self.repo_name)
        # Default to the Maven/owasp profile for back-compat when no repo is
        # given (e.g. unit tests calling get_adapter("java")).
        self._build_tool = str(profile.get("build_tool", "maven"))
        self._module = str(profile.get("module", "owasp-java-html-sanitizer"))
        self._test_source_dir = str(
            profile.get("test_source_dir", "owasp-java-html-sanitizer/src/test/java")
        )
        self._import_root = str(profile.get("import_root", ""))
        self._profile = profile

    @property
    def is_gradle(self) -> bool:
        return self._build_tool == "gradle"

    def _class_name(self, test_filename: str, test_code: str = "") -> str:
        # Prefer the public class the test code actually declares — Java will
        # not compile if the filename disagrees with it, and casing matters
        # (`TestSweDuelFeature`, not a mangled `Testsweduelfeature`).
        m = self._PUBLIC_CLASS_RE.search(test_code or "")
        if m:
            return m.group(1)
        # Fallback: derive a valid identifier from the filename. Preserve the
        # stem verbatim if it is already a valid Java identifier; only split on
        # underscores/illegal chars and TitleCase when necessary.
        stem = PurePosixPath(test_filename).stem
        if re.fullmatch(r"[A-Za-z_]\w*", stem):
            return stem
        stem = re.sub(r"[^0-9A-Za-z_]", "_", stem)
        parts = [p for p in stem.split("_") if p]
        return "".join(p[:1].upper() + p[1:] for p in parts) or "SweDuelTest"

    def _java_test_path(self, test_filename: str, test_code: str = "") -> str:
        class_name = self._class_name(test_filename, test_code)
        return f"{self._test_source_dir}/{class_name}.java"

    def prepare_injected(
        self,
        test_filename: str,
        test_code: str,
        target_files: list[str] | None,
    ) -> tuple[dict[str, str], str]:
        path = self._java_test_path(test_filename, test_code)
        class_name = PurePosixPath(path).stem
        if self.is_gradle:
            # Gradle: filter the module's test task to just this class. The
            # file is placed in the module's test source set so the Java/Groovy
            # compiler can resolve imports against the module's classpath.
            command = f"./gradlew {self._module}:test --console=plain --tests {class_name}"
        else:
            # `clean` is essential: the repo image bakes a warmed `target/`
            # (compiled .class files) to speed `mvn test`. The gate seeds a
            # fresh container from that image, so the stale `target/` is
            # present; when a source OVERRIDE changes a file's mtime, Maven
            # does an INCREMENTAL recompile that mixes stale and fresh classes
            # and spuriously fails to compile untouched files. A clean build
            # recompiles everything from source, matching the agent's own
            # bind-mounted clone (which has no target/).
            command = (
                f"mvn -B -pl {self._module} -am clean "
                f"-Dtest={class_name} -DfailIfNoTests=false "
                f"-Dsurefire.failIfNoSpecifiedTests=false test"
            )
        return {path: test_code}, command

    def filter_file_overrides(self, file_overrides: dict[str, str]) -> dict[str, str]:
        return {
            rel: content
            for rel, content in file_overrides.items()
            if not self._BUILD_FILE_RE.search(rel) and not self._SWE_DUEL_TEST_RE.search(rel)
        }

    def build_existing_command(self, base_command: str, deselects: list[str]) -> str:
        if self.is_gradle:
            # Gradle has no per-test deselect. Ensure --console=plain so output
            # is parseable, and force a rerun so a stale "UP-TO-DATE" cache
            # (from the image's build-time warm-up) does not mask regressions
            # introduced by the agent's source overrides.
            cmd = base_command
            if "--console=plain" not in cmd:
                cmd = f"{cmd} --console=plain"
            if "--rerun-tasks" not in cmd:
                cmd = f"{cmd} --rerun-tasks"
            return cmd
        # Maven Surefire's default includes pick up anything named Test*.java,
        # which would run the injected SWE-Duel bug tests during gate_existing_tests.
        # Tell Surefire to skip them; injected tests are still run explicitly
        # via -Dtest=... by prepare_injected.
        #
        # `clean` is injected for the same reason as prepare_injected: the
        # image's baked `target/` makes Maven do an incremental recompile
        # against stale classes once an override changes an mtime, which
        # spuriously breaks compilation of untouched sources. Force a
        # from-source build.
        if "surefire.excludes" in base_command:
            return base_command
        cmd = base_command
        if " clean " not in f" {cmd} " and not cmd.strip().startswith("mvn clean"):
            # Insert `clean` right after the `mvn` (and any -B/-pl/-am flags
            # are fine after goals); simplest correct form is
            # `mvn clean <rest>`.
            cmd = re.sub(r"^(\s*mvn)\b", r"\1 clean", cmd, count=1)
        return f"{cmd} -Dsurefire.excludes='**/TestSweDuel*.java'"

    def parse(self, stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
        if self.is_gradle:
            return _parse_gradle_test(stdout, exec_result)
        return _parse_maven_surefire(stdout, exec_result)

    def prompt_hints(self) -> dict[str, str]:
        return {
            "build_tool": self._build_tool,
            "module": self._module,
            "test_source_dir": self._test_source_dir,
            "import_root": self._import_root,
        }

    def count_assertions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        patterns = (
            r"\bassert\w*\s*\(",
            r"\bAssert\.\w+\s*\(",
            r"\bfail\s*\(",
        )
        return sum(len(re.findall(p, test_code)) for p in patterns)

    def count_test_functions(self, test_code: str) -> int:
        if not test_code.strip():
            return 0
        return max(
            len(re.findall(r"@Test\b", test_code)),
            len(re.findall(r"\bpublic\s+void\s+test\w+\s*\(", test_code)),
        )


def _parse_maven_surefire(
    stdout: str, exec_result: ExecutionResult
) -> TestExecutionResult:
    """Parse Maven Surefire's "Tests run: X, Failures: Y, Errors: Z" summary."""
    summary_re = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*" r"Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
    )
    matches = list(summary_re.finditer(stdout))
    failure_messages: list[str] = []

    if not matches:
        if exec_result.return_code in (0, None):
            return TestExecutionResult(
                passed=True,
                total=0,
                passed_count=0,
                failed_count=0,
                error_count=0,
                failure_messages=[],
                stdout=stdout,
                stderr=exec_result.stderr,
                duration_ms=exec_result.duration_ms,
            )
        failure_messages.append(infer_collection_error(stdout, exec_result.stderr))
        return TestExecutionResult(
            passed=False,
            total=0,
            passed_count=0,
            failed_count=0,
            error_count=1,
            failure_messages=failure_messages,
            stdout=stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
        )

    # Use the last summary line; for a single-class run this is the class total.
    total, failures, errors, skipped = (int(g) for g in matches[-1].groups())
    passed_count = max(0, total - failures - errors - skipped)
    passed = failures == 0 and errors == 0 and total > 0

    if not passed:
        failure_messages.append(infer_collection_error(stdout, exec_result.stderr))

    return TestExecutionResult(
        passed=passed,
        total=total,
        passed_count=passed_count,
        failed_count=failures,
        error_count=errors,
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )


def _parse_gradle_test(stdout: str, exec_result: ExecutionResult) -> TestExecutionResult:
    """Parse ``./gradlew :<module>:test --console=plain`` output.

    Gradle's plain console prints, per test task, a summary line like
    ``<N> tests completed, <M> failed`` (``M`` omitted when zero) plus
    ``BUILD SUCCESSFUL`` / ``BUILD FAILED``. Individual failures appear as
    ``<fully.qualified.Class> > methodName FAILED``. When ``--tests <Class>``
    filters to a single class, the task summary reflects just that class.
    """
    completed_matches = list(
        re.finditer(r"(\d+)\s+tests completed(?:,\s*(\d+)\s+failed)?", stdout)
    )
    failed_line_re = re.compile(r"^([A-Za-z_][\w.]*?)\s*>\s*(\w+)\s+FAILED\b", re.M)
    failure_messages = [f"{m.group(1)}>{m.group(2)}" for m in failed_line_re.finditer(stdout)]

    build_success = "BUILD SUCCESSFUL" in stdout

    if not completed_matches:
        # No summary line — either zero tests ran or a build/compile error
        # aborted before the test task. Distinguish by the exit code + banner.
        if build_success and exec_result.return_code in (0, None):
            return TestExecutionResult(
                passed=True,
                total=0,
                passed_count=0,
                failed_count=0,
                error_count=0,
                failure_messages=[],
                stdout=stdout,
                stderr=exec_result.stderr,
                duration_ms=exec_result.duration_ms,
            )
        msg = infer_collection_error(stdout, exec_result.stderr)
        if not failure_messages:
            failure_messages.append(msg)
        return TestExecutionResult(
            passed=False,
            total=0,
            passed_count=0,
            failed_count=0,
            error_count=1,
            failure_messages=failure_messages,
            stdout=stdout,
            stderr=exec_result.stderr,
            duration_ms=exec_result.duration_ms,
        )

    total, failed = (
        int(completed_matches[-1].group(1)),
        int(completed_matches[-1].group(2) or 0),
    )
    passed_count = max(0, total - failed)
    passed = failed == 0 and total > 0 and build_success and exec_result.return_code in (0, None)

    if not passed and not failure_messages:
        failure_messages.append(infer_collection_error(stdout, exec_result.stderr))

    return TestExecutionResult(
        passed=passed,
        total=total,
        passed_count=passed_count,
        failed_count=failed,
        error_count=0,
        failure_messages=failure_messages,
        stdout=stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms,
    )
