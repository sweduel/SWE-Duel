You are a software engineer contributing a new feature to the {{ repo_name }} project.

The project is cloned at {{ workspace_path }}.

This is a **{{ lang }}** project. You MUST write the feature AND its tests in the
project's own language using **{{ test_framework }}** — do NOT write tests in any
other language (e.g. no Python/pytest wrappers for a Go or Node project).
{% if previous_gists %}
DIVERSITY CONSTRAINT — previously generated challenges for this repo are listed below.
You MUST pick a DIFFERENT target module from every listed feature.
Avoid features that are conceptually similar (e.g. another variant of formatting, 
another serializer tweak, another caching helper) to what has already been done.

Previous challenges:
{% for g in previous_gists %}
- feature location(s): {{ g.target_files }}
  bug location(s): {{ g.bug_location }}
  bug_type: {{ g.bug_type }}
  feature_spec: {{ g.feature_spec }}
{% endfor %}
{% endif %}

BUDGET GUIDANCE — you have a limited number of steps. Be efficient:
  - Skim, don't exhaustively cat every file. 1-3 exploration steps is usually enough.
  - Edit files in place. Do NOT create backup copies anywhere (not in /tmp, not in
    the workspace, not next to the file as `.backup` / `.bak` / `.orig` / `.original`
    / `.old` / `.new` / `.save` / `.prev`). If you want to remember the pre-edit
    version, keep it in your own working memory — do NOT write it to disk. Any such
    file will be deleted by the harness and if it survives it will trivially leak
    the change to the reviewer.
  - Spend the bulk of your steps on: (a) writing the feature, (b) writing the feature
    tests, (c) writing `_swe-duel/metadata.json`.

THIS IS PHASE 1 OF 2 — FEATURE GENERATION.
In this phase you must implement a new feature and write tests for it. 
A separate subsequent phase will ask you to do that.
The feature you write here MUST be plausible.

Your task:
1. RUN the original test suite to check for any pre-existing failures.
   - DO NOT fix any pre-existing bugs found.
   - This step is only to make sure that the newly implemented feature in the next step is not related to any pre-existing bugs.
   - RECORD the exact identifiers of any pre-existing failures — you MUST list
     them in `_swe-duel/metadata.json` under the `pre_existing_failures` field
     (use the native test id, e.g. `tests/test_x.py::test_y` for Python or the
     `Package::TestName` for Go). If there are none, set this field to `[]`.
2. EXPLORE the codebase to understand its structure, key modules, test suite, and conventions.
3. SELECT a single source file (module) that would benefit from a new feature.
   - Prefer files with good test coverage and clear interfaces.
   - Avoid configuration files, package-init/setup files, or build manifests.
4. IMPLEMENT a non-trivial new feature in that module.
   - The feature must be genuinely useful (not cosmetic or trivial).
   - It must follow existing code conventions.
   - It must not break any existing tests.
   - It must be CORRECT (no intentional bug in this phase).
5. WRITE tests for your feature in the project's language.

COMPLEXITY REQUIREMENTS — your feature is checked by an automated complexity gate
BEFORE the next phase runs. A feature that is correct and useful but too small will
be REJECTED and the whole attempt is wasted. To pass the gate you MUST satisfy ALL
of the following (these are hard minimums — comfortably exceed them, do not aim
exactly at the floor):
  - The feature's source-code change (the diff of your edits, EXCLUDING the test
    file and `_swe-duel/` files) must ADD at least **{{ min_diff_lines }} new lines**.
    A one-line helper or a trivial wrapper will NOT pass. Implement enough genuine
    logic — input handling, edge cases, multiple branches — that the change is
    substantive. Aim for noticeably more than {{ min_diff_lines }} lines of real code.
  - Your feature test file must contain at least **{{ min_test_functions }} test
    function(s)** and at least **{{ min_test_assertions }} assertion(s)** in total.
    (The completion checklist below asks for more than this — meet the larger number.)
  - Choose a feature whose scope naturally justifies this size. If the module only
    admits a trivial addition, pick a DIFFERENT module or a richer feature rather
    than padding with dead code — reviewers and gates can tell.

HOW TO WRITE & RUN THE FEATURE TESTS FOR THIS PROJECT:
{{ test_instructions }}

When you are done, create these files:
- `_swe-duel/metadata.json` with this exact schema:
      ```json
      {
        "target_files": ["relative/path/to/source_file"],
        "exploration_summary": "Why you chose this module and what you learned",
        "feature_spec": "Human-readable description of the feature",
        "feature_rationale": "Why this feature belongs in this module",
        "pre_existing_failures": ["pre-existing failing test ids", "..."]
      }
      ```
- `_swe-duel/{{ feature_test_filename }}`: a complete test file (in the project's
  language, per the instructions above) exercising your new feature.
  Must have at least 3 test functions with at least 3 assertions total.
  These tests MUST pass against your (correct) feature implementation.

COMPLETION CHECKLIST — before you submit, you MUST verify ALL of these exist in the workspace:
  1. `_swe-duel/metadata.json` — populated with target_files, exploration_summary,
     feature_spec, feature_rationale.
  2. `_swe-duel/{{ feature_test_filename }}` — at least 3 test functions, and you
     have RUN them and confirmed they PASS against your feature.
  3. Your source-file edits are saved (correct feature implementation).

The `_swe-duel/` directory already exists in the workspace; just write files into it.
Do NOT submit the task before these files exist. Submitting without them discards the entire challenge.

Write the files first (use `cat <<'EOF' > _swe-duel/metadata.json … EOF` etc.), then run `ls _swe-duel/` and `cat _swe-duel/metadata.json` to verify, and ONLY THEN submit using the exact command on its own line:

    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
