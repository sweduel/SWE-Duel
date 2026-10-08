# Harness guide — SWE-Duel agent runtimes

A **harness** is a concrete coding-agent runtime that, given a model and a task
prompt, autonomously explores a workspace, edits files, runs shell commands,
and returns an `AgentTrajectory`. Competitors are identified by the 4-tuple
`(model_id, harness_id, reasoning_effort, provider)` — the same model under two
harnesses, at two reasoning efforts, or via two OpenRouter providers is three
distinct entrants. The effort/provider parts ride on the bound `ModelConfig`
(see `ModelConfig.with_selection`); each harness forwards what it can:

| harness | reasoning effort | provider pin | rate-limit enforcement |
| --- | --- | --- | --- |
| mini-swe-agent | any string (litellm `extra_body` → `reasoning.effort`) | yes (`provider.order`, no fallbacks) | yes — `RateLimitedModelProxy` around the model |
| OpenHands | any string (`litellm_extra_body`) | yes | yes — `LLM` subclass from `make_rate_limited_llm_class` |
| codex | `model_reasoning_effort` enum only (minimal/low/medium/high/xhigh) | no | no (CLI calls the API from inside the container) |
| claude-code | none | no | no (CLI calls the API from inside the container) |

`HARNESS_EFFORT_SUPPORT` / `HARNESS_PROVIDER_SUPPORT` / `HARNESS_RATE_LIMIT_SUPPORT`
in `agents/harness/base.py` are the machine-readable form of this table;
`get_harness` raises on an effort/provider selection the harness cannot honour
instead of silently dropping it. Rate limits are different: they are traffic
shaping, not competitor identity, so an unenforceable limit is **warned**
(`warn_unenforceable_rate_limits`, printed pre-Live by every TUI script) rather
than rejected — the CLI harnesses' own retry loops still absorb transient
provider 429s. Limits live in `config/models.yaml` under `provider_rate_limits:`
(provider slug → requests/min, `"default"` for auto-route), are process-wide and
**divided by all parallel workers**, and are probed/refreshed with
`swe-duel-probe-rate-limits` (bundled as `swe_duel/cli/probe_provider_rate_limits.py`). Any harness that makes its LLM calls
from the host process should acquire the participant's throttle
(`rate_limit.throttle_for(model_config)`) before every completion call, and call
`install_rate_limit_log_filter()` + `set_thread_noise_absorbed(not console_echo)`
at run start so 429 retry chatter is mirrored to `data/logs/model_retries.log`
instead of corrupting the live TUI.

Score-side code never imports a specific runtime. It always does:

```python
from swe_duel.agents.harness import get_harness
h = get_harness(harness_id, model_config)
traj = h.run(workspace_path, task_prompt, docker_image=..., ...)
```

All four harnesses honour the same `AgentHarness.run` contract (step budget +
3×6 recovery turns, wall-clock, `step_callback` progress, `_swe-duel/*.log`
streaming, normalized `AgentTrajectory`).

---

## The four harnesses at a glance

| | **mini-swe-agent** | **OpenHands** | **Codex** | **Claude Code** |
| --- | --- | --- | --- | --- |
| Harness id | `mini-swe-agent` | `openhands` | `codex` | `claude-code` |
| Module | `harness/mini_swe.py` | `harness/openhands.py` | `harness/codex.py` | `harness/claude_code.py` |
| Where the agent loop runs | Host process (Python) | Host process (Python SDK) + **agent-server in container** | **CLI inside container** | **CLI inside container** |
| Shell / edits execute in | Per-repo Docker image via `docker exec` | Same image via agent-server HTTP (tools run server-side) | Same image (CLI is in-container) | Same image (CLI is in-container) |
| Host → container transport | mini-swe `DockerEnvironment` | OpenHands `DockerWorkspace` | Long-lived `sleep infinity` container (`cli_container.py`) | Same as Codex |
| Non-interactive entrypoint | mini-swe `DefaultAgent` loop | SDK `Conversation.run()` | `codex exec --json …` | `claude --bare -p … --output-format stream-json` |
| Model routing | litellm → OpenRouter (`openrouter/<model_id>`) | OpenHands `LLM` → OpenRouter base URL | Custom Codex provider `openrouter` (config.toml under `CODEX_HOME`) | Anthropic-compatible gateway: `ANTHROPIC_BASE_URL=https://openrouter.ai/api` |
| Auth env | `SWE_DUEL_OPENROUTER_API_KEY` / `OPENROUTER_API_KEY` (litellm) | `SWE_DUEL_OPENROUTER_API_KEY` forwarded into container | `OPENROUTER_API_KEY` / `CODEX_API_KEY` / `OPENAI_API_KEY` (all set to the OpenRouter key) | `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` (the OpenRouter key) |
| Baked into image | Nothing extra (uses image toolchain only) | OpenHands agent-server (`install-agent-server.sh` or pip on `swe-duel-base`) | `codex` CLI (`install-cli-agents.sh`) | `claude` CLI (`install-cli-agents.sh`) |
| Container entrypoint mode | Verbatim (`sleep infinity`) | `--host …` → `agent-server` | Verbatim | Verbatim |
| Recovery | Extend step budget + `agent.step()` with reminder | `send_message` + `run()` | re-nudge within initial `max_steps`, then +6 recovery turns | re-nudge within initial `max_steps`, then `--resume` +6 |
| Step signal | mini-swe n_calls / assistant messages | `ActionEvent` count | JSONL `item.completed` (command_execution / file_change / …) | stream-json `assistant` tool_use blocks |
| Permissions in automation | Free shell (agent proposes bash) | Terminal + FileEditor tools unrestricted | `--sandbox danger-full-access` + `approval_policy=never` | `--dangerously-skip-permissions` / bare mode |
| Container name | `swe-duel-<repo>-mini-swe-agent-<pid>-<uuid8>` | `swe-duel-<repo>-openhands-…` (rename after OH start) | `swe-duel-<repo>-codex-…` | `swe-duel-<repo>-claude-code-…` |

Shared infrastructure:

- Bind-mount host workspace at `/workspace` (`CONTAINER_WORKSPACE`).
- `preserve_paths` become anonymous `-v /workspace/<sub>` volumes so image-baked
  dirs (`node_modules`, …) show through the bind mount.
- Dual-mode entrypoint `docker/swe-duel-entrypoint.sh`: args starting with `-` (or
  empty) → OpenHands `agent-server`; anything else exec'd verbatim.
- Host process requires `SWE_DUEL_OPENROUTER_API_KEY` (OpenRouter key); see `AgentHarness.__init__`.

---

## Trajectory step shape (mandatory for every harness)

HTML dumps (`_render_trajectory_steps` in `sandbox/diff_utils.py`), failure
artefacts, and analysis all assume the **mini-swe-agent** trajectory layout.
Every harness must produce the same structure — do not invent a second model.

### Step dict fields

Each step in `AgentTrajectory.steps` is a dict with at least:

```python
{
    "observation": str,   # what the agent *saw before* this thought/action
    "thought": str,       # model reasoning / internal monologue for this step
    "action": str,        # tool call / shell command / edit for this step
    "input_tokens": int,
    "output_tokens": int,
    "cached_tokens": int,   # optional but preferred when available
    "cost_usd": float,
    "step_seconds": float,
    "cum_seconds": float,
}
```

Renderers **always** show all three of Observation / Thought / Action (empty
values appear as `(empty)`). Omitting a key or systematically leaving `thought`
blank looks like a UI bug even when the agent is working correctly.

### Ordering: preceding observation (not following)

Chronological order for each step is **observation → thought → action**:

```
step 1:  observation = <task prompt>
         thought     = first reasoning
         action      = first tool call
         ── tool runs, yields tool_output_1 ──

step 2:  observation = tool_output_1          # result of *previous* action
         thought     = next reasoning
         action      = next tool call
         …
```

**Seed step 1’s observation with `task_prompt`.** Without that seed the first
captured tool call looks “offset by one” — its shell output is shown as if it
were the prompt, and the real prompt never appears in the trajectory HTML.

**Do not** attach a tool result to the same step that issued the tool call
(`obs = my_own_stdout`). That was the zip-offset bug fixed for Codex /
OpenHands / Claude Code: hold the result in a `next_observation` buffer (or
`pending_observation`) and only write it onto the *following* step.

Reference implementation: `MiniSweHarness._extract_trajectory` pairs each
assistant message with the user/tool content that **preceded** it; the initial
user task message is the observation for step 1.

### Filling `thought`

Whatever native channel the runtime uses, flatten it into `step["thought"]`:

| Harness | Pull thinking from |
| --- | --- |
| mini-swe-agent | assistant message `content` (+ `reasoning_content` / response payload) |
| OpenHands | `ActionEvent.thought`, `reasoning_content`, `thinking_blocks`, `responses_reasoning_item`, `summary` (see `_event_thought`) |
| Codex | JSONL `item` types `reasoning` / `agent_message` / `message` (`text` field); buffer until the next tool step, flush trailing final message onto the last step |
| Claude Code | `assistant` content blocks of type `text` / `thinking` **before** each `tool_use` |

Trailing “final answer” messages that arrive **after** the last tool call
should be **appended to the last step’s thought** (or become a thought-only
step if there are no steps yet). Do not drop them — that is how “no Thought”
appears in the HTML for otherwise successful runs.

### Per-runtime mapping cheatsheet

| Runtime event | Becomes |
| --- | --- |
| Task prompt / initial user message | `observation` of step 1 |
| Prior tool stdout / `ObservationEvent` / `tool_result` | `observation` of the **next** step |
| Model reasoning / text / thinking blocks | `thought` of the current step |
| Shell / file edit / MCP tool call | `action` of the current step |
| Usage on `turn.completed` / metrics / `result` | token + cost fields on the latest step |

### Checklist when adding a harness

When normalizing events into `AgentTrajectory`:

1. Seed with `task_prompt` as step‑1 observation.
2. One agent/tool call → one step (`total_steps` matches visible steps).
3. Carry `next_observation` / `pending_observation` forward; never zip
   (action\_i, result\_i) onto the same step.
4. Populate **thought** from every reasoning surface the SDK/CLI exposes.
5. Leave empty strings rather than deleting keys when a channel is missing.
6. Unit-test the first step: `steps[0]["observation"] == task_prompt` and
   `steps[0]["action"]` is the first tool call; tool stdout is on
   `steps[1]["observation"]` (or in a pending buffer if only one step).

Unit tests that lock this in: `tests/test_cli_harnesses.py`,
`tests/test_openhands_harness.py::test_extract_trajectory_maps_events_and_metrics`.

---

## Per-harness notes

### 1. mini-swe-agent (`mini-swe-agent`)

- Original SWE-Duel harness (still aliased as `AgentWrapper`).
- Config loaded from mini-swe's bundled `mini.yaml`; model name is
  `model_config.openrouter_model_id` (prefixed `openrouter/…`).
- Console noise silenced via `MSWEA_SILENT_STARTUP=1` + logger suppression so
  the generate/match TUI's `rich.live.Live` is not corrupted.

### 2. OpenHands (`openhands`)

- Heavy SDK imports are lazy (selecting other harnesses never pays for them).
- Agent-server **must** be baked into every repo image (Python via pip on
  `swe-duel-base`; non-Python via `docker/install-agent-server.sh` which builds a
  standalone CPython 3.12 venv at `/opt/oh`).
- Persistence redirected with `OH_CONVERSATIONS_PATH` / `OH_BASH_EVENTS_DIR`
  outside `/workspace` so root-owned server state never lands on the host bind
  mount (would corrupt diffs and crash cleanup).
- Console noise: `OPENHANDS_SUPPRESS_BANNER`, `LOG_LEVEL=WARNING`, quieted
  `execute_command`, `detach_logs=False`.
- Trajectory: `_extract_trajectory(..., task_prompt=task_prompt)` uses the
  **preceding-observation** model (seed step 1 with the prompt; each
  `ObservationEvent` becomes the *next* step’s observation). Pull thought from
  every `ActionEvent` reasoning field via `_event_thought` — not just
  `thought` / `reasoning_content`.

### 3. Codex (`codex`)

- Docs: [Non-interactive mode](https://developers.openai.com/codex/non-interactive-mode)
  (`codex exec`).
- Host module writes `/var/tmp/swe-duel-codex/config.toml` inside the container
  (`CODEX_HOME` must **not** be under `/tmp` — Codex refuses tempdirs for helper
  binaries):

  ```toml
  model = "<model_id from models.yaml>"
  model_provider = "openrouter"
  approval_policy = "never"
  sandbox_mode = "danger-full-access"

  [model_providers.openrouter]
  name = "OpenRouter"
  base_url = "https://openrouter.ai/api/v1"
  env_key = "OPENROUTER_API_KEY"
  wire_api = "responses"   # "chat" removed in Codex 0.144+
  ```

- Initial / recovery call:
  ```bash
  codex exec --json --skip-git-repo-check \
    --sandbox danger-full-access \
    --dangerously-bypass-approvals-and-sandbox \
    --model <id> "<prompt>"
  ```
  Recovery first re-nudges **within the remaining initial `max_steps` budget**
  if Codex exits early (mirrors OpenHands); only after that ceiling is spent do
  the constrained +6 recovery turns begin.
- Trajectory from JSONL (`item.*`, `turn.completed` usage). Buffer
  `reasoning` / `agent_message` into `thought`; on `command_execution` /
  `file_change` completeness, emit a step with **preceding** observation
  (step 1 = task prompt) and queue the tool stdout as `next_observation`.
  Flush a trailing final message onto the last step’s thought.

### 4. Claude Code (`claude-code`)

- Docs: [Headless / non-interactive](https://code.claude.com/docs/en/headless)
  (`claude -p`).
- OpenRouter via Claude's LLM-gateway hook:

  ```bash
  ANTHROPIC_BASE_URL=https://openrouter.ai/api
  ANTHROPIC_AUTH_TOKEN=$SWE_DUEL_OPENROUTER_API_KEY
  ANTHROPIC_API_KEY=$SWE_DUEL_OPENROUTER_API_KEY   # bare mode still wants a non-empty key
  ```

- Initial call:

  ```bash
  claude --bare -p "<prompt>" \
    --output-format stream-json --verbose \
    --dangerously-skip-permissions \
    --model <model_id> \
    --allowedTools "Bash,Read,Edit,Write,MultiEdit,Glob,Grep"
  ```

- Recovery uses `--resume <session_id>` captured from stream-json envelopes.
- Trajectory: each `assistant` `tool_use` (or pure-text turn) → one step with
  **preceding** observation (seed first turn with `task_prompt`); text /
  thinking blocks → `thought`; `user.tool_result` fills `next_observation`
  for the *following* step (not the current one). Cost: a gateway-reported
  `usage.cost` wins when the CLI passes it through; otherwise token counts
  (per-message, else the cumulative `result.usage`) are priced with the cached
  OpenRouter per-token rates (`cost_tracking.fallback_cost_for`). The CLI's
  own `result.total_cost_usd` (Anthropic price-table estimate) is ignored.

---

## Shared CLI container helper

`src/swe_duel/agents/harness/cli_container.py` is shared by Codex and Claude Code:

- `CliContainer.start()` / `cleanup()` — long-lived `docker run … sleep infinity`
  with swe-duel- labels and `swe-duel-…` name prefix.
- `run_command(argv)` — `docker exec` (or host subprocess) with line-streamed
  stdout/stderr, wall-clock timeout, and cooperative stop (step budget) via
  process kill.
- `write_text(path, content)` — drop small config files into the container
  before launching the CLI.

Host unit tests pass `docker_image=None` so the CLI path can be unit-tested
without Docker when convenient.

---

## Image install surface

| Script | Purpose | Used by |
| --- | --- | --- |
| `docker/install-agent-server.sh` | Standalone py3.12 + OpenHands agent-server at `/opt/oh` | Every **non-Python** repo Dockerfile |
| `docker/install-cli-agents.sh` | Portable Node 22 (if needed) + global `@openai/codex` + `@anthropic-ai/claude-code` | `docker/base` **and** every non-Python Dockerfile |
| `docker/swe-duel-entrypoint.sh` | Dual-mode ENTRYPOINT | Every image |

Python repo images (`flask`, `jinja`, `sqlalchemy`, `mock`) inherit CLIs
from `swe-duel-base`. After any install-script change, rebuild:

```bash
make build-docker
# or a single image/repo subset: swe-duel setup docker --only <repo>
```

Pinned CLI package versions are env-overridable at build time:

```bash
SWE_DUEL_CODEX_PKG=@openai/codex@0.144.5
SWE_DUEL_CLAUDE_PKG=@anthropic-ai/claude-code@2.1.212
SWE_DUEL_NODE_VERSION=22.14.0
```

---

## How to add a fifth harness

Follow this checklist. Keep the rest of SWE-Duel (Red/Blue roles, gates, match
orchesrator) unaware of the new runtime — they only speak `AgentHarness`.

### 1. Implement `AgentHarness`

Create `src/swe_duel/agents/harness/<name>.py`:

```python
class MyHarness(AgentHarness):
    harness_id = "my-harness"

    def run(self, workspace_path, task_prompt, *, …) -> AgentTrajectory:
        ...
```

Hard requirements of `run`:

1. When `docker_image` is set, execute **inside that image** with the host
   workspace at `/workspace`. Use `preserve_paths` (anonymous volumes) and
   host uid/gid so edits persist and image-baked deps remain visible.
2. Honour `max_steps` for the initial pass, then up to `max_recovery_turns`
   (default 3) of **6 steps each** while `completion_check()` is non-empty —
   feed reminders via `reminder_builder(missing)`.
3. Call `step_callback(current, limit)` once per agent step when provided.
4. Write reasoning to `log_file` even when `console_echo=False` (TUI owns
   stdout).
5. On wall-clock breach return `exit_status="WallClockTimeout"`.
6. On success with no missing artifacts return `exit_status="Submitted"`.
7. Normalize trajectory into the mini-swe step dict shape
   (`observation` / `thought` / `action` / tokens / cost / timings) **with the
   preceding-observation ordering** documented in
   [Trajectory step shape](#trajectory-step-shape-mandatory-for-every-harness):
   seed step 1 with `task_prompt`, attach tool results to the *next* step,
   and never leave thought empty when the runtime emitted reasoning.
8. Name containers with `swe_duel_container_name(repo_from_image(image), self.harness_id)`.
9. Never share harness instances across threads — per-run mutable state lives
   on the instance or is constructed inside `run`.
10. Source per-step cost dynamically (`cost_tracking.py`): prefer the
    gateway-reported `usage.cost` when your runtime can see the LLM API
    response body (`response_reported_cost`), else price token counts with the
    cached OpenRouter per-token rates (`fallback_cost_for`). Never hardcode
    per-token prices in a harness or config.

Prefer:

- **Library-on-host + remote tools in container** (mini-swe / OpenHands style)
  when there is a Python SDK.
- **CLI-in-container** (`cli_container.py`) when the agent is a standalone
  binary that owns its own loop (Codex / Claude Code style). Always prefer
  non-interactive / headless flags so the harness can drive recovery and
  budgets without a TTY.

### 2. Register it

In `src/swe_duel/agents/harness/base.py`:

```python
HARNESS_IDS = (…, "my-harness")
_HARNESS_DISPLAY["my-harness"] = "My Harness"

def _my_factory(model_config):
    from swe_duel.agents.harness.my_harness import MyHarness
    return MyHarness(model_config)

_HARNESS_FACTORIES["my-harness"] = _my_factory
```

Keep the factory imports lazy so unused heavy deps stay off the critical path.

### 3. Bake any binaries into the images

- Prefer a single `docker/install-<thing>.sh` that works on both `swe-duel-base`
  (Python slim) and non-Python bases (Go/Node/Java/C).
- Call it from `docker/base/Dockerfile` **and** every Dockerfile that already
  runs `install-agent-server.sh`. Images that `FROM swe-duel-base` need no extra
  line.
- Do **not** change the dual-mode entrypoint dispatch unless the new harness
  needs a third launch mode. Prefer "run as a normal command" (Codex/Claude
  Code) so gates, mini-swe, and the new harness all share the exec path.
- Rebuild: `make build-docker`.

### 4. Route auth + models through OpenRouter

SWE-Duel's only required secret is `SWE_DUEL_OPENROUTER_API_KEY` (OpenRouter). Map it into whatever
the new runtime expects (`OPENAI_API_KEY`, custom provider `env_key`,
`ANTHROPIC_AUTH_TOKEN`, …). Prefer a custom provider / base URL over
hard-coding a single vendors model list so every entry in `config/models.yaml`
remains selectable.

### 5. Tests

Add pure unit tests (no Docker/LLM) covering:

- registry returns the new id (`HARNESS_IDS`, `get_harness`, display name);
- whatever event/log parse path produces `AgentTrajectory` steps;
- `exit_status` mapping (Submitted / LimitsExceeded / WallClockTimeout).

Update existing registry assertions in `tests/test_openhands_harness.py`.

### 6. Docs & CLIs

- Update this file (`harnesses.md`) with a fifth column/row.
- Update `AGENTS.md` + `.claude/CLAUDE.md` one-liners that enumerate harnesses.
- Expand `--harness*` help strings in `swe_duel/cli/run_match.py`,
  `swe_duel/cli/evaluate_blue.py`, `swe_duel/cli/run_tournament*.py`,
  `swe_duel/cli/generate_challenges.py` (selectors already iterate `HARNESS_IDS`).
- If the harness changes image contents, note the rebuild step in
  `new_repo.md`'s Dockerfile templates.

### 7. Lint / typecheck

```bash
./venv/bin/ruff check src/swe_duel/agents/harness tests/test_cli_harnesses.py
./venv/bin/mypy src/swe_duel/agents/harness
./venv/bin/pytest tests/test_openhands_harness.py tests/test_cli_harnesses.py -v
```

---

## Quick reference: selecting a harness

```bash
# Generation (required with --models)
./venv/bin/python scripts/generate_challenges.py \
  --models claude-opus-4.8 \
  --harnesses mini-swe-agent openhands codex claude-code \
  --repos flask --target-per-repo 1

# Single match
./venv/bin/python scripts/run_match.py \
  --model-a … --harness-a codex \
  --model-b … --harness-b claude-code \
  …

# Tournaments: multi-select harnesses per model; each pair is a competitor
```

Interactive TUIs multi-select harnesses from `HARNESS_IDS` automatically —
adding a registry entry is enough for them to appear.

---

## Design principles (do not violate)

1. **Same image for agents and gates.** Scoring must never see a different
   toolchain than the agent did.
2. **No LLM-as-judge.** Harnesses only produce trajectories + file edits;
   gates and the scorer run tests.
3. **Explicit harness selection.** There is no default harness.
4. **Competitor = (model, harness, effort, provider).** Pool keys, defense
   cache, Swiss seats, and ratings all key on the composite id.
5. **Cleanup is mandatory.** Long-lived containers use the `swe-duel-` name prefix
   and session labels so a crash cannot leak co-tenant containers.
