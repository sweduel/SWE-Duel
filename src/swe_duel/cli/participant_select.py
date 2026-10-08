"""Paged participant selection for challenge generation and tournaments.

A participant (competitor) is the 4-tuple (model, harness, reasoning_effort,
provider). Listing every combination on one screen would explode the vertical
space (the old one-screen checkbox showed model × harness; adding effort and
provider multiplies it further), so selection is broken into a page-per-dimension
wizard:

    1. model page    — one pick from config/models.yaml
    2. harness page  — one pick from the registered harnesses
    3. effort page   — reasoning efforts defined for the model in models.yaml,
                       filtered to what the chosen harness can express
    4. provider page — OpenRouter providers defined for the model in models.yaml
                       (only for harnesses that can pin a provider)

After a 4-tuple is chosen the wizard returns to a landing page listing the
selected participants, where the user adds another, removes the last one, or
starts the run.

Tournament callers pass ``require_slots=True`` plus the set of bank identities
(``bank_ids``) so options that have no challenge slots across the selected
repos are dimmed — a competitor without slots cannot play (its challenges would
all be opponent auto-wins).
"""

from __future__ import annotations

from dataclasses import dataclass

import questionary

from swe_duel.agents.harness import (  # noqa: E402
    HARNESS_IDS,
    harness_display_name,
    harness_supported_efforts,
    harness_supports_provider,
)
from swe_duel.config import ModelConfig  # noqa: E402
from swe_duel.models import composite_id, display_composite_id, split_composite_id  # noqa: E402


@dataclass(frozen=True)
class Participant:
    """One selected competitor = (model, harness, reasoning_effort, provider)."""

    nick: str
    model_config: ModelConfig  # with the effort/provider selection applied
    harness_id: str
    reasoning_effort: str
    provider: str

    @property
    def model_id(self) -> str:
        return self.model_config.model_id

    @property
    def cid(self) -> str:
        """Competitor identity string (model#harness[#effort#provider])."""
        return composite_id(
            self.model_id, self.harness_id, self.reasoning_effort, self.provider
        )

    @property
    def label(self) -> str:
        return display_composite_id(self.cid)


# ── option builders ────────────────────────────────────────────────────────


def _bank_has(bank_ids: set[str], cid: str) -> bool:
    return not bank_ids or cid in bank_ids


def _effort_options(cfg: ModelConfig, harness_id: str) -> list[str]:
    """Selectable effort strings (may include "" = model default).

    The yaml menu is the source of truth; "" (model default) is always offered
    as the escape hatch matching legacy pools. Efforts the harness cannot
    express are dropped (e.g. "max" under codex, everything under claude-code).
    """
    supported = harness_supported_efforts(harness_id)
    options: list[str] = []
    for e in cfg.reasoning_efforts:
        if supported is not None and e not in supported:
            continue
        options.append(e)
    if "" not in options:
        options.append("")
    return options


def _provider_options(cfg: ModelConfig, harness_id: str) -> list[str]:
    """Selectable provider slugs ("" = OpenRouter auto-routing), or just ""
    when the harness cannot pin a provider."""
    if not harness_supports_provider(harness_id):
        return [""]
    options: list[str] = list(cfg.providers)
    if "" not in options:
        options.append("")
    return options


def _effort_title(effort: str) -> str:
    return "(model default)" if not effort else f"effort: {effort}"


def _provider_title(provider: str) -> str:
    return "(OpenRouter auto-route)" if not provider else f"provider: {provider}"


# ── the wizard ─────────────────────────────────────────────────────────────


def _pick_model(
    all_models: dict[str, ModelConfig],
    model_status: dict[str, str] | None,
) -> str | None:
    choices = []
    for nick, cfg in all_models.items():
        title = f"{nick}  [{cfg.model_id}]"
        if model_status and nick in model_status:
            title += f"  — {model_status[nick]}"
        choices.append(questionary.Choice(title=title, value=nick))
    return questionary.select(
        "Page 1/4 — select the model:",
        choices=choices,
    ).ask()


def _pick_harness(
    nick: str,
    cfg: ModelConfig,
    require_slots: bool,
    bank_ids: set[str],
    selected_cids: set[str],
) -> str | None:
    choices = []
    for hid in HARNESS_IDS:
        disabled = None
        if require_slots and not any(
            _cid_prefix_matches(bid, cfg.model_id, hid) for bid in bank_ids
        ):
            disabled = "no challenge slots for this harness"
        title = f"{harness_display_name(hid)}"
        if not harness_supports_provider(hid):
            title += "  (cannot pin a provider)"
        if harness_supported_efforts(hid) == frozenset():
            title += "  (no reasoning-effort control)"
        if composite_id(cfg.model_id, hid, "", "") in selected_cids:
            title += "  (default identity already selected)"
        choices.append(questionary.Choice(title=title, value=hid, disabled=disabled))
    return questionary.select(
        f"Page 2/4 — {nick} [{cfg.model_id}]: select the agent harness:",
        choices=choices,
    ).ask()


def _cid_prefix_matches(bid: str, model_id: str, harness_id: str) -> bool:
    m, h, _e, _p = split_composite_id(bid)
    return m == model_id and h == harness_id


def _pick_effort(
    nick: str,
    cfg: ModelConfig,
    harness_id: str,
    require_slots: bool,
    bank_ids: set[str],
) -> str | None:
    options = _effort_options(cfg, harness_id)
    if len(options) == 1:
        # Single option (usually "(model default)" for harnesses without
        # effort control, or models without an effort menu) — skip the page.
        only = options[0]
        if only:
            print(f"  effort: auto-selected {_effort_title(only)}")
        else:
            print(
                "  effort: no selectable efforts for this model/harness — "
                "using (model default)"
            )
        return only
    choices = []
    for e in options:
        disabled = None
        if require_slots and not any(
            _cid_prefix_matches(bid, cfg.model_id, harness_id)
            and split_composite_id(bid)[2] == e
            for bid in bank_ids
        ):
            disabled = "no challenge slots at this effort"
        choices.append(
            questionary.Choice(title=_effort_title(e), value=e, disabled=disabled)
        )
    return questionary.select(
        f"Page 3/4 — {nick} via {harness_display_name(harness_id)}: "
        "select the reasoning effort:",
        choices=choices,
    ).ask()


def _pick_provider(
    nick: str,
    cfg: ModelConfig,
    harness_id: str,
    effort: str,
    require_slots: bool,
    bank_ids: set[str],
) -> str | None:
    if not harness_supports_provider(harness_id):
        print(
            f"  provider: {harness_display_name(harness_id)} cannot pin an "
            f"OpenRouter provider — using (OpenRouter auto-route)"
        )
        return ""
    options = _provider_options(cfg, harness_id)
    if len(options) == 1:
        print("  provider: no providers defined for this model — using (OpenRouter auto-route)")
        return options[0]
    choices = []
    for pv in options:
        disabled = None
        if require_slots and not any(
            _cid_prefix_matches(bid, cfg.model_id, harness_id)
            and split_composite_id(bid)[2] == effort
            and split_composite_id(bid)[3] == pv
            for bid in bank_ids
        ):
            disabled = "no challenge slots via this provider"
        choices.append(
            questionary.Choice(title=_provider_title(pv), value=pv, disabled=disabled)
        )
    return questionary.select(
        f"Page 4/4 — {nick} via {harness_display_name(harness_id)} "
        f"({_effort_title(effort)}): select the OpenRouter provider:",
        choices=choices,
    ).ask()


def _landing(
    participants: list[Participant],
    start_verb: str,
    slot_status: dict[str, str] | None,
) -> str | None:
    print()
    print("  ── Selected participants ─────────────────────────────")
    if not participants:
        print("  (none yet)")
    for i, p in enumerate(participants, 1):
        line = f"   {i:>2}. {p.label}"
        if slot_status and p.cid in slot_status:
            line += f"  — {slot_status[p.cid]}"
        print(line)
    print("  ─────────────────────────────────────────────────────")
    choices = [
        questionary.Choice(title="Add another participant", value="add"),
        questionary.Choice(
            title=f"{start_verb} with the {len(participants)} participant(s) above",
            value="start",
            disabled=None if participants else "no participants selected yet",
        ),
    ]
    if participants:
        choices.append(
            questionary.Choice(title="Remove the last participant", value="remove")
        )
    choices.append(questionary.Choice(title="Abort", value="abort"))
    return questionary.select(
        "What next?",
        choices=choices,
    ).ask()


def select_participants(
    all_models: dict[str, ModelConfig],
    *,
    start_verb: str = "Start",
    require_slots: bool = False,
    bank_ids: set[str] | None = None,
    model_status: dict[str, str] | None = None,
    slot_status: dict[str, str] | None = None,
    preselected: list[Participant] | None = None,
) -> list[Participant]:
    """Run the paged (model → harness → effort → provider) selection wizard.

    Returns the chosen :class:`Participant` list (possibly empty if the user
    aborts — callers decide whether empty is fatal). ``require_slots`` dims
    options with no bank identity in ``bank_ids`` (tournament mode).
    ``model_status`` / ``slot_status`` map model nick / composite id to a
    short annotation shown on the model page / landing page. ``preselected``
    seeds the landing page (used by the tournament add-participant flow, which
    diffs the result against the seed to get the newcomers).
    """
    bank = bank_ids or set()
    participants: list[Participant] = list(preselected or [])
    while True:
        action = _landing(participants, start_verb, slot_status)
        if action is None or action == "abort":
            raise SystemExit("aborted.")
        if action == "start":
            return participants
        if action == "remove":
            if participants:
                removed = participants.pop()
                print(f"  removed: {removed.label}")
            continue

        nick = _pick_model(all_models, model_status)
        if nick is None:
            raise SystemExit("aborted.")
        cfg = all_models[nick]

        harness_id = _pick_harness(
            nick, cfg, require_slots, bank, {p.cid for p in participants}
        )
        if harness_id is None:
            raise SystemExit("aborted.")

        effort = _pick_effort(nick, cfg, harness_id, require_slots, bank)
        if effort is None:
            raise SystemExit("aborted.")

        provider = _pick_provider(nick, cfg, harness_id, effort, require_slots, bank)
        if provider is None:
            raise SystemExit("aborted.")

        selected_cfg = cfg.with_selection(effort, provider)
        participant = Participant(
            nick=nick,
            model_config=selected_cfg,
            harness_id=harness_id,
            reasoning_effort=effort,
            provider=provider,
        )
        if participant.cid in {p.cid for p in participants}:
            print(f"  {participant.label} is already selected — skipping.")
            continue
        participants.append(participant)
        print(f"  + added {participant.label}")


# ── bank table printing ───────────────────────────────────────────────────


def print_pool_table(store, repo_names: list[str]) -> None:
    """Print one row per *existing* bank pool (participant identity × repo).

    Rows are (model, harness, effort, provider) identities read from the
    bank's pool keys, so the table stays bounded by what has actually been
    generated — listing every yaml-defined combination would explode the
    vertical space. Each cell is successful_slots / total_attempted_slots.
    """
    print()
    print(
        "Existing challenge-bank pools per participant identity. Each cell is "
        "successful_slots / total_attempted_slots:"
    )
    print()
    pools = store.list_pools()
    if not pools:
        print("  (bank is empty — every selection starts from zero slots)")
        print()
        return
    identities: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for model_id, harness_id, effort, provider, repo, count in pools:
        identities.setdefault((model_id, harness_id, effort, provider), {})[repo] = count
    header = f"  {'model / harness / effort / provider':<64}  " + "  ".join(
        f"{rn:>18}" for rn in repo_names
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (model_id, harness_id, effort, provider), per_repo in identities.items():
        label = display_composite_id(
            composite_id(model_id, harness_id, effort, provider)
        )
        cells = []
        for rn in repo_names:
            succ = per_repo.get(rn, 0)
            total = store.distinct_slot_count(
                model_id, rn, harness_id, effort, provider
            )
            cells.append(f"{succ:>3} / {total:>3} slots")
        print(f"  {label:<64}  " + "  ".join(f"{c:>18}" for c in cells))
    print()


# ── bank status (slots / successes per participant identity) ─────────────


def bank_status(
    store,
    model_configs: dict[str, ModelConfig],
    repo_names: list[str],
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Derive wizard annotations from the challenge bank.

    Returns ``(bank_ids, model_status, slot_status)``:

    * ``bank_ids`` — composite ids of every participant with ≥1 attempted slot
      (successful OR failed) across ``repo_names``; tournament mode dims
      everything else.
    * ``slot_status`` — composite id → ``"{succ}/{n} repos w/ challenges,
      {slots} w/ slots"`` (per-identity, matching the pre-wizard table text).
    * ``model_status`` — model nick → the same summary aggregated over every
      identity of that model.
    """
    repos = set(repo_names)
    n_repos = len(repo_names)
    with_slots: dict[str, set[str]] = {}
    with_success: dict[str, set[str]] = {}

    for entry in store.index.get("entries", {}).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("repo_name") not in repos:
            continue
        cid = composite_id(
            str(entry.get("red_model_id", "")),
            str(entry.get("red_harness_id", "mini-swe-agent") or "mini-swe-agent"),
            str(entry.get("red_reasoning_effort", "") or ""),
            str(entry.get("red_provider", "") or ""),
        )
        with_slots.setdefault(cid, set()).add(str(entry.get("repo_name")))

    for model_id, harness_id, effort, provider, repo, _count in store.list_pools():
        if repo not in repos:
            continue
        cid = composite_id(model_id, harness_id, effort, provider)
        with_success.setdefault(cid, set()).add(repo)
        with_slots.setdefault(cid, set()).add(repo)

    bank_ids = set(with_slots)

    def _fmt(slots: int, succ: int) -> str:
        return f"{succ}/{n_repos} repos w/ challenges, {slots} w/ slots"

    slot_status = {
        cid: _fmt(len(rs), len(with_success.get(cid, set())))
        for cid, rs in with_slots.items()
    }

    # Aggregate per model nick (across harness/effort/provider identities).
    model_slots: dict[str, set[str]] = {}
    model_succ: dict[str, set[str]] = {}
    nick_by_model = {cfg.model_id: nick for nick, cfg in model_configs.items()}
    for cid, rs in with_slots.items():
        model_id, _h, _e, _p = split_composite_id(cid)
        nick = nick_by_model.get(model_id)
        if nick is None:
            continue
        model_slots.setdefault(nick, set()).update(rs)
        model_succ.setdefault(nick, set()).update(
            rs & with_success.get(cid, set())
        )
    model_status = {
        nick: _fmt(len(rs), len(model_succ.get(nick, set())))
        for nick, rs in model_slots.items()
    }
    return bank_ids, model_status, slot_status


# ── non-interactive (CLI) resolution ─────────────────────────────────────


def resolve_cli_participants(
    all_models: dict[str, ModelConfig],
    models: list[str],
    harnesses: list[str],
    efforts: list[str] | None,
    providers: list[str] | None,
) -> list[Participant]:
    """Cross-product the --models/--harnesses/--efforts/--providers CLI lists.

    ``efforts``/``providers`` default to the "" (default identity) selection so
    bare `--models X --harnesses Y` keeps its legacy meaning. Each selection is
    validated against the model's yaml menu and the harness's capabilities.
    """
    from swe_duel.agents.harness import HARNESS_IDS as _HARNESS_IDS

    matched: list[tuple[str, ModelConfig]] = []
    for nick, cfg in all_models.items():
        if nick in models or cfg.model_id in models:
            matched.append((nick, cfg))
    if not matched:
        raise SystemExit(f"none of --models {models!r} found in config/models.yaml")

    bad = [h for h in harnesses if h not in _HARNESS_IDS]
    if bad:
        raise SystemExit(f"Unknown harness(es) {bad}; known: {list(_HARNESS_IDS)}")

    effort_list = list(efforts) if efforts else [""]
    provider_list = list(providers) if providers else [""]

    out: list[Participant] = []
    for nick, cfg in matched:
        for hid in harnesses:
            for e in effort_list:
                for pv in provider_list:
                    if e and e not in cfg.reasoning_efforts:
                        raise SystemExit(
                            f"reasoning effort {e!r} is not in the models.yaml menu "
                            f"for {nick} ({cfg.model_id}); menu: {cfg.reasoning_efforts}"
                        )
                    supported = harness_supported_efforts(hid)
                    if e and supported is not None and e not in supported:
                        raise SystemExit(
                            f"harness {hid!r} cannot express reasoning effort {e!r} "
                            f"(supported: {sorted(supported) or 'none'})"
                        )
                    if pv and pv not in cfg.providers:
                        raise SystemExit(
                            f"provider {pv!r} is not in the models.yaml menu for "
                            f"{nick} ({cfg.model_id}); menu: {cfg.providers}"
                        )
                    if pv and not harness_supports_provider(hid):
                        raise SystemExit(
                            f"harness {hid!r} cannot pin an OpenRouter provider "
                            f"(requested {pv!r})"
                        )
                    out.append(
                        Participant(
                            nick=nick,
                            model_config=cfg.with_selection(e, pv),
                            harness_id=hid,
                            reasoning_effort=e,
                            provider=pv,
                        )
                    )
    return out


def participant_model_configs(participants: list[Participant]) -> dict[str, ModelConfig]:
    """Model-config map keyed by participant composite id (with selection)."""
    return {p.cid: p.model_config for p in participants}
