#!/usr/bin/env python
"""Probe sustainable request rates for (model, provider) pairs on OpenRouter.

Some OpenRouter endpoints are served by upstream providers with very low
shared-pool rate limits (e.g. Alibaba for qwen3.8-flash); hammering them from
parallel agent workers triggers `Provider returned error 429` storms that
stall generation/tournament runs. ``config/models.yaml`` carries a
client-side ``provider_rate_limits`` throttle per (model, provider); this
script empirically estimates the requests/minute each provider tolerates so
the YAML can be filled with grounded numbers.

Method: for each (model, provider) it walks a descending staircase of
candidate RPMs. At each rate it fires ``--requests`` tiny completions
(max_tokens=16, no retries, 30s request timeout — a capacity-queued endpoint
fails fast instead of hanging on litellm's 600s default) spaced ``60/rpm``
seconds apart and counts 429s. The highest rate with zero 429s is reported as
the sustainable rate, plus a suggested (rounded-down) ``provider_rate_limits``
entry. Results are snapshot estimates — shared upstream pools vary with
global load, so prefer the conservative suggestion. Per-request progress is
printed live ("." ok / "R" rate-limited / "E" other error).

Examples:
    # Probe every configured provider of one model:
    swe-duel-probe-rate-limits --models qwen3.8-flash

    # Probe one specific provider slug:
    swe-duel-probe-rate-limits \
        --models qwen3.8-flash --providers alibaba
"""

from __future__ import annotations

import argparse
import sys
import time

from swe_duel.cli._common import setup


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "ratelimit" in text or "rate limit" in text


def _probe_rate(model_id: str, provider: str, rpm: int, n: int) -> tuple[int, float]:
    """Fire ``n`` tiny requests spaced for ``rpm``; return (n_429, elapsed_s)."""
    import litellm

    litellm.suppress_debug_info = True
    spacing = 60.0 / rpm
    n_429 = 0
    t0 = time.monotonic()
    for i in range(n):
        if i > 0:
            time.sleep(spacing)
        try:
            litellm.completion(
                model=f"openrouter/{model_id}",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with the single word: ok"},
                ],
                temperature=0.0,
                max_tokens=16,
                max_retries=0,
                timeout=30.0,
                extra_body={"provider": {"order": [provider], "allow_fallbacks": False}},
            )
            print(".", end="", flush=True)
        except Exception as e:  # noqa: BLE001
            if _is_rate_limit_error(e):
                n_429 += 1
                print("R", end="", flush=True)
            else:
                # Non-rate-limit failure (outage, auth, ...): count as a miss
                # so a broken endpoint never reports a passing rate.
                n_429 += 1
                print("E", end="", flush=True)
                print(f"\n      · non-rate-limit error: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print("", flush=True)
    return n_429, time.monotonic() - t0


def _suggest_rpm(sustainable: int | None) -> int | None:
    """Round the sustainable rate down to a conservative config value."""
    if sustainable is None:
        return None
    for nice in (60, 40, 30, 24, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1):
        if nice <= sustainable:
            return nice
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenRouter provider rate limits")
    parser.add_argument("--models", nargs="+", required=True, help="Model nicks from config/models.yaml")
    parser.add_argument("--providers", nargs="+", default=None, help="Optional provider-slug filter")
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=[60, 40, 30, 20, 12, 6],
        help="Descending candidate RPM staircase (default: 60 40 30 20 12 6).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Requests to send per candidate rate (default 10).",
    )
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--repos-dir", default="repos")
    args = parser.parse_args()

    args.repos = []
    ctx = setup(args, preflight=False)
    model_configs = ctx["model_configs"]

    nicks = [n for n in args.models if n in model_configs or n in
             {c.model_id: k for k, c in model_configs.items()}]
    if not nicks:
        print(f"[probe] no matching model nicks in {ctx['config_dir']}/models.yaml", file=sys.stderr)
        return 2

    rates = sorted({int(r) for r in args.rates}, reverse=True)
    suggestions: dict[tuple[str, str], int] = {}
    for nick in nicks:
        cfg = model_configs[nick]
        providers = list(cfg.providers or [])
        if args.providers:
            providers = [p for p in providers if p in set(args.providers)]
        if not providers:
            providers = [""]  # auto-route probe
        for provider in providers:
            label = provider or "auto-route"
            print(f"\n[probe] {cfg.model_id} @ {label}: staircase {rates}, "
                  f"{args.requests} requests per rate", flush=True)
            sustainable: int | None = None
            for i, rpm in enumerate(rates):
                if i > 0:
                    # Let the upstream pool recover from the previous (failing)
                    # rate before judging the next one — an immediate re-probe
                    # would smear the earlier 429 storm into this verdict.
                    print(f"    · settling 15s after the 429s at {rates[i - 1]} rpm…", flush=True)
                    time.sleep(15)
                n_429, elapsed = _probe_rate(cfg.model_id, provider, rpm, args.requests)
                verdict = "OK" if n_429 == 0 else f"{n_429}/{args.requests} rate-limited"
                print(f"    · {rpm:>3} rpm: {verdict}  ({elapsed:.0f}s)", flush=True)
                if n_429 == 0:
                    sustainable = rpm
                    break
            suggestion = _suggest_rpm(sustainable)
            if suggestion is None:
                print("    ✗ every candidate rate hit 429s — endpoint unusable right now")
            else:
                print(f"    ✓ sustainable ≈ {sustainable} rpm → suggest "
                      f"provider_rate_limits: {{{provider or 'default'}: {suggestion}}}")
                suggestions[(cfg.model_id, provider)] = suggestion

    if suggestions:
        print("\n[probe] Suggested config/models.yaml snippet:")
        for nick in nicks:
            cfg = model_configs[nick]
            entries = {
                p: rpm
                for (mid, p), rpm in suggestions.items()
                if mid == cfg.model_id
            }
            if not entries:
                continue
            body = ", ".join(f'"{p or "default"}": {rpm}' for p, rpm in sorted(entries.items()))
            print(f"  {nick}:")
            print(f"    provider_rate_limits: {{{body}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
