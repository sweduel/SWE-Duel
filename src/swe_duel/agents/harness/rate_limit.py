"""Client-side request-rate throttling and rate-limit noise absorption.

Two problems this module solves for provider-pinned participants on heavily
rate-limited OpenRouter endpoints (e.g. Alibaba serving ``qwen3.8-flash`` —
an upstream **shared pool** with very low capacity):

1. **429 storms stall runs.** Several parallel agent workers (generation
   pools, tournament defense rounds) sending requests to the same pinned
   provider trip ``Provider returned error 429``; the harnesses' internal
   retry loops then spend minutes in exponential backoff per LLM call, and a
   persistent storm burns whole generation attempts. The fix is a
   **process-wide throttle** configured per (model, provider) in
   ``config/models.yaml`` under ``provider_rate_limits`` (requests/minute,
   special key ``"default"`` for auto-route/unlisted slugs — probe grounded
   values with ``scripts/probe_provider_rate_limits.py``). The throttle is a
   reservation-based min-spacing limiter shared by **all threads**, so
   parallel workers automatically divide one budget: with a 30 rpm cap and 5
   workers, each request start is spaced ≥2s apart no matter which worker
   sends it. Cross-process coordination is intentionally out of scope
   (one generation/tournament run = one process).

2. **429 messages corrupt the live TUI.** The retry machinery logs the full
   provider error through stdlib logging — tenacity's ``before_sleep`` via
   the ``litellm_model`` logger (mini-swe) and ``RetryMixin.log_retry_attempt``
   via the ``openhands`` tree (OpenHands) — which reaches stderr (root
   handler / ``lastResort``) from worker threads mid-``rich.live.Live`` and
   breaks the progress bar's cursor tracking. :func:`install_rate_limit_log_filter`
   attaches a filter that mirrors rate-limit records to a dedicated file
   (``data/logs/model_retries.log``, env ``SWE_DUEL_MODEL_RETRY_LOG``) and drops
   them from the console **for threads running under a live TUI**
   (:func:`set_thread_noise_absorbed` — set from ``AgentHarness.run`` with
   ``not console_echo``). Console (non-TUI) runs keep seeing retry warnings.

Throttling is enforced where the harness itself makes LLM calls on the host:
mini-swe-agent (:class:`RateLimitedModelProxy` wraps the model object) and
OpenHands (a rate-limited ``LLM`` subclass built by
:func:`make_rate_limited_llm_class`). The Codex / Claude Code CLIs run
in-container and make their own API calls, so a configured limit cannot be
honoured there — :data:`swe_duel.agents.harness.base.HARNESS_RATE_LIMIT_SUPPORT`
declares this and :func:`warn_unenforceable_rate_limits` surfaces it at TUI
startup (the CLI harnesses' own retry loops still absorb transient 429s).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from swe_duel.config import ModelConfig

# ── throttle registry ─────────────────────────────────────────────────────


class _ProviderThrottle:
    """Reservation-based min-spacing limiter, shared by all threads.

    ``acquire()`` hands out future time slots spaced ``60 / rpm`` seconds
    apart under a lock, then the caller sleeps its slot's remaining delay
    outside the lock — a fixed queue with no burst credit, which is exactly
    what a severely limited shared upstream pool wants (bursts are what
    trigger 429s). ``on_wait(waited)`` fires (outside the lock) when the
    caller actually slept, so the harness can log throttle waits.
    """

    def __init__(self, rpm: int) -> None:
        self._rpm = max(1, int(rpm))
        self._spacing = 60.0 / self._rpm
        self._lock = threading.Lock()
        self._next_slot = 0.0  # monotonic timestamp of the next free slot

    @property
    def rpm(self) -> int:
        return self._rpm

    def acquire(
        self, on_wait: Callable[[float], None] | None = None
    ) -> float:
        """Block until this caller may send a request; returns seconds waited."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._spacing
        wait = slot - now
        if wait > 0:
            time.sleep(wait)
            if on_wait is not None:
                try:
                    on_wait(wait)
                except Exception:
                    pass
        return wait


_THROTTLES: dict[tuple[str, str], _ProviderThrottle] = {}
_THROTTLES_LOCK = threading.Lock()


def throttle_for(model_config: ModelConfig) -> _ProviderThrottle | None:
    """Process-wide throttle for the participant's (model, provider) selection.

    ``None`` when no ``provider_rate_limits`` entry applies (no throttling).
    The registry is keyed by (model_id, provider) and shared across all
    threads, so every worker sending to the same pinned provider divides one
    requests/minute budget. Harnesses are constructed per agent run; the
    registry (and therefore the budget) persists for the whole process.
    """
    rpm = model_config.rate_limit_rpm()
    if not rpm:
        return None
    provider = (getattr(model_config, "provider", "") or "").strip()
    key = (model_config.model_id, provider)
    with _THROTTLES_LOCK:
        throttle = _THROTTLES.get(key)
        if throttle is None or throttle.rpm != rpm:
            # A changed models.yaml value (e.g. between test runs) replaces
            # the stale throttle; concurrent callers share the new one.
            throttle = _ProviderThrottle(rpm)
            _THROTTLES[key] = throttle
        return throttle


def reset_throttles() -> None:
    """Drop the registry (test isolation only)."""
    with _THROTTLES_LOCK:
        _THROTTLES.clear()


# ── mini-swe-agent model proxy ─────────────────────────────────────────────


class RateLimitedModelProxy:
    """Wraps a mini-swe-agent model object; throttles before each ``query()``.

    mini-swe's ``DefaultAgent`` only needs ``query`` / ``format_message`` /
    ``format_observation_messages`` / ``get_template_vars`` / ``serialize``
    / ``config`` — everything is delegated via ``__getattr__``. Only one
    proxy wraps one model instance; the throttle itself is process-wide.
    """

    def __init__(
        self,
        inner: Any,
        throttle: _ProviderThrottle,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_throttle", throttle)
        object.__setattr__(self, "_on_wait", on_wait)

    def query(self, *args: Any, **kwargs: Any) -> Any:
        self._throttle.acquire(self._on_wait)
        return self._inner.query(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_inner", "_throttle", "_on_wait"):
            raise AttributeError(f"{name} is read-only on RateLimitedModelProxy")
        setattr(self._inner, name, value)


# ── OpenHands LLM subclass factory ─────────────────────────────────────────


def make_rate_limited_llm_class(
    llm_base: type,
    throttle: _ProviderThrottle,
    on_wait: Callable[[float], None] | None = None,
) -> type:
    """Build an ``llm_base`` subclass whose completion methods throttle first.

    OpenHands' ``Agent`` validates ``llm: LLM``, so a duck-typed proxy cannot
    be substituted — a subclass passes isinstance checks and keeps pydantic
    validation intact. Both ``completion`` and ``acompletion`` are wrapped
    (the SDK's sync and async step paths); retries inside them still space
    themselves via the SDK's own exponential backoff. The throttle is closed
    over (not a class attribute) so pydantic never sees it as a field/private
    attribute of the model.
    """
    if not isinstance(llm_base, type):
        # Test stubs pass a lambda/object as "LLM"; subclassing is impossible.
        # The harness falls back to the unwrapped LLM when this happens —
        # throttling is skipped, never a crash.
        raise TypeError(
            f"cannot subclass non-class LLM {llm_base!r}; rate limiting skipped"
        )

    class _RateLimitedLLM(llm_base):  # type: ignore[misc]
        def completion(self, *args: Any, **kwargs: Any) -> Any:
            throttle.acquire(on_wait)
            return super().completion(*args, **kwargs)

        async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
            throttle.acquire(on_wait)
            return await super().acompletion(*args, **kwargs)

    return _RateLimitedLLM


# ── rate-limit noise absorption ────────────────────────────────────────────

# Records whose formatted message matches any marker are treated as
# rate-limit chatter: mirrored to the retry log and dropped from the console
# for TUI-bound threads. Match on substrings that appear in both harnesses'
# retry outputs (litellm RateLimitError text and OpenRouter's own body).
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "ratelimiterror",
    "rate limit",
    "rate-limit",
    "provider returned error",
    "temporarily rate-limited",
    "too many requests",
)

_ABSORB = threading.local()
_ABSORB_DEFAULT = False
_NOISE_FILTER: "_RateLimitNoiseFilter | None" = None
_NOISE_INSTALL_LOCK = threading.Lock()

_MIRROR_LOGGER_NAME = "swe_duel.model_retries"


def set_thread_noise_absorbed(absorbed: bool) -> None:
    """Mark the CURRENT thread as running under a live TUI (or not).

    Called from ``AgentHarness.run`` with ``not console_echo``: threads whose
    console output is suppressed (the Live view owns the terminal) get
    rate-limit log records absorbed; plain-console threads keep them. The
    mode is also stored as the process-wide default so threads the harness
    does not directly control (SDK-internal threads, parallel callers) and
    threads created later inherit it — every agent run in one process shares
    the same console mode anyway.
    """
    global _ABSORB_DEFAULT
    _ABSORB_DEFAULT = bool(absorbed)
    _ABSORB.absorbed = bool(absorbed)


def _thread_absorbs() -> bool:
    return bool(getattr(_ABSORB, "absorbed", _ABSORB_DEFAULT))


def _is_rate_limit_noise(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


class _RateLimitNoiseFilter(logging.Filter):
    """Mirror rate-limit records to the retry log; drop them for TUI threads."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.name == _MIRROR_LOGGER_NAME:
                return True  # never filter our own mirror output
            if not _is_rate_limit_noise(record.getMessage()):
                return True
            _mirror_record(record)
            return not _thread_absorbs()
        except Exception:
            # A logging filter must never raise — failing open keeps logging.
            return True


def _mirror_logger() -> logging.Logger:
    """Dedicated file-backed logger for absorbed rate-limit chatter.

    Path: ``$SWE_DUEL_MODEL_RETRY_LOG`` (default ``data/logs/model_retries.log``,
    relative to the process CWD — every swe-duel CLI runs from the repo root).
    Created lazily on first absorbed record; if the file cannot be opened the
    logger carries a NullHandler and records are simply dropped.
    """
    logger = logging.getLogger(_MIRROR_LOGGER_NAME)
    if logger.handlers or getattr(logger, "_swe_duel_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    path = Path(
        os.environ.get("SWE_DUEL_MODEL_RETRY_LOG", "data/logs/model_retries.log")
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
    except Exception:
        handler = logging.NullHandler()
    logger.addHandler(handler)
    logger._swe_duel_configured = True  # type: ignore[attr-defined]
    return logger


def _mirror_record(record: logging.LogRecord) -> None:
    logger = _mirror_logger()
    if not logger.handlers or isinstance(logger.handlers[0], logging.NullHandler):
        return
    logger.log(
        record.levelno,
        "[absorbed from %s] %s",
        record.name,
        record.getMessage(),
    )


def install_rate_limit_log_filter() -> None:
    """Attach the noise filter to every path rate-limit chatter reaches stderr.

    Idempotent, and re-runnable: OpenHands adds a root StreamHandler at SDK
    import time (which happens inside harness ``run()``), so this is re-run
    per harness run and only ever appends the single filter instance.

    Coverage:

    * the ``litellm_model`` logger (mini-swe's tenacity ``before_sleep``
      warnings) — logger-level, so records are dropped before propagation;
    * every handler currently on the root logger (OpenHands's RichHandler —
      its ``openhands.*`` retry records reach root handlers, not root
      logger-level filters);
    * ``logging.lastResort`` — the implicit stderr handler used when no root
      handler exists (mini-swe-only runs).
    """
    global _NOISE_FILTER
    with _NOISE_INSTALL_LOCK:
        if _NOISE_FILTER is None:
            _NOISE_FILTER = _RateLimitNoiseFilter()
        filt = _NOISE_FILTER

        def _attach(target_has_filters: Any) -> None:
            if filt not in target_has_filters.filters:
                target_has_filters.addFilter(filt)

        _attach(logging.getLogger("litellm_model"))
        _attach(logging.getLogger("openhands"))
        for handler in logging.getLogger().handlers:
            _attach(handler)
        _attach(logging.lastResort)


# ── TUI startup helpers ────────────────────────────────────────────────────


def _harness_supports_rate_limit(harness_id: str) -> bool:
    # Imported lazily to avoid a circular import (base.py imports this module
    # for the capability table consumers; the table itself lives there).
    from swe_duel.agents.harness.base import HARNESS_RATE_LIMIT_SUPPORT

    return HARNESS_RATE_LIMIT_SUPPORT.get(harness_id, False)


def rate_limit_label(model_config: ModelConfig) -> str:
    """Human label for the participant's effective request-rate cap."""
    provider = (getattr(model_config, "provider", "") or "").strip()
    rpm = model_config.rate_limit_rpm()
    if not rpm:
        return ""
    slug = provider or "auto-route"
    return f"{rpm} rpm via {slug}"


def warn_unenforceable_rate_limits(
    participant_pairs: Iterable[tuple[ModelConfig, str]],
    emit: Callable[[str], None] = print,
) -> None:
    """Warn (once per combo, at TUI startup — safe, pre-Live) when a
    configured rate limit cannot be honoured by the harness that will run it.

    codex / claude-code CLIs run in-container and make their own LLM calls,
    so the host-side throttle is bypassed there; their built-in retry loops
    still absorb transient 429s, but a configured limit silently does
    nothing. Takes (model_config, harness_id) pairs — exactly the selected
    participants — so only real combos are reported.
    """
    seen: set[tuple[str, str, str]] = set()
    for cfg, harness_id in participant_pairs:
        provider = (getattr(cfg, "provider", "") or "").strip()
        if not cfg.rate_limit_rpm():
            continue
        if _harness_supports_rate_limit(harness_id):
            continue
        key = (cfg.model_id, provider, harness_id)
        if key in seen:
            continue
        seen.add(key)
        emit(
            f"[warn] {cfg.model_id} (provider: {provider or 'auto-route'}) has a "
            f"rate limit configured, but harness {harness_id!r} runs its CLI "
            f"in-container and cannot enforce it — expect provider 429s to be "
            f"handled by the CLI's own retries only"
        )


def summarize_rate_limits(
    model_configs: Iterable[ModelConfig],
    emit: Callable[[str], None] = print,
) -> None:
    """Print each participant's effective rate cap before the live TUI starts.

    The throttle is process-wide and shared across parallel workers, so the
    printed rate is the TOTAL budget all workers divide; surface it up front
    so the operator can pick --max-workers accordingly (e.g. 30 rpm with 5
    workers ≈ 10s average queue wait per agent step).
    """
    seen: set[tuple[str, str]] = set()
    for cfg in model_configs:
        provider = (getattr(cfg, "provider", "") or "").strip()
        key = (cfg.model_id, provider)
        if key in seen:
            continue
        seen.add(key)
        rpm = cfg.rate_limit_rpm()
        if not rpm:
            continue
        emit(
            f"[rate-limit] {cfg.model_id} (provider: {provider or 'auto-route'}) "
            f"throttled to {rpm} requests/min — shared by ALL workers in this "
            f"process; reduce --max-workers if wall-clock budgets get tight"
        )
