"""Rate limiting + 429-noise absorption (swe_duel.agents.harness.rate_limit).

Covers the two halves of the provider-429 problem:

- the process-wide throttle shared by all worker threads (workers divide one
  requests/minute budget per (model, provider)), and
- the log filter that mirrors rate-limit chatter to data/logs/model_retries.log
  and keeps it off the console for threads running under a live TUI.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from swe_duel.agents.harness import rate_limit as rl
from swe_duel.config import ModelConfig


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        model_id="qwen/qwen3.8-flash",
        temperature=0.2,
        max_tokens=2048,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    """Fresh throttles + a temp absorb log per test; no cross-test leakage."""
    rl.reset_throttles()
    monkeypatch.setenv("SWE_DUEL_MODEL_RETRY_LOG", str(tmp_path / "retries.log"))
    # The mirror logger caches its FileHandler on first use — reset it so each
    # test's records land in ITS OWN temp log.
    mirror = logging.getLogger("swe_duel.model_retries")
    for h in list(mirror.handlers):
        mirror.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    if hasattr(mirror, "_swe_duel_configured"):
        del mirror._swe_duel_configured
    yield
    rl.reset_throttles()


# ── config: provider_rate_limits parsing / resolution ──────────────────────


class TestRateLimitConfig:
    def test_rate_limits_parse_from_field(self, record):
        cfg = _cfg(
            providers=["alibaba"],
            provider_rate_limits={"alibaba": 40, "default": 20},
        )
        record("provider_rate_limits", cfg.provider_rate_limits)
        assert cfg.provider_rate_limits == {"alibaba": 40, "default": 20}

    def test_rate_limit_rejects_bad_values(self, record):
        for bad in ({"alibaba": 0}, {"alibaba": -5}, {"alibaba": 2.5},
                    {"": 10}, {"bad slug": 10}, {"a#b": 10}):
            with pytest.raises(ValidationError):
                _cfg(provider_rate_limits=bad)
            record("rejected", bad)

    def test_rpm_resolution_provider_then_default(self, record):
        cfg = _cfg(
            provider_rate_limits={"alibaba": 40, "default": 20},
        )
        pinned = cfg.with_selection(provider="alibaba")
        other = cfg.with_selection(provider="fireworks")
        auto = cfg.with_selection(provider="")
        assert pinned.rate_limit_rpm() == 40
        assert other.rate_limit_rpm() == 20  # falls back to "default"
        assert auto.rate_limit_rpm() == 20  # auto-route consults "default"
        assert _cfg().rate_limit_rpm() is None  # unconfigured = no throttle
        # Explicit provider argument overrides the bound selection.
        assert cfg.rate_limit_rpm("alibaba") == 40
        record("resolution", {
            "pinned": pinned.rate_limit_rpm(),
            "other": other.rate_limit_rpm(),
            "auto": auto.rate_limit_rpm(),
        })

    def test_rate_limits_load_from_models_yaml(self, config_dir: Path, record):
        # The committed config pins the two models with probed caps; guard
        # the knob against silent removal from models.yaml.
        from swe_duel.config import load_models_config

        cfgs = load_models_config(config_dir)
        qwen = cfgs["qwen3.8-flash"]
        assert qwen.provider_rate_limits["alibaba"] > 0
        assert qwen.with_selection(provider="alibaba").rate_limit_rpm() > 0
        record("qwen_limits", qwen.provider_rate_limits)


# ── throttle mechanics ─────────────────────────────────────────────────────


class TestProviderThrottle:
    def test_no_throttle_when_unconfigured(self):
        assert rl.throttle_for(_cfg()) is None

    def test_spacing_between_acquire_slots(self):
        throttle = rl._ProviderThrottle(rpm=600)  # 0.1s spacing
        t0 = time.monotonic()
        waited = [throttle.acquire() for _ in range(3)]
        elapsed = time.monotonic() - t0
        assert waited[0] == 0.0
        # 3 back-to-back acquires must space ≥ 2 intervals total.
        assert elapsed + waited[2] >= 0.2 - 0.02
        assert waited[1] > 0

    def test_registry_shares_one_throttle_across_workers(self):
        cfg = _cfg(provider="alibaba", provider_rate_limits={"alibaba": 600})
        throttles = [rl.throttle_for(cfg) for _ in range(5)]
        assert len({id(t) for t in throttles}) == 1  # one shared instance

    def test_parallel_workers_divide_one_budget(self):
        """N threads hammering acquire() together stay under the rpm cap."""
        rpm = 300  # 0.2s spacing
        cfg = _cfg(provider="alibaba", provider_rate_limits={"alibaba": rpm})
        throttle = rl.throttle_for(cfg)
        n_threads, n_calls = 4, 8
        starts: list[float] = []
        lock = threading.Lock()

        def _worker():
            for _ in range(n_calls):
                throttle.acquire()
                with lock:
                    starts.append(time.monotonic())

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0

        starts.sort()
        # All request starts must respect the min spacing (± scheduler slop).
        for a, b in zip(starts, starts[1:]):
            assert b - a >= 0.2 - 0.03
        # And the whole batch takes ≈ spacing × (calls - 1), i.e. the budget
        # was divided, not multiplied per worker.
        assert elapsed >= 0.2 * (n_threads * n_calls - 1) - 0.05
        assert elapsed <= 0.2 * (n_threads * n_calls) + 1.0

    def test_on_wait_callback_fires(self):
        throttle = rl._ProviderThrottle(rpm=600)
        waited_vals: list[float] = []

        def _on_wait(w: float) -> None:
            waited_vals.append(w)

        throttle.acquire(_on_wait)  # first slot: no wait, no callback
        throttle.acquire(_on_wait)  # second: waits ~0.1s
        assert len(waited_vals) == 1
        assert waited_vals[0] >= 0.05

    def test_changed_rpm_replaces_throttle(self):
        a = rl.throttle_for(_cfg(provider="alibaba", provider_rate_limits={"alibaba": 40}))
        b = rl.throttle_for(_cfg(provider="alibaba", provider_rate_limits={"alibaba": 12}))
        assert a is not b
        assert b.rpm == 12
        # The registry entry is the new throttle for subsequent callers.
        assert rl.throttle_for(_cfg(provider="alibaba", provider_rate_limits={"alibaba": 12})) is b


# ── mini-swe model proxy ───────────────────────────────────────────────────


class _FakeInnerModel:
    def __init__(self) -> None:
        self.config = {"model_name": "openrouter/qwen/qwen3.8-flash"}
        self.queries = 0

    def query(self, messages):  # noqa: ANN001
        self.queries += 1
        return {"role": "assistant", "content": "ok"}

    def format_message(self, *, role: str, content: str) -> dict:
        return {"role": role, "content": content}


class TestRateLimitedModelProxy:
    def test_proxy_throttles_query_and_delegates_everything(self):
        inner = _FakeInnerModel()
        throttle = rl._ProviderThrottle(rpm=600)
        waits: list[float] = []
        proxy = rl.RateLimitedModelProxy(inner, throttle, on_wait=waits.append)

        assert proxy.config == inner.config  # __getattr__ delegation
        assert proxy.format_message(role="user", content="hi") == {
            "role": "user", "content": "hi",
        }
        assert proxy.query([{"role": "user", "content": "x"}])["content"] == "ok"
        assert inner.queries == 1
        # A second immediate query must be spaced → on_wait fired.
        proxy.query([{"role": "user", "content": "y"}])
        assert len(waits) == 1 and waits[0] > 0

    def test_proxy_setattr_writes_through(self):
        inner = _FakeInnerModel()
        proxy = rl.RateLimitedModelProxy(inner, rl._ProviderThrottle(rpm=600))
        proxy.new_flag = 7
        assert inner.new_flag == 7  # set on the inner model, not the proxy
        with pytest.raises(AttributeError):
            proxy._throttle = None  # guard attrs are read-only


# ── OpenHands LLM subclass factory ─────────────────────────────────────────


class TestRateLimitedLLMClass:
    def test_subclass_wraps_completion_and_acompletion(self):
        calls: list[str] = []

        class _FakeLLM:
            def completion(self, *args, **kwargs):  # noqa: ANN002, ANN003
                calls.append("completion")
                return {"ok": True}

            async def acompletion(self, *args, **kwargs):  # noqa: ANN002, ANN003
                calls.append("acompletion")
                return {"ok": True}

        throttle = rl._ProviderThrottle(rpm=10_000)  # effectively free
        cls = rl.make_rate_limited_llm_class(_FakeLLM, throttle)
        llm = cls()
        assert isinstance(llm, _FakeLLM)
        assert llm.completion(messages=[]) == {"ok": True}

        import asyncio

        assert asyncio.run(llm.acompletion(messages=[])) == {"ok": True}
        assert calls == ["completion", "acompletion"]

    def test_subclass_acquires_throttle_before_call(self):
        acquired: list[int] = []

        class _FakeLLM:
            def completion(self, *args, **kwargs):  # noqa: ANN002, ANN003
                acquired.append(1)
                return {}

        rpm = 600  # 0.1s spacing
        throttle = rl._ProviderThrottle(rpm=rpm)
        cls = rl.make_rate_limited_llm_class(_FakeLLM, throttle)
        llm = cls()
        t0 = time.monotonic()
        llm.completion()
        llm.completion()
        # Two calls spaced by the throttle.
        assert time.monotonic() - t0 >= 0.1 - 0.02

    def test_non_class_llm_base_raises_typeerror(self):
        throttle = rl._ProviderThrottle(rpm=600)
        with pytest.raises(TypeError):
            rl.make_rate_limited_llm_class(lambda **k: object(), throttle)


# ── rate-limit noise absorption ────────────────────────────────────────────


def _make_record(name: str, message: str, level: int = logging.WARNING) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )


class TestNoiseAbsorption:
    @pytest.fixture(autouse=True)
    def _fresh_logging(self):
        # Detach our filter from anything a previous test attached it to so
        # each test starts from the pre-install state.
        rl._NOISE_FILTER = None
        saved_root_handlers = list(logging.getLogger().handlers)
        for h in saved_root_handlers:
            h.filters = [f for f in h.filters if not isinstance(f, rl._RateLimitNoiseFilter)]
        yield
        for h in saved_root_handlers:
            h.filters = [f for f in h.filters if not isinstance(f, rl._RateLimitNoiseFilter)]
        logging.getLogger("litellm_model").filters = [
            f for f in logging.getLogger("litellm_model").filters
            if not isinstance(f, rl._RateLimitNoiseFilter)
        ]

    def test_marker_matching(self):
        assert rl._is_rate_limit_noise(
            "Retrying <unknown> in 4 seconds as it raised RateLimitError: "
            'litellm.RateLimitError: OpenrouterException - {"error":{"message":'
            '"Provider returned error","code":429}}'
        )
        assert rl._is_rate_limit_noise(
            "litellm.RateLimitError: RateLimitError: ... Attempt #1 | "
            "You can customize retry values in the configuration."
        )
        assert rl._is_rate_limit_noise("qwen/qwen3.8-flash is temporarily rate-limited upstream")
        assert not rl._is_rate_limit_noise("some unrelated authentication warning")

    def test_filter_drops_records_only_for_absorbed_threads(self, tmp_path):
        filt = rl._RateLimitNoiseFilter()
        record = _make_record(
            "litellm_model",
            "Retrying in 4 seconds as it raised RateLimitError: "
            "Provider returned error 429",
        )
        clean = _make_record("litellm_model", "container started")

        rl.set_thread_noise_absorbed(False)
        assert filt.filter(record) is True  # console mode: keep everything
        assert filt.filter(clean) is True

        rl.set_thread_noise_absorbed(True)
        assert filt.filter(record) is False  # TUI mode: absorbed
        assert filt.filter(clean) is True
        rl.set_thread_noise_absorbed(False)

    def test_absorb_mode_is_inherited_by_new_threads(self):
        """The harness sets the mode on ITS thread; SDK-internal or parallel
        threads created later must inherit it (all agent runs in one process
        share the same console mode)."""
        filt = rl._RateLimitNoiseFilter()
        record = _make_record(
            "litellm_model", "RateLimitError: Provider returned error 429"
        )
        seen: list[bool] = []

        rl.set_thread_noise_absorbed(True)
        try:
            t = threading.Thread(
                target=lambda: seen.append(filt.filter(record))
            )
            t.start()
            t.join()
            assert seen == [False]  # child thread inherited TUI absorption
        finally:
            rl.set_thread_noise_absorbed(False)
        t = threading.Thread(target=lambda: seen.append(filt.filter(record)))
        t.start()
        t.join()
        assert seen[-1] is True  # console mode inherited: record kept

    def test_absorbed_records_are_mirrored_to_log_file(self, tmp_path):
        rl.set_thread_noise_absorbed(True)
        try:
            filt = rl._RateLimitNoiseFilter()
            record = _make_record(
                "litellm_model",
                "Retrying in 4 seconds as it raised RateLimitError: "
                '{"error":{"message":"Provider returned error","code":429}}',
            )
            assert filt.filter(record) is False
            log_path = tmp_path / "retries.log"
            assert log_path.exists()
            text = log_path.read_text()
            assert "Provider returned error" in text
            assert "absorbed from litellm_model" in text
        finally:
            rl.set_thread_noise_absorbed(False)

    def test_install_is_idempotent_and_covers_root_and_lastresort(self):
        rl.install_rate_limit_log_filter()
        rl.install_rate_limit_log_filter()
        rl.install_rate_limit_log_filter()
        assert isinstance(rl._NOISE_FILTER, rl._RateLimitNoiseFilter)
        filt = rl._NOISE_FILTER
        # litellm_model logger-level (mini-swe tenacity path)…
        assert filt in logging.getLogger("litellm_model").filters
        # …root handlers (OpenHands adds a RichHandler to the root logger)…
        if logging.getLogger().handlers:
            assert all(filt in h.filters for h in logging.getLogger().handlers)
        # …and the implicit stderr fallback handler.
        assert filt in logging.lastResort.filters
        # Exactly one filter instance everywhere (never stacked).
        assert (
            sum(isinstance(f, rl._RateLimitNoiseFilter)
                for f in logging.getLogger("litellm_model").filters)
            == 1
        )

    def test_end_to_end_tenacity_style_warning_stays_off_stderr(
        self, tmp_path, capsys
    ):
        """The real spill path: records from a logger with neither handlers
        nor our logger-level filter (e.g. OpenHands' retry-mixin logger)
        reach stderr via logging.lastResort when no root handler exists.
        With the filter installed and the thread marked TUI-bound, the 429
        body must not reach that stream — but must land in the mirror log."""
        saved_root_handlers = list(logging.getLogger().handlers)
        for h in saved_root_handlers:
            logging.getLogger().removeHandler(h)
        saved_last_resort = logging.lastResort
        stream = __import__("io").StringIO()
        logging.lastResort = logging.StreamHandler(stream)
        try:
            rl.install_rate_limit_log_filter()
            rl.set_thread_noise_absorbed(True)
            logger = logging.getLogger("openhands.sdk.llm.utils.retry_mixin")
            logger.setLevel(logging.WARNING)
            logger.warning(
                "Retrying in 4 seconds as it raised RateLimitError: "
                "OpenrouterException - "
                '{"error":{"message":"Provider returned error","code":429}}'
            )
            assert "Provider returned error" not in stream.getvalue()
            assert "Provider returned error" in (tmp_path / "retries.log").read_text()
        finally:
            rl.set_thread_noise_absorbed(False)
            logging.lastResort = saved_last_resort
            for h in saved_root_handlers:
                logging.getLogger().addHandler(h)


# ── TUI startup helpers ────────────────────────────────────────────────────


class TestStartupHelpers:
    def test_summarize_rate_limits_lists_only_capped_participants(self):
        lines: list[str] = []
        capped = _cfg(provider="alibaba", provider_rate_limits={"alibaba": 40})
        uncapped = _cfg(model_id="z-ai/glm-5.3-flash")
        rl.summarize_rate_limits([capped, uncapped], emit=lines.append)
        assert len(lines) == 1
        assert "qwen/qwen3.8-flash" in lines[0]
        assert "40 requests/min" in lines[0]

    def test_warn_unenforceable_only_for_unsupported_harnesses(self):
        lines: list[str] = []
        capped = _cfg(provider="alibaba", provider_rate_limits={"alibaba": 40})
        uncapped = _cfg(model_id="z-ai/glm-5.3-flash")
        pairs = [
            (capped, "mini-swe-agent"),   # supported → no warning
            (capped, "codex"),            # unsupported → warn
            (capped, "codex"),            # duplicate → once
            (uncapped, "codex"),          # no limit configured → no warning
        ]
        rl.warn_unenforceable_rate_limits(pairs, emit=lines.append)
        assert len(lines) == 1
        assert "codex" in lines[0]

    def test_harness_capability_table(self):
        from swe_duel.agents.harness.base import HARNESS_RATE_LIMIT_SUPPORT

        assert HARNESS_RATE_LIMIT_SUPPORT["mini-swe-agent"] is True
        assert HARNESS_RATE_LIMIT_SUPPORT["openhands"] is True
        assert HARNESS_RATE_LIMIT_SUPPORT["codex"] is False
        assert HARNESS_RATE_LIMIT_SUPPORT["claude-code"] is False
