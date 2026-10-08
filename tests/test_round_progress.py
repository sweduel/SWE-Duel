"""Tests for the parallel-round Blue defense TUI reporter.

These exercise the thread-safe data model + rich rendering of
`swe_duel.engine.round_progress` without running any agent — the reporter is pure
state, so we can drive it directly (including concurrently) and assert on the
rendered tree text.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rich.console import Console

from swe_duel.engine.round_progress import DefenseRoundReporter, Status, live


def _render_text(reporter: DefenseRoundReporter) -> str:
    console = Console(width=200, file=open("/dev/null", "w"))
    with console.capture() as cap:
        console.print(reporter)
    return cap.get()


def test_register_and_finish_subturn(record):
    r = DefenseRoundReporter(round_index=1)
    r.register_match("m1", "A vs B")
    h = r.register_subturn(
        "m1", "flask:b_defends_a:0:abcd1234",
        label="blue=B defends A's flask challenge",
        blue_model="vendor/b", repo="flask",
    )
    h.start()
    h.step(5, 50)
    h.phase("regression", "running")
    txt_running = _render_text(r)
    record("running_render", txt_running)
    assert "Round 1" in txt_running
    assert "A vs B" in txt_running
    assert "5/50" in txt_running

    h.phase("regression", "success")
    h.finish(Status.SUCCESS, detail="blue=1.00 $0.0100")
    txt_done = _render_text(r)
    record("done_render", txt_done)
    assert "blue=1.00" in txt_done
    # finished sub-turn collapses (no step bar in its row anymore)
    assert "✓" in txt_done


def test_cached_subturn_marked(record):
    r = DefenseRoundReporter()
    r.register_match("m1", "A vs B")
    r.register_subturn(
        "m1", "s1", label="cached one", blue_model="b", repo="flask", cached=True
    )
    txt = _render_text(r)
    record("render", txt)
    assert "CACHED" in txt


def test_done_counter_in_root(record):
    r = DefenseRoundReporter(round_index=2)
    r.register_match("m1", "A vs B")
    h1 = r.register_subturn("m1", "s1", label="one", blue_model="b", repo="flask")
    h2 = r.register_subturn("m1", "s2", label="two", blue_model="b", repo="jwt")
    h1.finish(Status.SUCCESS, detail="ok")
    txt = _render_text(r)
    record("render", txt)
    # 1 of 2 done
    assert "[1/2 sub-turns]" in txt
    h2.finish(Status.FAIL, detail="x")
    assert "[2/2 sub-turns]" in _render_text(r)


def test_concurrent_updates_are_threadsafe(record):
    # Many threads updating distinct sub-turns must not corrupt state.
    r = DefenseRoundReporter()
    r.register_match("m1", "A vs B")
    handles = [
        r.register_subturn("m1", f"s{i}", label=f"t{i}", blue_model="b", repo="r")
        for i in range(40)
    ]

    def drive(h):
        h.start()
        for s in range(1, 11):
            h.step(s, 10)
            h.phase("regression", "running")
        h.phase("regression", "success")
        h.finish(Status.SUCCESS, detail="done")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(drive, handles))

    txt = _render_text(r)
    record("final_render", txt)
    assert "[40/40 sub-turns]" in txt


def test_live_context_manager_runs(record):
    # Smoke test: the live() context manager should start/stop cleanly with a
    # non-tty console (no exceptions).
    r = DefenseRoundReporter()
    r.register_match("m1", "A vs B")
    h = r.register_subturn("m1", "s1", label="t", blue_model="b", repo="r")
    console = Console(file=open("/dev/null", "w"), force_terminal=False)
    with live(r, console=console, refresh_per_second=4):
        h.start()
        h.step(1, 5)
        h.finish(Status.SUCCESS, detail="ok")
    record("ok", True)
    assert True
