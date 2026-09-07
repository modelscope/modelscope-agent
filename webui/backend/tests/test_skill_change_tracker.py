import time

from app.backends.ms_agent.skill_change_tracker import SkillChangeTracker


def _wait_for_generation(tracker, scope: str, previous: int, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = tracker.snapshot(scope)
        if snapshot.generation > previous:
            return snapshot
        time.sleep(0.02)
    raise AssertionError("skill watcher did not publish the filesystem change")


def test_visible_file_change_advances_scope_generation(tmp_path):
    tracker = SkillChangeTracker()
    try:
        initial = tracker.register("global", [tmp_path])
        assert initial.ready and not initial.fail_safe

        (tmp_path / "reference.md").write_text("changed", encoding="utf-8")
        changed = _wait_for_generation(tracker, "global", initial.generation)

        assert changed.events >= 1
        assert changed.batches >= 1
    finally:
        tracker.stop_all()


def test_many_events_are_coalesced_into_bounded_dirty_state(tmp_path):
    tracker = SkillChangeTracker()
    try:
        initial = tracker.register("project:p", [tmp_path])
        for index in range(30):
            (tmp_path / f"file-{index}.txt").write_text(str(index), encoding="utf-8")

        changed = _wait_for_generation(tracker, "project:p", initial.generation)
        time.sleep(0.2)
        settled = tracker.snapshot("project:p")

        assert settled.events >= 1
        assert settled.batches <= 3
        assert settled.generation - initial.generation <= 3
        assert changed.generation <= settled.generation
    finally:
        tracker.stop_all()


def test_watcher_failure_enters_explicit_fail_safe(tmp_path):
    calls = 0

    def _broken_watch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if False:
            yield set()
        raise RuntimeError("watch unavailable")

    tracker = SkillChangeTracker(watch_fn=_broken_watch)
    try:
        snapshot = tracker.register("global", [tmp_path])
        assert snapshot.ready
        assert snapshot.fail_safe
        assert "watch unavailable" in (snapshot.error or "")
        repeated = tracker.register("global", [tmp_path])
        assert repeated.fail_safe
        assert calls == 1
    finally:
        tracker.stop_all()


def test_source_root_reconfiguration_restarts_scope(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    tracker = SkillChangeTracker()
    try:
        first = tracker.register("global", [first_root])
        second = tracker.register("global", [second_root])

        assert second.ready and not second.fail_safe
        assert second.generation > first.generation
        assert second.roots == (str(second_root.resolve()),)
    finally:
        tracker.stop_all()


def test_webui_mutation_marks_scope_dirty_without_a_watcher():
    tracker = SkillChangeTracker()
    try:
        first = tracker.snapshot("global")
        generation = tracker.mark_dirty("global")
        second = tracker.snapshot("global")

        assert generation == first.generation + 1
        assert second.generation == generation
        assert second.roots == ()
    finally:
        tracker.stop_all()
