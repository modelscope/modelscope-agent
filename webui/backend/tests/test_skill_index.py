import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from app.backends.ms_agent import skill_index as skill_index_module
from app.backends.ms_agent.skill_change_tracker import (
    ChangeSnapshot,
    SkillChangeTracker,
)
from app.backends.ms_agent.skill_index import SkillIndex


def _make_skill(root: Path, skill_id: str):
    skill = root / skill_id
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: test skill\n---\nbody\n",
        encoding="utf-8",
    )
    return skill


def _wait_for_scope_generation(
    tracker, scope: str, previous: int, timeout: float = 3.0
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = tracker.snapshot(scope)
        if snapshot.generation > previous:
            return snapshot
        time.sleep(0.02)
    raise AssertionError("skill watcher did not publish the change")


class _StaticTracker:
    """Trusted tracker that never emits an event.

    This isolates the synchronous source/SKILL.md marker safety net from the
    normal watchfiles path.
    """

    def __init__(self):
        self._roots: dict[str, tuple[str, ...]] = {}

    def register(self, scope, roots):
        normalised = tuple(sorted(str(Path(root).resolve()) for root in roots))
        self._roots[scope] = normalised
        return ChangeSnapshot(0, True, False, normalised, 0, 0)

    def snapshot(self, scope):
        return ChangeSnapshot(0, True, False, self._roots.get(scope, ()), 0, 0)

    def mark_dirty(self, _scope):
        return 0

    def stop_all(self):
        return None


def test_scope_index_uses_metadata_only_and_sees_direct_new_skill_immediately(
    tmp_path, monkeypatch
):
    home_dir = tmp_path / "home"
    tree = home_dir / "skills"
    _make_skill(tree, "alpha")
    monkeypatch.setattr(skill_index_module, "home", lambda: str(home_dir))
    tracker = SkillChangeTracker()
    index = SkillIndex(tracker)
    try:

        def _unexpected_rglob(*args, **kwargs):
            raise AssertionError("scope index must not walk skill support files")

        monkeypatch.setattr(Path, "rglob", _unexpected_rglob)
        first = index.snapshot("global")
        assert [row[0] for row in first.rows] == ["alpha"]
        assert first.find("alpha") is not None

        _make_skill(tree, "beta")
        second = index.snapshot("global")
        assert [row[0] for row in second.rows] == ["alpha", "beta"]
        assert second.token != first.token
    finally:
        index.stop()


def test_skill_marker_sees_edit_and_delete_without_watcher_event(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    skill = _make_skill(home_dir / "skills", "alpha")
    monkeypatch.setattr(skill_index_module, "home", lambda: str(home_dir))
    index = SkillIndex(_StaticTracker())
    try:
        first = index.snapshot("global")
        assert first.find("alpha")[0].description == "test skill"

        (skill / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: changed immediately\n---\nbody\n",
            encoding="utf-8",
        )
        second = index.snapshot("global")
        assert second.find("alpha")[0].description == "changed immediately"
        assert second.token != first.token

        (skill / "SKILL.md").unlink()
        third = index.snapshot("global")
        assert third.find("alpha") is None
        assert third.token != second.token
    finally:
        index.stop()


def test_unchanged_scope_has_one_builder_for_concurrent_reads(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    _make_skill(home_dir / "skills", "alpha")
    monkeypatch.setattr(skill_index_module, "home", lambda: str(home_dir))

    from ms_agent.skill.loader import SkillLoader

    original = SkillLoader.discover_skills
    call_count = 0
    count_lock = threading.Lock()

    def _counted(self, paths):
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.02)
        return original(self, paths)

    monkeypatch.setattr(SkillLoader, "discover_skills", _counted)
    tracker = SkillChangeTracker()
    index = SkillIndex(tracker)
    try:
        with ThreadPoolExecutor(max_workers=20) as pool:
            snapshots = list(pool.map(lambda _: index.snapshot("global"), range(20)))

        assert call_count == 1
        assert len({id(snapshot) for snapshot in snapshots}) == 1
        assert [row[0] for row in snapshots[0].rows] == ["alpha"]
    finally:
        index.stop()


def test_runtime_tracks_local_sources_outside_managed_scope(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    source = _make_skill(tmp_path / "yaml-skills", "alpha")
    project_dir.mkdir()
    monkeypatch.setattr(skill_index_module, "home", lambda: str(home_dir))

    from ms_agent.skill import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module, "BUILTIN_SKILLS_DIR", tmp_path / "missing-builtins"
    )
    monkeypatch.setattr(
        catalog_module, "USER_SKILLS_DIR", tmp_path / "missing-user-skills"
    )
    config = SimpleNamespace(
        sources=[
            SimpleNamespace(
                type="local",
                path=str(source.parent),
                enabled=True,
            )
        ],
        auto_discover=False,
    )
    tracker = SkillChangeTracker()
    index = SkillIndex(tracker)
    try:
        first = index.runtime_snapshot("p1", str(project_dir), config)
        assert first.trusted

        generation = tracker.snapshot("runtime:p1").generation
        (source / "reference.md").write_text("changed", encoding="utf-8")
        _wait_for_scope_generation(tracker, "runtime:p1", generation)
        second = index.runtime_snapshot("p1", str(project_dir), config)

        assert second.trusted
        assert second.token != first.token
    finally:
        index.stop()


def test_remote_runtime_source_keeps_legacy_full_sync_mode(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(skill_index_module, "home", lambda: str(home_dir))
    config = SimpleNamespace(
        sources=[
            SimpleNamespace(
                type="modelscope",
                repo_id="owner/skill",
                enabled=True,
            )
        ],
        auto_discover=False,
    )
    tracker = SkillChangeTracker()
    index = SkillIndex(tracker)
    try:
        snapshot = index.runtime_snapshot("p1", str(project_dir), config)
        assert not snapshot.trusted
    finally:
        index.stop()


def test_resource_generation_changes_runtime_token(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    skill = _make_skill(home_dir / "skills", "alpha")
    monkeypatch.setattr(skill_index_module, "home", lambda: str(home_dir))
    tracker = SkillChangeTracker()
    index = SkillIndex(tracker)
    try:
        first = index.snapshot("global")
        (skill / "reference.md").write_text("new resource", encoding="utf-8")
        tracker.mark_dirty("global")
        second = index.snapshot("global")

        assert second.token != first.token
        assert second.generation > first.generation
    finally:
        index.stop()
