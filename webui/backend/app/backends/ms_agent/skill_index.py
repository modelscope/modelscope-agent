"""Lightweight per-scope skill discovery and runtime change tokens."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from app.backends.ms_agent.common import home
from app.backends.ms_agent.skill_change_tracker import (
    SkillChangeTracker,
    skill_change_tracker,
)


@dataclass(frozen=True)
class ScopeSkillSnapshot:
    scope: str
    rows: tuple[tuple[str, object, bool], ...]
    token: str
    state_token: str
    generation: int
    trusted: bool

    def find(self, skill_id: str):
        # Catalog registration is last-source-wins.  Resolve ids in the same
        # direction so management reads, deletes and overwrites target the Skill
        # the agent actually uses when two roots provide the same id.
        for runtime_id, skill, enabled in reversed(self.rows):
            if runtime_id == skill_id:
                return skill, enabled
        return None


@dataclass(frozen=True)
class RuntimeSkillSnapshot:
    token: str
    trusted: bool


class SkillIndex:
    """Discover SKILL.md metadata while caching unchanged descriptors.

    Each source keeps its own SDK ``SkillLoader``.  An unchanged tracker and
    config token returns the immutable scope snapshot directly; a changed
    token performs only bounded marker discovery and SKILL.md reads, never a
    support-tree walk.  The native tracker generation covers arbitrary
    visible support-file changes between calls.
    """

    def __init__(self, tracker: SkillChangeTracker = skill_change_tracker):
        self._tracker = tracker
        self._lock = threading.RLock()
        self._loaders: dict[tuple[str, str], object] = {}
        self._snapshots: dict[str, ScopeSkillSnapshot] = {}

    @staticmethod
    def _path_marker(path: Path):
        try:
            path_stat = path.stat()
            return (path_stat.st_mtime_ns, path_stat.st_ctime_ns, path_stat.st_size)
        except OSError:
            return None

    @classmethod
    def _skill_markers(cls, rows: tuple | list) -> tuple:
        markers = []
        for _runtime_id, skill, _enabled in rows:
            root = Path(str(getattr(skill, "skill_path", "") or ""))
            marker = root if root.name == "SKILL.md" else root / "SKILL.md"
            markers.append((str(marker), cls._path_marker(marker)))
        return tuple(markers)

    @staticmethod
    def _scope_inputs(scope: str, project_path: str | None):
        from ms_agent.config.skills_manager import SkillsConfigManager
        from ms_agent.skill.sources import SkillSourceType, parse_skill_source

        manager = SkillsConfigManager(global_dir=home())
        if scope == "global":
            sources = manager.list_sources(scope="global")
            disabled = set(manager.load_global().get("disabled", []))
        else:
            sources = manager.list_sources(scope="project", project_path=project_path)
            disabled = set(manager.load_merged(project_path).get("disabled", []))

        local_sources: list[str] = []
        roots: list[Path] = []
        has_untracked_source = False
        raw_sources = tuple(str(source) for source in sources)
        for source_text in raw_sources:
            try:
                source = parse_skill_source(source_text)
            except Exception:
                has_untracked_source = True
                continue
            if source.type != SkillSourceType.LOCAL_DIR or not source.path:
                has_untracked_source = True
                continue
            source_path = Path(source.path).expanduser()
            if source_path.is_file():
                has_untracked_source = True
                continue
            local_sources.append(str(source_path))
            if source_path.is_dir():
                roots.append(source_path)
        return (
            raw_sources,
            tuple(sorted(disabled)),
            tuple(local_sources),
            tuple(roots),
            has_untracked_source,
        )

    def snapshot(
        self, scope: str, project_path: str | None = None
    ) -> ScopeSkillSnapshot:
        from ms_agent.skill.loader import SkillLoader

        with self._lock:
            (
                raw_sources,
                disabled_values,
                local_sources,
                roots,
                has_untracked_source,
            ) = self._scope_inputs(scope, project_path)
            change = self._tracker.register(scope, roots)
            cached = self._snapshots.get(scope)
            state_payload = {
                "sources": raw_sources,
                "disabled": disabled_values,
                "roots": change.roots,
                # Directory metadata catches a direct live-tree add/delete
                # synchronously, before the debounced OS event is delivered.
                # This is one stat per source, never a directory listing.
                "root_markers": tuple(
                    (str(root.resolve()), self._path_marker(root)) for root in roots
                ),
                # Known SKILL.md metadata preserves immediate metadata edits
                # while keeping cost proportional to Skill roots, not files.
                "skill_markers": self._skill_markers(cached.rows if cached else ()),
                "generation": change.generation,
                "ready": change.ready,
                "fail_safe": change.fail_safe,
                "has_untracked_source": has_untracked_source,
            }
            state_token = hashlib.sha256(
                json.dumps(
                    state_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if cached is not None and cached.state_token == state_token:
                return cached

            disabled = set(disabled_values)
            rows: list[tuple[str, object, bool]] = []
            active_loader_keys = {(scope, source) for source in local_sources}
            for key in list(self._loaders):
                if key[0] == scope and key not in active_loader_keys:
                    self._loaders.pop(key, None)

            token_rows: list[dict] = []
            for source_text in local_sources:
                key = (scope, source_text)
                loader = self._loaders.get(key)
                if loader is None:
                    loader = SkillLoader()
                    self._loaders[key] = loader
                for loaded_key, skill in (
                    loader.discover_skills(source_text) or {}
                ).items():
                    runtime_id = (
                        getattr(skill, "skill_id", None)
                        or str(loaded_key).split("@", 1)[0]
                    )
                    enabled = runtime_id not in disabled and loaded_key not in disabled
                    rows.append((runtime_id, skill, enabled))
                    token_rows.append(
                        {
                            "source": source_text,
                            "id": runtime_id,
                            "version": getattr(skill, "version", "latest"),
                            "name": getattr(skill, "name", ""),
                            "description": getattr(skill, "description", ""),
                            "content": getattr(skill, "content", ""),
                            "path": str(getattr(skill, "skill_path", "")),
                            "enabled": enabled,
                        }
                    )

            token_payload = {
                "sources": raw_sources,
                "disabled": disabled_values,
                "rows": token_rows,
                "generation": change.generation,
            }
            token = hashlib.sha256(
                json.dumps(
                    token_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            state_payload["skill_markers"] = self._skill_markers(rows)
            state_token = hashlib.sha256(
                json.dumps(
                    state_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            snapshot = ScopeSkillSnapshot(
                scope=scope,
                rows=tuple(rows),
                token=token,
                state_token=state_token,
                generation=change.generation,
                trusted=(
                    change.ready and not change.fail_safe and not has_untracked_source
                ),
            )
            self._snapshots[scope] = snapshot
            return snapshot

    @staticmethod
    def _runtime_source_state(skills_config, tracked_roots: set[str]):
        """Return catalog source identity, additional roots, and trust.

        Managed global/project roots are already watched by their scope
        trackers.  This adds SDK builtins, YAML-declared local sources, and
        workspace auto-discovery without maintaining a per-file manifest.
        Remote sources retain the legacy full-sync path because this process
        has no authoritative signal for changes at the remote origin.
        """
        from ms_agent.skill.catalog import BUILTIN_SKILLS_DIR, USER_SKILLS_DIR

        rows: list[dict] = []
        extra_roots: set[Path] = set()
        trusted = True

        def _record_local(label: str, raw_path) -> None:
            nonlocal trusted
            if not raw_path:
                trusted = False
                rows.append({"kind": label, "path": None})
                return
            path = Path(str(raw_path)).expanduser().resolve()
            is_file = path.is_file()
            is_dir = path.is_dir()
            row = {
                "kind": label,
                "path": str(path),
                "is_file": is_file,
                "is_dir": is_dir,
            }
            if is_file:
                try:
                    file_stat = path.stat()
                    row["mtime_ns"] = file_stat.st_mtime_ns
                    row["size"] = file_stat.st_size
                except OSError:
                    trusted = False
                watch_root = path.parent
            else:
                watch_root = path if is_dir else None
            rows.append(row)
            if watch_root is not None and str(watch_root) not in tracked_roots:
                extra_roots.add(watch_root)

        _record_local("builtin", BUILTIN_SKILLS_DIR)
        _record_local("user", USER_SKILLS_DIR)

        try:
            sources = list(getattr(skills_config, "sources", None) or [])
            if sources:
                for source in sources:
                    raw_type = getattr(source, "type", "local")
                    source_type = str(getattr(raw_type, "value", raw_type))
                    source_row = {
                        key: getattr(source, key, None)
                        for key in (
                            "type",
                            "repo_id",
                            "url",
                            "revision",
                            "subdir",
                            "enabled",
                            "origin",
                            "plugin_id",
                            "capability",
                        )
                    }
                    source_row["type"] = source_type
                    rows.append(
                        {
                            "kind": "source",
                            **source_row,
                            "path": str(getattr(source, "path", None) or ""),
                        }
                    )
                    if not bool(getattr(source, "enabled", True)):
                        continue
                    if source_type not in {"local", "local_dir"}:
                        trusted = False
                        continue
                    _record_local("source-path", getattr(source, "path", None))
            else:
                legacy_paths = getattr(skills_config, "path", None)
                if legacy_paths:
                    values = (
                        [legacy_paths]
                        if isinstance(legacy_paths, str)
                        else legacy_paths
                    )
                    for value in values:
                        _record_local("legacy-path", value)

            if bool(getattr(skills_config, "auto_discover", False)):
                _record_local("workspace", Path.cwd() / "skills")
        except Exception:
            trusted = False
        return rows, tuple(sorted(extra_roots, key=str)), trusted

    def runtime_snapshot(
        self, project_id: str, project_path: str, skills_config=None
    ) -> RuntimeSkillSnapshot:
        global_snapshot = self.snapshot("global")
        project_scope = f"project:{project_id}"
        project_snapshot = self.snapshot(project_scope, project_path)
        tracked_roots = set(self._tracker.snapshot("global").roots)
        tracked_roots.update(self._tracker.snapshot(project_scope).roots)
        source_rows, extra_roots, sources_trusted = self._runtime_source_state(
            skills_config, tracked_roots
        )
        runtime_scope = f"runtime:{project_id}"
        runtime_change = self._tracker.register(runtime_scope, extra_roots)
        source_token = hashlib.sha256(
            json.dumps(
                source_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        token = hashlib.sha256(
            (
                f"{global_snapshot.token}\x1f{project_snapshot.token}\x1f"
                f"{source_token}\x1f{runtime_change.generation}"
            ).encode()
        ).hexdigest()
        return RuntimeSkillSnapshot(
            token=token,
            trusted=(
                global_snapshot.trusted
                and project_snapshot.trusted
                and runtime_change.ready
                and not runtime_change.fail_safe
                and sources_trusted
            ),
        )

    def invalidate(self, scope: str) -> None:
        with self._lock:
            self._tracker.mark_dirty(scope)
            self._snapshots.pop(scope, None)
            for key in list(self._loaders):
                if key[0] == scope:
                    self._loaders.pop(key, None)

    def stop(self) -> None:
        with self._lock:
            self._loaders.clear()
            self._snapshots.clear()
        self._tracker.stop_all()


skill_index = SkillIndex()
