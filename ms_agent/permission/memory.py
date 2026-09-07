"""PermissionMemory: persist user ``allow_always`` decisions across sessions.

Two storage scopes:
  - Project: ``.ms_agent/permission_memory.json``
  - Global:  ``~/.ms_agent/permission_memory.json``

Session-level memory (``allow_session``) lives only in-process.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from .matcher import PermissionMatcher

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


RuleKind = Literal['tool', 'shell', 'file', 'domain']


def infer_rule_kind(pattern: str) -> RuleKind:
    if ':domain:' in pattern or ':url-domain:' in pattern:
        return 'domain'
    if '---shell_executor' in pattern:
        return 'shell'
    if pattern.startswith('file_system---'):
        return 'file'
    return 'tool'


@dataclass(frozen=True)
class MemoryEntry:
    pattern: str
    scope: Literal['project', 'global']
    source: Literal['user', 'plugin', 'hook'] = 'user'
    created_at: str = ''
    id: str = ''
    kind: RuleKind = 'tool'


class PermissionMemory:
    """Manages persistent and session-level permission rules."""

    def __init__(
        self,
        project_path: str | Path | None = None,
        global_path: str | Path | None = None,
    ) -> None:
        self._matcher = PermissionMatcher()
        self._lock = threading.RLock()

        self._project_file: Path | None = None
        if project_path is not None:
            self._project_file = Path(
                project_path) / '.ms_agent' / 'permission_memory.json'

        if global_path is not None:
            self._global_file = Path(global_path)
        else:
            self._global_file = Path.home(
            ) / '.ms_agent' / 'permission_memory.json'

        self._project_entries: list[MemoryEntry] = []
        self._global_entries: list[MemoryEntry] = []
        self._session_patterns: list[str] = []

        self._project_root = (
            Path(project_path) if project_path is not None else None)

        self._load()

    @property
    def project_root(self) -> Path | None:
        return self._project_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        pattern: str,
        scope: Literal['project', 'global'] = 'project',
        source: Literal['user', 'plugin', 'hook'] = 'user',
        kind: RuleKind | None = None,
    ) -> MemoryEntry:
        if scope not in ('project', 'global'):
            raise ValueError(f'Unknown permission scope: {scope}')
        if kind is not None and kind not in ('tool', 'shell', 'file', 'domain'):
            raise ValueError(f'Unknown permission rule kind: {kind}')
        with self._scope_transaction(scope):
            entries = (
                self._project_entries
                if scope == 'project' else self._global_entries)
            existing = next((e for e in entries if e.pattern == pattern), None)
            if existing is not None:
                return existing
            entry = MemoryEntry(
                pattern=pattern,
                scope=scope,
                source=source,
                created_at=datetime.now(timezone.utc).isoformat(),
                id=uuid4().hex,
                kind=kind or infer_rule_kind(pattern),
            )
            entries.append(entry)
            return entry

    def add_session(self, pattern: str) -> None:
        if pattern not in self._session_patterns:
            self._session_patterns.append(pattern)

    def matches(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        with self._read_all():
            for pattern in self._session_patterns:
                if self._matcher.match_with_content(
                        pattern, tool_name, tool_args):
                    return True
            for entry in self._project_entries:
                if self._matcher.match_with_content(
                        entry.pattern, tool_name, tool_args):
                    return True
            for entry in self._global_entries:
                if self._matcher.match_with_content(
                        entry.pattern, tool_name, tool_args):
                    return True
            return False

    def revoke(self, pattern: str) -> int:
        """Remove all entries matching the given pattern. Returns count removed."""
        with self._all_scopes_transaction():
            count = 0
            before = len(self._project_entries)
            self._project_entries = [
                e for e in self._project_entries if e.pattern != pattern
            ]
            count += before - len(self._project_entries)

            before = len(self._global_entries)
            self._global_entries = [
                e for e in self._global_entries if e.pattern != pattern
            ]
            count += before - len(self._global_entries)

            self._session_patterns = [
                p for p in self._session_patterns if p != pattern
            ]
            return count

    def list_all(self) -> list[MemoryEntry]:
        return list(self._project_entries) + list(self._global_entries)

    def list(
        self,
        scope: Literal['project', 'global'] | None = None,
    ) -> list[MemoryEntry]:
        """List persistent entries, optionally filtered by scope."""
        entries = self.list_all()
        return [e for e in entries if scope is None or e.scope == scope]

    def update(
        self,
        entry_id: str,
        *,
        pattern: str | None = None,
        scope: Literal['project', 'global'] | None = None,
        source: Literal['user', 'plugin', 'hook'] | None = None,
        kind: RuleKind | None = None,
    ) -> MemoryEntry:
        """Update one entry while retaining its stable identifier."""
        if scope is not None and scope not in ('project', 'global'):
            raise ValueError(f'Unknown permission scope: {scope}')
        if kind is not None and kind not in ('tool', 'shell', 'file', 'domain'):
            raise ValueError(f'Unknown permission rule kind: {kind}')
        with self._all_scopes_transaction():
            current = next(
                (e for e in self.list_all() if e.id == entry_id), None)
            if current is None:
                raise KeyError(entry_id)
            target_scope = scope or current.scope
            updated = replace(
                current,
                pattern=current.pattern if pattern is None else pattern,
                scope=target_scope,
                source=current.source if source is None else source,
                kind=current.kind if kind is None else kind,
            )
            target = (
                self._project_entries
                if target_scope == 'project' else self._global_entries)
            if any(
                    e.pattern == updated.pattern and e.id != entry_id
                    for e in target):
                raise ValueError(
                    f'Permission pattern already exists: {updated.pattern}')
            self._project_entries = [
                e for e in self._project_entries if e.id != entry_id
            ]
            self._global_entries = [
                e for e in self._global_entries if e.id != entry_id
            ]
            target = (
                self._project_entries
                if target_scope == 'project' else self._global_entries)
            target.append(updated)
            return updated

    def delete(self, entry_id: str) -> bool:
        """Delete one persistent entry by identifier."""
        with self._all_scopes_transaction():
            before_project = len(self._project_entries)
            before_global = len(self._global_entries)
            self._project_entries = [
                e for e in self._project_entries if e.id != entry_id
            ]
            self._global_entries = [
                e for e in self._global_entries if e.id != entry_id
            ]
            return (
                before_project != len(self._project_entries)
                or before_global != len(self._global_entries)
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _file_for_scope(
        self,
        scope: Literal['project', 'global'],
    ) -> Path | None:
        return self._project_file if scope == 'project' else self._global_file

    def _reload_scope(self, scope: Literal['project', 'global']) -> None:
        path = self._file_for_scope(scope)
        if path is None:
            return
        entries = self._load_file(path, scope)
        if scope == 'project':
            self._project_entries = entries
        else:
            self._global_entries = entries

    @contextmanager
    def _read_all(self):
        paths = sorted(
            {
                path
                for path in (self._project_file, self._global_file)
                if path is not None
            },
            key=str,
        )
        with self._lock, ExitStack() as stack:
            for path in paths:
                stack.enter_context(self._file_lock(path))
            if self._project_file is not None:
                self._reload_scope('project')
            if self._global_file is not None:
                self._reload_scope('global')
            yield

    @contextmanager
    def _file_lock(self, path: Path | None):
        if path is None:
            yield
            return
        lock_path = path.with_suffix(f'{path.suffix}.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(lock_path, 0o600)
        with os.fdopen(fd, 'a+', encoding='utf-8') as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write('\0')
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    @contextmanager
    def _scope_transaction(
        self,
        scope: Literal['project', 'global'],
    ):
        with self._lock, self._file_lock(self._file_for_scope(scope)):
            self._reload_scope(scope)
            try:
                yield
                self._save(scope)
            except Exception:
                self._reload_scope(scope)
                raise

    @contextmanager
    def _all_scopes_transaction(self):
        paths = sorted(
            {
                path
                for path in (self._project_file, self._global_file)
                if path is not None
            },
            key=str,
        )
        with self._lock, ExitStack() as stack:
            for path in paths:
                stack.enter_context(self._file_lock(path))
            self._load()
            try:
                yield
                self._commit_all()
            except Exception:
                self._load()
                raise

    @staticmethod
    def _load_file(path: Path | None, scope: str) -> list[MemoryEntry]:
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                data = data.get('entries', [])
            entries: list[MemoryEntry] = []
            for raw in data:
                e = {'pattern': raw} if isinstance(raw, str) else raw
                pattern = e['pattern']
                entry_scope = e.get('scope', scope)
                source = e.get('source', 'user')
                created_at = e.get('created_at', '')
                kind = e.get('kind') or infer_rule_kind(pattern)
                stable_id = e.get('id') or uuid5(
                    NAMESPACE_URL,
                    json.dumps(
                        {
                            'scope': entry_scope,
                            'pattern': pattern,
                            'source': source,
                            'created_at': created_at,
                        },
                        sort_keys=True,
                    ),
                ).hex
                entries.append(MemoryEntry(
                    pattern=pattern,
                    scope=entry_scope,
                    source=source,
                    created_at=created_at,
                    id=stable_id,
                    kind=kind,
                ))
            return entries
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _save(self, scope: Literal['project', 'global']) -> None:
        if scope == 'project':
            self._save_file(self._project_file, self._project_entries)
        else:
            self._save_file(self._global_file, self._global_entries)

    def _journal_path(self) -> Path | None:
        path = self._project_file or self._global_file
        if path is None:
            return None
        return path.parent / f'.{path.name}.txn'

    def _commit_all(self) -> None:
        journal_path = self._journal_path()
        snapshot = {
            'project': self._read_raw(self._project_file),
            'global': self._read_raw(self._global_file),
        }
        if journal_path is not None:
            self._write_text(journal_path, json.dumps(snapshot))
        try:
            self._save('project')
            self._save('global')
        except Exception:
            self._restore_raw(self._project_file, snapshot['project'])
            self._restore_raw(self._global_file, snapshot['global'])
            raise
        finally:
            if journal_path is not None:
                try:
                    journal_path.unlink()
                except FileNotFoundError:
                    pass

    def _recover_journal(self) -> None:
        path = self._journal_path()
        if path is None or not path.exists():
            return
        try:
            snapshot = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            snapshot = None
        if isinstance(snapshot, dict):
            self._restore_raw(self._project_file, snapshot.get('project'))
            self._restore_raw(self._global_file, snapshot.get('global'))
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _load(self) -> None:
        self._recover_journal()
        self._project_entries = self._load_file(self._project_file, 'project')
        self._global_entries = self._load_file(self._global_file, 'global')

    @staticmethod
    def _read_raw(path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        return path.read_text(encoding='utf-8')

    @staticmethod
    def _restore_raw(path: Path | None, payload: str | None) -> None:
        if path is None:
            return
        if payload is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        PermissionMemory._write_text(path, payload)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _save_file(path: Path | None, entries: list[MemoryEntry]) -> None:
        if path is None:
            return
        PermissionMemory._write_text(
            path,
            json.dumps(
                [asdict(e) for e in entries],
                indent=2,
                ensure_ascii=False,
            ),
        )
