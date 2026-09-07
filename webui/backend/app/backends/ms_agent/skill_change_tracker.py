"""Process-local, scope-aware change tracking for local skill trees.

The tracker stores only a monotonically increasing generation per scope.  It
does not persist manifests or file contents; a generation change asks the SDK
to perform its existing full reconciliation at the next turn boundary.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from watchfiles import watch

logger = logging.getLogger("app.ms_agent.skill_change_tracker")


@dataclass(frozen=True)
class ChangeSnapshot:
    generation: int
    ready: bool
    fail_safe: bool
    roots: tuple[str, ...]
    batches: int
    events: int
    error: str | None = None


@dataclass
class _ScopeState:
    roots: tuple[Path, ...]
    generation: int = 0
    batches: int = 0
    events: int = 0
    fail_safe: bool = False
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    ready_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class SkillChangeTracker:
    """Own one recursive native watcher per active skill scope."""

    def __init__(self, watch_fn: Callable = watch) -> None:
        self._watch_fn = watch_fn
        self._lock = threading.RLock()
        self._states: dict[str, _ScopeState] = {}

    @staticmethod
    def _normalise_roots(roots: Iterable[str | Path]) -> tuple[Path, ...]:
        resolved = {
            Path(root).expanduser().resolve()
            for root in roots
            if Path(root).expanduser().is_dir()
        }
        return tuple(sorted(resolved, key=str))

    @staticmethod
    def _visible_filter(roots: tuple[Path, ...]):
        def _filter(_change, raw_path: str) -> bool:
            path = Path(raw_path)
            for root in roots:
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                return not any(part.startswith(".") for part in relative.parts)
            return False

        return _filter

    def register(self, scope: str, roots: Iterable[str | Path]) -> ChangeSnapshot:
        """Start or update a scope watcher and return its current state."""
        normalised = self._normalise_roots(roots)
        previous: _ScopeState | None = None
        with self._lock:
            current = self._states.get(scope)
            if (
                current is not None
                and current.roots == normalised
                and (
                    current.fail_safe
                    or not normalised
                    or (current.thread is not None and current.thread.is_alive())
                )
            ):
                state = current
            else:
                previous = current
                if previous is not None:
                    previous.stop_event.set()
                state = _ScopeState(
                    roots=normalised,
                    generation=(previous.generation if previous else 0)
                    + (1 if previous is not None else 0),
                )
                self._states[scope] = state
                if not normalised:
                    state.ready_event.set()
                else:
                    state.thread = threading.Thread(
                        target=self._run,
                        name=f"skill-watch-{scope}",
                        args=(scope, state),
                        daemon=True,
                    )
                    state.thread.start()

        if previous is not None and previous.thread is not None:
            previous.thread.join(timeout=0.25)
        # The first timeout yield happens after the native watcher is active.
        # Waiting briefly closes the startup race without walking the tree.
        if normalised:
            state.ready_event.wait(timeout=0.25)
        return self.snapshot(scope)

    def _run(self, scope: str, state: _ScopeState) -> None:
        try:
            changes_iter = self._watch_fn(
                *state.roots,
                watch_filter=self._visible_filter(state.roots),
                debounce=80,
                step=20,
                stop_event=state.stop_event,
                rust_timeout=20,
                yield_on_timeout=True,
                recursive=True,
                raise_interrupt=False,
            )
            for changes in changes_iter:
                state.ready_event.set()
                if state.stop_event.is_set():
                    break
                if not changes:
                    continue
                with self._lock:
                    if self._states.get(scope) is not state:
                        break
                    state.generation += 1
                    state.batches += 1
                    state.events += len(changes)
        except Exception as exc:
            with self._lock:
                if self._states.get(scope) is state and not state.stop_event.is_set():
                    state.fail_safe = True
                    state.error = f"{type(exc).__name__}: {exc}"
                    state.generation += 1
                    logger.warning(
                        "skill watcher failed for %s; using full-scan fail-safe",
                        scope,
                        exc_info=True,
                    )
        finally:
            state.ready_event.set()

    def mark_dirty(self, scope: str) -> int:
        """Synchronously invalidate a scope after a WebUI-owned mutation."""
        with self._lock:
            state = self._states.get(scope)
            if state is None:
                state = _ScopeState(roots=())
                state.ready_event.set()
                self._states[scope] = state
            state.generation += 1
            return state.generation

    def snapshot(self, scope: str) -> ChangeSnapshot:
        with self._lock:
            state = self._states.get(scope)
            if state is None:
                return ChangeSnapshot(0, True, False, (), 0, 0)
            return ChangeSnapshot(
                generation=state.generation,
                ready=state.ready_event.is_set(),
                fail_safe=state.fail_safe,
                roots=tuple(str(root) for root in state.roots),
                batches=state.batches,
                events=state.events,
                error=state.error,
            )

    def stop_all(self) -> None:
        with self._lock:
            states = list(self._states.values())
            self._states.clear()
        for state in states:
            state.stop_event.set()
        for state in states:
            if state.thread is not None:
                state.thread.join(timeout=0.25)


skill_change_tracker = SkillChangeTracker()
