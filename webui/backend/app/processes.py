# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process handling shared by the launcher and SDK dependency preparation."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

IS_WINDOWS = os.name == "nt"
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
CTRL_BREAK_EVENT = getattr(signal, "CTRL_BREAK_EVENT", 1)


class ProcessError(RuntimeError):
    pass


@contextmanager
def interruptible():
    """Turn termination into cleanup, then restore the caller's handlers."""
    previous = {}

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    try:
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            if handler is not None:
                signal.signal(sig, handler)


def _spawn(
    command: List[str], cwd: Path, env: Optional[Dict[str, str]], shell: bool = False
) -> subprocess.Popen:
    kwargs = {
        "cwd": str(cwd),
        "env": env,
    }
    if shell:
        kwargs["shell"] = True
    if IS_WINDOWS:
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    try:
        return subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise ProcessError(f"Could not start {Path(command[0]).name}: {exc}") from exc


def _terminate_process_tree(
    process: subprocess.Popen, grace_seconds: float = 5.0
) -> None:
    """Stop a launcher child and its descendants cross-platform."""
    leader_running = process.poll() is None

    if IS_WINDOWS:
        if not leader_running:
            # Best effort: taskkill can still find the tree during the short
            # interval before Windows finishes re-parenting descendants.
            if not _taskkill(process.pid, force=True):
                _warn_cleanup(process.pid)
            return
        try:
            process.send_signal(CTRL_BREAK_EVENT)
        except OSError:
            if not _taskkill(process.pid, force=True):
                _warn_cleanup(process.pid)
            try:
                process.wait(timeout=grace_seconds)
            except (OSError, subprocess.TimeoutExpired):
                _warn_cleanup(process.pid)
            return
    else:
        # A process-group leader may already have exited while descendants are
        # still alive. killpg remains valid in that state, so do not return
        # merely because Popen.poll() has a return code.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            if leader_running:
                try:
                    process.terminate()
                except OSError:
                    pass
        if not leader_running and _wait_for_posix_group_exit(
            process.pid, grace_seconds
        ):
            return
        if not leader_running:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            if not _wait_for_posix_group_exit(process.pid, grace_seconds):
                _warn_cleanup(process.pid)
            return

    try:
        process.wait(timeout=grace_seconds)
        if IS_WINDOWS or _wait_for_posix_group_exit(process.pid, 0.25):
            return
    except (OSError, subprocess.TimeoutExpired):
        pass

    if IS_WINDOWS:
        if not _taskkill(process.pid, force=True):
            _warn_cleanup(process.pid)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass

    try:
        process.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired):
        _warn_cleanup(process.pid)


def _taskkill(pid: int, force: bool) -> bool:
    command = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        command.append("/F")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _warn_cleanup(pid: int) -> None:
    print(
        f"Warning: could not confirm cleanup of process tree {pid}. "
        "Check for remaining Python/Node processes.",
        file=sys.stderr,
        flush=True,
    )


def _wait_for_posix_group_exit(pid: int, timeout: float) -> bool:
    """Wait until a POSIX process group no longer has live members."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        except OSError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
