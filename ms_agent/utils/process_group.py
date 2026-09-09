"""Kill a spawned shell and every process in its group.

``asyncio.create_subprocess_shell`` runs ``sh -c …``. ``Process.kill()`` only
signals the shell; grandchildren such as ``ssh`` are reparented and keep the
TTY. With ``start_new_session=True`` the shell is the session leader, so
``killpg`` tears down the whole tree.
"""

from __future__ import annotations

import os
import signal
from typing import Any


def kill_process_group(process: Any) -> None:
    """SIGKILL the process group (POSIX) or the process (elsewhere)."""
    pid = getattr(process, 'pid', None)
    if not pid:
        return
    if os.name == 'posix':
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass
