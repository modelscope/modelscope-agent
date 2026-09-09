"""Restore a cooked TTY after prompt_toolkit / child processes.

Raw mode (permission select, leftover ssh) turns off ``ONLCR``. Then ``\\n``
only moves down — it does not return to column 0 — so logs, the banner, and
the prompt staircase to the right. Call this before any TUI print and after
every inline Application.
"""

from __future__ import annotations

import sys


def restore_cooked_tty() -> None:
    """Re-enable ONLCR/ICANON/ECHO and send CR so the next print starts at col 0."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            fd = stream.fileno()
        except Exception:
            continue
        if fd < 0:
            continue
        try:
            if not stream.isatty():
                continue
        except Exception:
            continue
        _restore_fd(fd)
    try:
        sys.stdout.write('\r')
        sys.stdout.flush()
    except Exception:
        pass


def _restore_fd(fd: int) -> None:
    try:
        import termios
    except ImportError:
        return
    try:
        attrs = termios.tcgetattr(fd)
    except termios.error:
        return
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs
    oflag |= termios.OPOST | termios.ONLCR
    iflag |= termios.ICRNL
    lflag |= (
        termios.ECHO | termios.ECHOE | termios.ECHOK
        | termios.ICANON | termios.ISIG | termios.IEXTEN)
    echonl = getattr(termios, 'ECHONL', 0)
    if echonl:
        lflag |= echonl
    try:
        termios.tcsetattr(
            fd, termios.TCSADRAIN,
            [iflag, oflag, cflag, lflag, ispeed, ospeed, cc])
    except termios.error:
        pass
