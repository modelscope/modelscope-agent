"""restore_cooked_tty must never raise when stdin is not a TTY."""
from ms_agent.tui.tty import restore_cooked_tty


def test_restore_cooked_tty_is_safe():
    restore_cooked_tty()
