"""MCP ${VAR} placeholder support: .env injection into os.environ and the
probe's runtime view (placeholders resolved before any handshake)."""
import os

from app.backends.ms_agent import mcp_health


def test_env_files_published_to_environ():
    """settings.py publishes backend/.env + repo-root .env into os.environ
    (key presence only — values are never asserted or printed)."""
    from dotenv import dotenv_values

    from app.core.settings import _ENV_FILES, settings

    # Compare against whatever the dotenv chain actually declares rather than one
    # hard-coded key: this asserts the injection ran (conftest imports
    # app.core.settings) without breaking the next time a key is renamed.
    declared = {
        key
        for env_file in _ENV_FILES
        if env_file.exists()
        for key, value in dotenv_values(env_file).items()
        if value is not None
    }
    if not declared:
        import pytest

        pytest.skip("no .env present (copy backend/.env.example to run this)")

    assert declared <= set(os.environ)
    assert settings.cors_origin_list  # settings side intact


def test_probe_runtime_view_expands_placeholders(monkeypatch):
    monkeypatch.setenv("PROBE_TOKEN", "tok-123")
    entry = {
        "url": "https://gw/sse",
        "transport": "sse",
        "headers": {"Authorization": "Bearer ${PROBE_TOKEN}"},
    }
    view = mcp_health._runtime_view(entry)
    assert view["headers"]["Authorization"] == "Bearer tok-123"
    # Original entry untouched (management surfaces keep the placeholder).
    assert entry["headers"]["Authorization"] == "Bearer ${PROBE_TOKEN}"
    # Idempotent on already-expanded entries.
    assert mcp_health._runtime_view(view) == view


def test_probe_stdio_uses_expanded_command(monkeypatch):
    monkeypatch.setenv("MY_BIN", "python3")
    entry = {"command": "${MY_BIN}", "args": ["-V"]}
    import asyncio

    ok, err = asyncio.run(mcp_health.check_server(entry))
    assert ok, err  # python3 resolves on PATH once expanded
