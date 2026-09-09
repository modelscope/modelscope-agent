"""Async MCP health probe.

A remote MCP whose host is reachable but whose endpoint is stale/invalid (e.g.
"Session terminated") still breaks a chat turn: the agent's connect raises and
aborts the run. A TCP check can't catch that — only a real MCP handshake can.

This probes each enabled server (connect + initialize) with a timeout, fully
isolated in its own task so a failure/hang/anyio-teardown can't touch the chat,
and returns only the servers that initialized. Runs once per session build.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil

logger = logging.getLogger("app.ms_agent.mcp_health")


async def _remote_handshake(server: dict) -> bool:
    """connect + initialize a remote MCP, entered/exited within THIS task so an
    anyio cross-task teardown can't leak. Raises on failure."""
    url = server["url"]
    transport = str(server.get("transport") or "").lower()
    headers = server.get("headers") or None
    from mcp import ClientSession

    if transport == "sse":
        from mcp.client.sse import sse_client as connect
    else:  # http / streamable_http
        from mcp.client.streamable_http import streamablehttp_client as connect
    async with connect(url, headers=headers) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
    return True


async def _probe_remote(server: dict, timeout: float) -> bool:
    if not server.get("url"):
        return False
    try:
        return bool(await asyncio.wait_for(_remote_handshake(server), timeout))
    except Exception:
        return False


def _short_error(exc: BaseException) -> str:
    """Unwrap anyio ExceptionGroups to a short, human-readable reason."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    msg = str(exc).strip() or type(exc).__name__
    return msg[:200]


async def _http_status(server: dict, timeout: float = 2.5) -> int | None:
    """The endpoint's HTTP status, or None if it could not be obtained.

    The MCP client reports its own rejections as "Session terminated", which is
    true and completely unactionable: it reads as "the service is down" when the
    actual cause is usually a wrong URL. The status code separates those — a 404
    sends you to the address bar, a 401/403 to the credentials, a 5xx to the
    provider. Measured cause of every "Session terminated" seen here: a URL
    whose path segment was empty (``.../net//mcp``), i.e. a 404.

    Best-effort by construction: only ever called on the failure path, given a
    short deadline of its own, and any problem it hits returns None so the
    original reason is reported unchanged.
    """
    url = server.get("url")
    if not url:
        return None
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout,
                                     follow_redirects=True) as client:
            # POST an `initialize`, which is what the real handshake opens with,
            # so a server that only rejects that verb is not mislabelled. The
            # body is irrelevant — only the status is read, never logged.
            resp = await client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **(server.get("headers") or {}),
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "health", "version": "0"},
                    },
                },
            )
            return resp.status_code
    except Exception:  # noqa: BLE001 — an annotation must never mask the reason
        return None


async def _annotate(reason: str, server: dict) -> str:
    """``reason`` with the HTTP status appended when one can be established."""
    if not reason or "HTTP" in reason:
        return reason
    status = await _http_status(server)
    return f"{reason} (HTTP {status})" if status is not None else reason


def _runtime_view(server: dict) -> dict:
    """The server entry as the runtime will actually use it: ${VAR}
    placeholders resolved from the process environment. Management surfaces
    keep the placeholder form; probing with it would 401 against a healthy
    server. Idempotent on already-expanded entries."""
    from ms_agent.tui.managed_config import expand_env_placeholders

    return expand_env_placeholders(server)


async def _stdio_handshake(server: dict) -> bool:
    """Spawn the stdio server and complete a real MCP initialize, in THIS task.

    The shallow probe only answers "does the command exist" — a package that
    crashes on startup (observed: a server built against an older mcp SDK)
    keeps its green light right up until a session actually needs it. This is
    the truth an explicit single-server check owes the user. It can be slow (a
    cold uvx install downloads the package), which is exactly why only
    explicit checks run it, never the page-load sweep."""
    import tempfile

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from ms_agent.tools.mcp_client import (resolve_stdio_command,
                                           stdio_child_env)

    params = StdioServerParameters(
        command=resolve_stdio_command(server["command"]),
        args=list(server.get("args") or []),
        env=stdio_child_env(server.get("env")),
    )
    # The child's stderr is where a crashing package explains itself (its
    # traceback); without capturing it the client only ever reports
    # "Connection closed", which tells the user nothing actionable. A real
    # file, not StringIO: the SDK hands errlog to the subprocess as its
    # stderr, which needs a file descriptor.
    with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace") as errlog:
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
        except Exception as exc:
            tail = ""
            try:
                errlog.flush()
                errlog.seek(0)
                tail = errlog.read().strip()
            except Exception:  # noqa: BLE001 — the original error still stands
                pass
            if tail:
                raise RuntimeError(
                    f"{_short_error(exc)} — server stderr: …{tail[-300:]}"
                ) from exc
            raise
    return True


async def check_server(server: dict,
                       timeout: float = 6.0,
                       deep: bool = False) -> tuple[bool, str | None]:
    """Probe one server and return (healthy, error_reason). Same handshake as the
    build-time filter, but surfaces WHY an enabled server was dropped.

    ``deep`` upgrades the stdio probe from command-existence to a real
    spawn+initialize; remote probes are already real handshakes either way.

    A TIMEOUT is retried once; every other failure is taken at face value. The
    asymmetry is the point: a protocol answer ("Session terminated", 404) is the
    server telling us it is unusable, while a timeout only says nobody answered
    within N seconds of WALL CLOCK — which this process can cause by itself.
    The probe shares a runtime with a streaming turn and up to a dozen sibling
    handshakes, so a coroutine can be starved past the deadline while the
    network would have answered in well under a second (measured: 0.55-0.81 s
    warm and cold for the same endpoint that reported "timed out after 6s").
    Believing that first timeout is how a working server got struck through in
    the UI and dropped from the agent.
    """
    server = _runtime_view(server)
    if server.get("command"):
        ok = _probe_stdio(server)
        if not ok:
            return False, "command not found on PATH"
        if not deep:
            return True, None
        try:
            await asyncio.wait_for(_stdio_handshake(server), timeout)
            return True, None
        except asyncio.TimeoutError:
            return False, f"startup timed out after {int(timeout)}s"
        except Exception as exc:  # noqa: BLE001 — report the reason
            return False, _short_error(exc)
    if not server.get("url"):
        return False, "server has no url or command"
    for attempt in (1, 2):
        try:
            await asyncio.wait_for(_remote_handshake(server), timeout)
            return True, None
        except asyncio.TimeoutError:
            if attempt == 1:
                logger.debug("MCP probe timed out, retrying once: %s",
                             server.get("url"))
                continue
            # Deliberately NOT annotated: nothing answered, so asking again
            # would only add a second hang to a reason that is already clear.
            return False, f"timed out after {int(timeout)}s"
        except Exception as exc:  # noqa: BLE001 — report the reason, don't raise
            # The server DID answer, it just refused — so a status probe returns
            # immediately and turns "Session terminated" into something the user
            # can act on.
            return False, await _annotate(_short_error(exc), server)
    return False, f"timed out after {int(timeout)}s"


def _probe_stdio(server: dict) -> bool:
    command = server.get("command")
    if not command:
        return True
    try:
        # The SDK's resolver also searches the usual user bin dirs
        # (~/.local/bin, /opt/homebrew/bin, /usr/local/bin): a backend launched
        # by an IDE or launchd runs with a PATH that has never heard of uvx,
        # and judging by PATH alone dropped servers the SDK could spawn fine.
        from ms_agent.tools.mcp_client import resolve_stdio_command

        resolve_stdio_command(command)
        return True
    except FileNotFoundError:
        return False
    except Exception:  # noqa: BLE001 — health must not break on an import
        return bool(shutil.which(command)) or os.path.isfile(command)


#: name -> (monotonic deadline, healthy) for probes already performed. The probe
#: runs once per SESSION BUILD, so without this a single unreachable server
#: costs the full timeout every time a conversation starts — measured at a flat
#: +6 s per new chat against one black-holed address.
#:
#: Successes and failures are NOT cached for the same length of time, and the
#: difference matters more than the caching does. A cached success is cheap and
#: self-correcting: the worst case is one stale-good minute. A cached FAILURE is
#: load-bearing — `filter_healthy` uses it to withhold the server from the
#: model — so pinning one for a minute means a single spurious timeout silently
#: removes a working server's tools from an entire conversation. That is exactly
#: what happened to a healthy endpoint that answers in 0.7 s. Failures are
#: therefore remembered only long enough to absorb one burst (a page load and a
#: session build land within seconds of each other), not long enough to outlive
#: the condition that caused them.
_TTL_OK = 60.0
_TTL_FAIL = 10.0
_cache: dict[str, tuple[float, bool]] = {}


def _ttl(ok: bool) -> float:
    return _TTL_OK if ok else _TTL_FAIL


def _cache_key(name: str, server: dict) -> str:
    """Identity for caching: the wire target, not just the display name."""
    return f"{name}\x1f{server.get('url') or server.get('command') or ''}"


def invalidate_cache() -> None:
    """Forget every probe result (used by explicit user-triggered re-checks)."""
    _cache.clear()


def invalidate_server(name: str, server: dict) -> None:
    """Forget ONE server's probe result. A single-row re-check used to clear
    the whole cache, so the next page-load sweep re-paid every dead entry's
    timeout because one server was retested."""
    key = _cache_key(name, _runtime_view(server))
    _cache.pop(key, None)
    _reasons.pop(key, None)


#: name -> reason, alongside `_cache`, so a cached miss can still say why.
_reasons: dict[str, str] = {}


async def check_server_cached(name: str,
                              server: dict,
                              timeout: float = 6.0,
                              deep: bool = False
                              ) -> tuple[bool, str | None]:
    """:func:`check_server` with the shared TTL cache in front of it.

    Used by the listing endpoints, which probe every enabled server at once:
    without this, opening the MCP page (or reopening it) re-paid the full
    timeout for each dead entry.
    """
    resolved = _runtime_view(server)
    key = _cache_key(name, resolved)
    now = asyncio.get_running_loop().time()
    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1], (None if hit[1] else _reasons.get(key, "unavailable"))
    ok, reason = await check_server(resolved, timeout, deep=deep)
    _cache[key] = (now + _ttl(ok), ok)
    if ok:
        _reasons.pop(key, None)
    else:
        _reasons[key] = reason or "unavailable"
    return ok, reason


async def filter_healthy(servers: dict,
                         timeout: float = 6.0,
                         use_cache: bool = True) -> tuple[dict, dict]:
    """Split ``servers`` into the ones that answer and the ones that do not.

    Returns ``(healthy, dropped)`` where ``dropped`` maps name -> reason. The
    reason used to be discarded, which is how a stale endpoint became invisible:
    the model quietly lost the tools while every UI surface still counted the
    server as attached.
    """
    if not servers:
        return {}, {}

    now = asyncio.get_running_loop().time()

    async def _check(name: str, server: dict) -> tuple[str, bool, str | None]:
        resolved = _runtime_view(server)
        key = _cache_key(name, resolved)
        if use_cache:
            hit = _cache.get(key)
            if hit and hit[0] > now:
                return name, hit[1], (None if hit[1] else _reasons.get(
                    key, "unavailable"))
        ok, reason = await check_server(resolved, timeout)
        _cache[key] = (now + _ttl(ok), ok)
        if ok:
            _reasons.pop(key, None)
        else:
            _reasons[key] = reason or "unavailable"
        return name, ok, reason

    results = await asyncio.gather(*(_check(n, s) for n, s in servers.items()))
    healthy = {name: servers[name] for name, ok, _ in results if ok}
    dropped = {
        name: (reason or "unavailable")
        for name, ok, reason in results if not ok
    }
    if dropped:
        logger.warning("dropping unhealthy MCP server(s): %s", ", ".join(
            f"{name} ({reason})" for name, reason in sorted(dropped.items())))
    return healthy, dropped
