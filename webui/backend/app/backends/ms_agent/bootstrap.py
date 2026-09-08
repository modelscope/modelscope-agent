"""ms_agent backend startup: home/env, default project, LLM settings seed.

Runs once at app boot. Idempotent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.settings import settings


_PROVIDER_BRAND_NAME_MIGRATIONS = {
    "kimi": (("Moonshot Kimi",), "Kimi (Moonshot AI)"),
    "dashscope": (
        (
            "Alibaba DashScope",
            "Alibaba Cloud Model Studio (DashScope)",
        ),
        "Alibaba (DashScope)",
    ),
}


def bootstrap() -> None:
    from app.backends.ms_agent.common import apply_home_env, home, pm

    from app.backends.ms_agent import model_link

    apply_home_env()
    _export_env()
    pm()  # ProjectManager.__init__ ensures ~/.ms_agent/projects + default project
    _ensure_prompt_files(home())
    _seed_llm_settings(home())
    _seed_tools_settings(home())
    _migrate_provider_brand_names(home())
    # Normalize the model link so the chat dropdown lists the active model and
    # default_model is a full "provider/model" (see model_link).
    model_link.ensure_link()
    _migrate_vision_default(home())
    _export_fastembed_cache()


def _ensure_prompt_files(home_dir: str) -> None:
    """Materialize the editable global prompt files during WebUI startup.

    The SDK owns the lifecycle rules: an existing file is preserved, a
    pristine built-in may be upgraded, and a file deliberately deleted after
    being seeded stays deleted. Starting the WebUI eagerly invokes that same
    logic so users can discover and manage the files before opening
    Personalization or sending the first chat message.
    """
    from ms_agent.prompting.workspace_files import ensure_home_files

    ensure_home_files(Path(home_dir))


def _migrate_provider_brand_names(home_dir: str) -> None:
    """Upgrade historical built-in display names exactly once.

    Built-in provider entries also carry credentials, endpoints and model
    catalogs, so the WebUI merges them over the SDK registry. Older homes saved
    the then-default display names in those entries; without a migration they
    permanently mask later brand-copy corrections in the registry. Only the
    exact historical defaults are rewritten. Arbitrary user names and every
    sibling field remain untouched.
    """
    import logging

    from app.backends.ms_agent import sidecar

    log = logging.getLogger("app.ms_agent.bootstrap")
    migration_key = "provider_brand_names_v2"
    try:
        flags = sidecar.get("flags", migration_key) or {}
        if flags.get("migrated"):
            return

        path = Path(home_dir) / "settings.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            providers = data.get("providers")
            changed = False
            if isinstance(providers, dict):
                for provider_id, names in (
                    _PROVIDER_BRAND_NAME_MIGRATIONS.items()
                ):
                    legacy_names, canonical_name = names
                    entry = providers.get(provider_id)
                    if (
                        isinstance(entry, dict)
                        and entry.get("name") in legacy_names
                    ):
                        entry["name"] = canonical_name
                        changed = True
            if changed:
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp, path)

        sidecar.merge("flags", migration_key, {"migrated": True})
    except Exception:  # a migration must never block boot
        log.warning("provider brand-name migration skipped", exc_info=True)


def _export_fastembed_cache() -> None:
    """Pin FASTEMBED_CACHE_PATH before anything can construct a TextEmbedding.

    mem0 builds its embedder deep inside the agent process with no cache_dir
    argument; only this env var reaches it. Exporting at boot makes every
    consumer — our probes, mem0, fastembed's own fallback download — resolve
    one shared, persistent cache (instead of fastembed's default in a TEMP
    dir), which is also where the ModelScope warm-up lays files out.
    """
    try:
        from app.backends.ms_agent.embed_warmup import fastembed_cache_dir

        fastembed_cache_dir()
    except Exception:  # cache pinning must never block boot
        import logging

        logging.getLogger("app.ms_agent.bootstrap").debug(
            "fastembed cache pinning failed", exc_info=True)


def _migrate_vision_default(home_dir: str) -> None:
    """Materialize the old implicit "unset" into an explicit ``false``. Once.

    "Image understanding" used to be a tri-state whose middle value fell through
    to the provider's declared ``vision`` capability — and because nearly every
    provider declares it, *unset meant images were sent*. The switch is now a
    plain two-state control that defaults to off, so leaving those models unset
    would silently flip behaviour under existing users with no record of why.

    Writing the value down instead makes the change visible and inspectable: the
    model form shows exactly what the runtime will do, and a user who wants
    images back ticks one box. Guarded by a marker so it never re-runs and can
    never overwrite a later choice.
    """
    import logging

    from app.backends.ms_agent import sidecar

    log = logging.getLogger("app.ms_agent.bootstrap")
    try:
        flags = sidecar.get("flags", "vision_two_state") or {}
        if flags.get("migrated"):
            return
        models = sidecar.section("models") or {}
        touched = [
            mid for mid, meta in models.items()
            if isinstance(meta, dict) and meta.get("supports_vision") is None
        ]
        for mid in touched:
            sidecar.merge("models", mid, {"supports_vision": False})
        sidecar.merge("flags", "vision_two_state", {"migrated": True})
        if touched:
            log.info(
                "image understanding is now a two-state switch; %d model(s) "
                "that were previously unset are recorded as off", len(touched))
    except Exception:  # a migration must never block boot
        log.warning("vision two-state migration skipped", exc_info=True)


def _export_env() -> None:
    """Push backend credentials into the process env the SDK / MCP servers read."""
    for key, value in {
        "OPENAI_API_KEY": settings.openai_api_key,
        "OPENAI_BASE_URL": settings.openai_base_url,
        "EXA_API_KEY": settings.exa_api_key,
    }.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def _seed_llm_settings(home_dir: str) -> None:
    """Write settings.json `llm` from env when absent, so ConfigResolver yields a
    working model. Matches §3.1: llm.{provider,model,api_key,base_url}."""
    path = Path(home_dir) / "settings.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    if "llm" in data:
        return  # never overwrite an existing config (e.g. a shared ~/.ms_agent)

    model = settings.ms_agent_llm_model
    if not model:
        return  # nothing to seed; rely on framework default + env credentials
    provider = settings.ms_agent_llm_provider or "openai"
    llm: dict = {"provider": provider, "model": model}
    # OPENAI_* are OpenAI's credentials, so only apply them when OpenAI is the
    # provider being seeded. Copying them onto e.g. dashscope pinned the wrong
    # base_url onto the llm block; leaving them out lets the SDK's
    # CredentialResolver pick up that provider's own env vars
    # (DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, …).
    if provider == "openai":
        if settings.openai_api_key:
            llm["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            llm["base_url"] = settings.openai_base_url
    data["llm"] = llm
    data.setdefault("default_model", f"{provider}/{model}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# Builtin repo tools seeded into settings.json so they are default-enabled and
# toggleable through the standard multi-level config resolve (framework yaml ->
# settings.json -> project yaml -> session). Presence of a `tools.<id>` key
# enables it; add `enabled: false` to disable (honored by the SDK ToolManager).
_DEFAULT_TOOLS: dict = {
    "file_system": {
        "mcp": False,
        "include": ["read_file", "grep", "glob", "edit_file", "write_file"],
    },
    "todo_list": {"mcp": False},
    # Shell/terminal only. implementation must be the SDK's "python_env" (a
    # local Jupyter kernel, no Docker); "local"/"sandbox" route to the Docker
    # CodeExecutionTool (needs ms-enclave). `include: [shell_executor]` exposes
    # ONLY the terminal tool (not notebook/python/file_operation). Restricted
    # permission gates shell_executor (not whitelisted) so every command asks.
    "code_executor": {
        "mcp": False,
        "implementation": "python_env",
        "include": ["shell_executor"],
    },
    # Web search is ON by default, and Tavily is the default engine because it
    # is the only web-wide one that WORKS with no credentials: the SDK falls
    # back to Tavily's keyless tier (tavily/search.py KEYLESS_HEADER) when no
    # key is configured, so a fresh install can search immediately instead of
    # silently having no engine until someone visits Settings -> Search. The
    # keyless quota is a small sliding hourly bucket; adding a key lifts it, and
    # a configured key always takes precedence.
    "web_search": {"mcp": False, "engine": "tavily", "enabled": True},
}

# task_control has no WebUI rendering component yet; strip it from any home that
# was seeded with the earlier default so it isn't loaded (idempotent migration).
_DROP_TOOLS = ("task_control",)


def _seed_tools_settings(home_dir: str) -> None:
    """Write the default builtin-tool config into settings.json when absent, so
    the tools are on out of the box and users can flip `enabled` per tool. Also
    prunes retired defaults (``_DROP_TOOLS``) from an existing config."""
    path = Path(home_dir) / "settings.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    changed = False
    if "tools" not in data:
        data["tools"] = dict(_DEFAULT_TOOLS)
        changed = True
    else:
        tools = data["tools"]
        # Migration: drop retired default tools (only when present) so an
        # already-seeded home stops loading a component the UI can't render.
        for name in _DROP_TOOLS:
            if name in tools:
                del tools[name]
                changed = True
        # Add newly-introduced default tools that this home predates (e.g.
        # code_executor), without overwriting a user's existing tool configs.
        for name, cfg in _DEFAULT_TOOLS.items():
            if name not in tools:
                tools[name] = cfg
                changed = True
        # Enforce "terminal = shell only": a code_executor without an explicit
        # include/exclude (i.e. still the un-customized default) is narrowed to
        # shell_executor. A user's own include/exclude is left untouched.
        ce = tools.get("code_executor")
        if isinstance(ce, dict) and "include" not in ce and "exclude" not in ce:
            ce["include"] = ["shell_executor"]
            changed = True
    if not changed:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
