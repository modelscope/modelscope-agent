from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    agent_settings,
    chat,
    instructions,
    mcps,
    memory,
    models,
    presence,
    profile,
    projects,
    providers,
    search,
    sessions,
    skills,
    workspace,
)
from app.core.envelope import register_exception_handlers
from app.core.settings import settings


def create_app() -> FastAPI:
    app = FastAPI(title="ms-agent-webui backend", version="0.2.0")

    # Uniform envelope-shaped errors for every route (HTTP / validation / crash).
    register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure the SDK home + default project + settings.json llm are ready.
    from app.backends.ms_agent.bootstrap import bootstrap

    bootstrap()

    # HOME semantics, pinned (docs and code agree from here on): the default is
    # the SHARED ~/.ms_agent — the same home CLI/TUI use; isolation is an
    # explicit choice via MS_AGENT_HOME / ms_agent_home in backend/.env. Print
    # the effective home once so "which home am I on" is never guesswork.
    # Runs in the startup hook (not at import) so uvicorn's logging is wired;
    # uvicorn.error is the logger whose handlers reach the dev console.
    @app.on_event("startup")
    async def _log_effective_home() -> None:
        import logging
        import os

        from app.backends.ms_agent.common import home

        effective = home()
        shared_default = os.path.expanduser("~/.ms_agent")
        logging.getLogger("uvicorn.error").info(
            "ms_agent home: %s (%s)", effective,
            "shared default" if effective == shared_default else
            "isolated via MS_AGENT_HOME")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        from app.backends.ms_agent.runtime import registry
        from app.backends.ms_agent.skill_index import skill_index

        await registry.close_all()
        skill_index.stop()

    app.include_router(chat.router)
    app.include_router(presence.router)
    app.include_router(projects.router)
    app.include_router(sessions.router)
    app.include_router(mcps.router)
    app.include_router(skills.router)
    app.include_router(memory.router)
    app.include_router(instructions.router)
    app.include_router(providers.router)
    app.include_router(models.router)
    app.include_router(agent_settings.router)
    app.include_router(search.router)
    app.include_router(profile.router)
    app.include_router(workspace.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


def _run(reload: bool) -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        reload_dirs=["app"] if reload else None,
        reload_includes=["*.py", "*.env"] if reload else None,
    )


def dev() -> None:
    """Run with hot reload."""
    _run(reload=True)


def serve() -> None:
    """Run without reload (production)."""
    _run(reload=False)


if __name__ == "__main__":
    dev()
