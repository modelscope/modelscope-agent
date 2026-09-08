from fastapi import APIRouter

from app.core.envelope import EnvelopeRoute
from app.schemas.search import SearchProvider, SearchSettings, SearchSettingsUpdate

router = APIRouter(prefix="/api/search-settings", tags=["search-settings"],
                   route_class=EnvelopeRoute)


@router.get("/providers")
def list_providers() -> list[SearchProvider]:
    from app.backends.ms_agent import search

    return search.list_providers()


@router.get("")
def get_settings() -> SearchSettings:
    from app.backends.ms_agent import search

    return search.get_settings()


@router.put("")
def update_settings(body: SearchSettingsUpdate) -> SearchSettings:
    from app.backends.ms_agent import search

    return search.update_settings(body)
