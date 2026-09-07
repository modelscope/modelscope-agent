from datetime import datetime

from pydantic import BaseModel, Field


class Profile(BaseModel):
    # "" = not set; the old "User" default was schema boilerplate that the
    # model never saw anyway (the field used to be a dead sidecar entry).
    agent_calls_user: str = ""
    description: str = ""
    updated_at: datetime


class ProfileUpsert(BaseModel):
    agent_calls_user: str | None = Field(default=None, max_length=80)
    description: str | None = None
