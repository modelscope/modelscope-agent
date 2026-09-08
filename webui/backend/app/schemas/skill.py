from datetime import datetime

from pydantic import BaseModel, Field


class Skill(BaseModel):
    id: str
    name: str
    kind: str = "file-type"
    content: str = ""
    enabled: bool = True
    scope: str
    created_at: datetime
    # managed: copied into ms-agent's skills tree; legacy-path: an older
    # skills.json reference; standard: discovered from .agents/skills; content:
    # old sidecar-only row.  The UI uses this to describe deletion accurately.
    origin: str = "content"
    removable: bool = True


class SkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: str = "file-type"
    content: str = ""
    enabled: bool = True
    scope: str
    # Bundle imports only. False (default) rejects a name that already exists in
    # the scope; True replaces that skill's directory. Importing used to silently
    # write a second `<name>-2` directory, which the registry then shadowed — the
    # caller saw success while nothing changed and a stray copy accumulated on
    # disk. Callers must now choose explicitly.
    overwrite: bool = False


class SkillPathImport(BaseModel):
    path: str = Field(min_length=1)
    scope: str
    overwrite: bool = False


class SkillUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    kind: str | None = None
    content: str | None = None
    enabled: bool | None = None


class SkillFile(BaseModel):
    """One file inside a skill's on-disk directory (relative path)."""

    path: str
    size: int | None = None


class SkillFileContent(BaseModel):
    path: str
    # None ⇒ the file is binary / not valid UTF-8 (frontend shows a notice).
    content: str | None = None
