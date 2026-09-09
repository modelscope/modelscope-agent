"""Backend-domain errors that map straight to HTTP status codes.

Subclassing HTTPException lets adapters raise semantic errors while FastAPI
renders them natively — routes stay thin and don't repeat status mapping.
"""
from __future__ import annotations

from fastapi import HTTPException


class NotFound(HTTPException):
    def __init__(self, detail: str = "The requested item was not found.") -> None:
        super().__init__(404, detail)


class BadRequest(HTTPException):
    def __init__(self, detail: str = "The request could not be processed.") -> None:
        super().__init__(400, detail)


class Conflict(HTTPException):
    def __init__(self, detail: str = "This action conflicts with the current state.") -> None:
        super().__init__(409, detail)
