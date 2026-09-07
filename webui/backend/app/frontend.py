"""Shared build validation; no SDK or web-server imports are needed here."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


class BuildError(RuntimeError):
    """The frontend must be built or repaired before it can be served."""


def source_files(frontend: Path) -> list[Path]:
    files = []
    for name in ("app", "scripts", "public"):
        directory = frontend / name
        for file in directory.rglob("*"):
            rel = file.relative_to(frontend).as_posix()
            if any(part.startswith(".") for part in file.relative_to(frontend).parts):
                continue
            if rel.startswith(("public/antd/", "public/assets/")):
                continue
            if file.is_file():
                files.append(file)
    files.extend(
        file
        for file in frontend.iterdir()
        if file.is_file() and re.search(r"\.(?:json|ya?ml|[cm]?js|ts)$", file.name)
    )
    return sorted(files)


def sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _local_file(frontend: Path, relative: str) -> Path:
    file = (frontend / relative).resolve()
    if not file.is_relative_to(frontend.resolve()) or not file.is_file():
        raise BuildError(f"Missing or invalid frontend file: {relative}")
    return file


def validate_build(frontend: Path, *, check_sources: bool = True) -> str:
    """Validate SSR/CSS and return the local CSS URL for HTTP readiness checks."""
    try:
        _local_file(frontend, "server.js")
        _local_file(frontend, "build/server/index.js")
        css_manifest = json.loads(
            _local_file(frontend, "build/client/antd/manifest.json").read_text()
        )
        href = css_manifest["href"]
        parsed = urlsplit(href)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or not href.startswith("/assets/")
            or not href.endswith(".css")
        ):
            raise BuildError("Invalid Ant Design CSS URL in the build manifest")
        _local_file(frontend / "build/client", href.lstrip("/"))
        build = json.loads(_local_file(frontend, "build/webui-build.json").read_text())
        if (
            build.get("format") != 1
            or not build.get("outputs")
            or not build.get("inputs")
        ):
            raise BuildError("Missing frontend build provenance")
        required = {
            "build/server/index.js",
            "build/client/antd/manifest.json",
            "build/client" + href,
        }
        if not required.issubset(build["outputs"]):
            raise BuildError("Incomplete frontend build manifest")
        for rel, digest in build["outputs"].items():
            if (
                not rel.startswith("build/")
                or sha256(_local_file(frontend, rel)) != digest
            ):
                raise BuildError(f"Frontend output changed or is incomplete: {rel}")
        if check_sources:
            current = {
                file.relative_to(frontend).as_posix(): sha256(file)
                for file in source_files(frontend)
            }
            if current != build["inputs"]:
                raise BuildError("Frontend sources changed since the last build")
        return href
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BuildError(f"Invalid or missing frontend build: {exc}") from exc
