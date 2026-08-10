"""Version-aware JSON persistence for RayPath SCPT project documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import PROJECT_SCHEMA_VERSION, SUPPORTED_PROJECT_SCHEMA_VERSIONS


PROJECT_FORMAT = "RayPath SCPT Project"


@dataclass(frozen=True)
class ProjectDocument:
    """A validated project payload with its resolved schema version."""

    path: Path
    schema_version: int
    payload: dict[str, Any]


def project_schema_version(payload: Mapping[str, Any]) -> int:
    """Resolve and validate the explicit or legacy project schema field."""

    try:
        version = int(payload.get("schema_version", payload.get("version", 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError("The RayPath project schema version is invalid.") from exc
    if payload.get("format") != PROJECT_FORMAT or version not in SUPPORTED_PROJECT_SCHEMA_VERSIONS:
        raise ValueError("This is not a supported RayPath SCPT project file.")
    return version


def read_project_file(path: str | Path) -> ProjectDocument:
    """Read a UTF-8 JSON project and validate its identity and schema."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A RayPath SCPT project must contain one JSON object.")
    version = project_schema_version(payload)
    return ProjectDocument(path=source, schema_version=version, payload=payload)


def write_project_file(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Validate and write a current-schema project using deterministic JSON."""

    target = Path(path)
    version = project_schema_version(payload)
    if version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"New project saves must use schema {PROJECT_SCHEMA_VERSION}; received schema {version}."
        )
    target.write_text(json.dumps(dict(payload), indent=2, allow_nan=False), encoding="utf-8")
