"""Project-relative configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Raised when a project configuration value is unsafe or malformed."""


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML mapping from the repository's ``configs`` directory."""
    config_path = PROJECT_ROOT / "configs" / filename
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigurationError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a relative path and reject paths outside the repository root."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        raise ConfigurationError("Use a path relative to the repository root.")

    resolved = (PROJECT_ROOT / candidate).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ConfigurationError("Path must remain inside the repository root.")
    return resolved


def configured_path(paths_config: dict[str, Any], key: str) -> Path:
    """Return a validated project-relative path from ``paths.yaml``."""
    value = paths_config.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"Missing or invalid path configuration: {key}")
    return resolve_project_path(value)
