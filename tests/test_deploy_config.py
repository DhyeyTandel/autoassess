"""Static checks on the deployment files (no Docker daemon, no weights)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
NOTICE = (
    "Research demo, non-commercial. Models trained on CarDD, VehiDE and Carparts-Seg. "
    "Uploaded photos are deleted from the server within about 20 minutes."
)


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    data = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
    service: dict[str, Any] = data["services"]["autoassess"]
    return service


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (_ROOT / "Dockerfile").read_text()


def test_compose_port_bound_to_loopback(compose: dict[str, Any]) -> None:
    assert "127.0.0.1:7860:7860" in [str(p) for p in compose["ports"]]


def test_compose_runs_mount_read_only(compose: dict[str, Any]) -> None:
    assert "./runs:/app/runs:ro" in compose["volumes"]


def test_compose_limits_and_restart(compose: dict[str, Any]) -> None:
    assert compose["restart"] == "unless-stopped"
    assert float(compose["cpus"]) == 4
    assert compose["mem_limit"] == "3g"
    assert compose["image"] == "autoassess:latest"
    assert compose["container_name"] == "autoassess"


def test_dockerfile_uses_cpu_torch_index(dockerfile: str) -> None:
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "python:3.12-slim" in dockerfile


def test_dockerfile_non_root_user(dockerfile: str) -> None:
    users = [ln.split()[1] for ln in dockerfile.splitlines() if ln.startswith("USER ")]
    assert users and users[-1] not in {"root", "0"}


def test_dockerfile_env_and_healthcheck(dockerfile: str) -> None:
    assert "GRADIO_SERVER_NAME=0.0.0.0" in dockerfile
    assert "GRADIO_SERVER_PORT=7860" in dockerfile
    assert "AUTOASSESS_DEVICE=cpu" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "urllib" in dockerfile


def test_dockerignore_excludes_heavy_dirs() -> None:
    lines = {
        ln.strip().rstrip("/")
        for ln in (_ROOT / ".dockerignore").read_text().splitlines()
    }
    assert {"runs", "data", ".git"} <= lines


def test_app_shows_notice() -> None:
    spec = importlib.util.spec_from_file_location("autoassess_deploy_app", _ROOT / "app.py")
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert NOTICE in module.NOTICE_HTML
    values = [str(getattr(b, "value", "")) for b in module.demo.blocks.values()]
    assert any(NOTICE in v for v in values)
    assert "deleted" in module.NOTICE_HTML
    assert module.demo.delete_cache == (600, 600)
    assert module.CACHE_SWEEP == (600, 600)
