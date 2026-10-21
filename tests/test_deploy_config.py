"""Static checks on the deployment files (no Docker daemon, no weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent


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
    for env in (
        "AUTOASSESS_HOST=0.0.0.0",
        "AUTOASSESS_PORT=7860",
        "AUTOASSESS_DEVICE=cpu",
        "AUTOASSESS_STATIC_DIR=/app/frontend/dist",
        "AUTOASSESS_MAX_UPLOAD_MB=15",
    ):
        assert env in dockerfile
    assert "GRADIO" not in dockerfile.upper()
    assert "HEALTHCHECK" in dockerfile
    assert "urllib" in dockerfile
    assert "http://127.0.0.1:7860/api/health" in dockerfile
    assert 'CMD ["autoassess-web"]' in dockerfile


def test_dockerfile_has_both_stages(dockerfile: str) -> None:
    froms = [ln for ln in dockerfile.splitlines() if ln.startswith("FROM ")]
    assert froms[0] == "FROM node:22-slim AS web"
    assert froms[-1] == "FROM python:3.12-slim"
    assert "npm ci" in dockerfile and "npm run build" in dockerfile


def test_dockerfile_copies_dist_into_final_stage(dockerfile: str) -> None:
    final = dockerfile.split("FROM python:3.12-slim", 1)[1]
    assert "COPY --from=web /app/frontend/dist /app/frontend/dist" in final


def test_dockerfile_keeps_cpu_assertion(dockerfile: str) -> None:
    assert "'+cpu' in torch.__version__" in dockerfile


def test_dockerignore_excludes_heavy_dirs() -> None:
    lines = {
        ln.strip().rstrip("/")
        for ln in (_ROOT / ".dockerignore").read_text().splitlines()
    }
    assert {"runs", "data", ".git"} <= lines


def test_dockerignore_keeps_frontend_sources() -> None:
    lines = {
        ln.strip().rstrip("/")
        for ln in (_ROOT / ".dockerignore").read_text().splitlines()
    }
    assert "frontend" not in lines and "frontend/*" not in lines
    assert {"frontend/node_modules", "frontend/dist"} <= lines
