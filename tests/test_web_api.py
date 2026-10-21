"""Tests for autoassess.web.api via TestClient, with a fake Assessor (no weights)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from autoassess.web import api, service


class FakeAssessor:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], str]] = []

    def assess(self, image: Image.Image, sensitivity: str) -> service.AssessResult:
        self.calls.append((image.size, sensitivity))
        result: dict[str, Any] = {
            "image_width": image.size[0],
            "image_height": image.size[1],
            "instances": [
                {
                    "part": "door", "damage_type": "dent", "severity": "minor",
                    "mask_area_px": 120.0, "confidence": 0.71234,
                    "part_coverage": 0.9, "source_model": "yolov8_seg_v1",
                }
            ],
            "triage": {
                "decision": "auto_approve", "total_damaged_area_px": 120.0,
                "n_instances": 1, "n_severe": 0, "n_moderate": 0, "n_minor": 1,
            },
        }
        return service.AssessResult(
            result=result,
            overlay=Image.new("RGB", image.size, "blue"),
            secondary_conf=0.07 if sensitivity == "high_recall" else 0.15,
        )


@pytest.fixture
def fake() -> FakeAssessor:
    return FakeAssessor()


@pytest.fixture
def client(fake: FakeAssessor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("AUTOASSESS_STATIC_DIR", str(tmp_path / "missing"))
    return TestClient(api.create_app(assessor=fake))


def _png(size: tuple[int, int] = (20, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "green").save(buf, format="PNG")
    return buf.getvalue()


def test_assess_ok_contract(client: TestClient, fake: FakeAssessor) -> None:
    r = client.post("/api/assess", files={"image": ("a.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {
        "triage", "instances", "panels", "unassigned", "overlay",
        "image", "sensitivity", "secondary_conf", "elapsed_ms",
    }
    assert body["triage"]["label"] == "Auto-approve"
    assert body["image"] == {"width": 20, "height": 10}
    assert body["sensitivity"] == "high_recall"
    assert body["secondary_conf"] == 0.07
    assert body["instances"][0]["confidence"] == 0.7123
    assert body["overlay"].startswith("data:image/jpeg;base64,")
    assert fake.calls == [((20, 10), "high_recall")]


def test_assess_standard_sensitivity(client: TestClient) -> None:
    r = client.post(
        "/api/assess",
        files={"image": ("a.png", _png(), "image/png")},
        data={"sensitivity": "standard"},
    )
    assert r.status_code == 200
    assert r.json()["secondary_conf"] == 0.15


def test_bad_sensitivity_422(client: TestClient) -> None:
    r = client.post(
        "/api/assess",
        files={"image": ("a.png", _png(), "image/png")},
        data={"sensitivity": "High recall"},
    )
    assert r.status_code == 422


def test_text_file_400(client: TestClient) -> None:
    r = client.post("/api/assess", files={"image": ("a.jpg", b"just text", "image/jpeg")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_missing_image_422(client: TestClient) -> None:
    assert client.post("/api/assess").status_code == 422


def test_oversize_413(
    fake: FakeAssessor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOASSESS_STATIC_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("AUTOASSESS_MAX_UPLOAD_MB", "0.0001")  # ~104 bytes
    c = TestClient(api.create_app(assessor=fake))
    r = c.post("/api/assess", files={"image": ("a.png", _png((200, 200)), "image/png")})
    assert r.status_code == 413
    assert "detail" in r.json()
    assert fake.calls == []


def test_heic_accepted(client: TestClient) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    try:
        Image.new("RGB", (32, 32), "red").save(buf, format="HEIF")
    except Exception as exc:  # encoder missing in this build
        pytest.skip(f"cannot encode HEIC here: {exc}")
    r = client.post("/api/assess", files={"image": ("a.heic", buf.getvalue(), "image/heic")})
    assert r.status_code == 200, r.text
    assert r.json()["image"] == {"width": 32, "height": 32}


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "busy": False}


def test_root_dev_mode_json(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "note" in r.json()


def test_static_dir_served(
    fake: FakeAssessor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<html>hello-spa</html>")
    monkeypatch.setenv("AUTOASSESS_STATIC_DIR", str(static))
    c = TestClient(api.create_app(assessor=fake))
    assert "hello-spa" in c.get("/").text
    assert c.get("/api/health").json()["status"] == "ok"  # API routes still win


def test_busy_true_during_assessment(
    fake: FakeAssessor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOASSESS_STATIC_DIR", str(tmp_path / "missing"))
    seen: list[bool] = []
    app = api.create_app(assessor=fake)
    original = fake.assess

    def spy(image: Image.Image, sensitivity: str) -> service.AssessResult:
        seen.append(app.state.lock.locked())
        return original(image, sensitivity)

    monkeypatch.setattr(fake, "assess", spy)
    c = TestClient(app)
    assert c.post("/api/assess", files={"image": ("a.png", _png(), "image/png")}).status_code == 200
    assert seen == [True]
    assert c.get("/api/health").json()["busy"] is False
