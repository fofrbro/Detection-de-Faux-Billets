"""Tests de l'API HTTP."""

from fastapi.testclient import TestClient

from assistant import api


def test_health_base_disponible(monkeypatch) -> None:
    monkeypatch.setattr(api.db, "ping", lambda: True)
    reponse = TestClient(api.app).get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["database"] == "ok"


def test_health_base_indisponible(monkeypatch) -> None:
    monkeypatch.setattr(api.db, "ping", lambda: False)
    reponse = TestClient(api.app).get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["database"] == "indisponible"
