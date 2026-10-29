from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_health_check_requires_no_auth():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_protected_route_without_token_is_401():
    resp = client.get("/postings")
    assert resp.status_code == 401


def test_protected_route_with_wrong_token_is_401():
    resp = client.get("/postings", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_protected_route_with_correct_token_is_200():
    resp = client.get("/postings", headers={"Authorization": f"Bearer {settings.api_token}"})
    assert resp.status_code == 200


def test_settings_read_the_repo_root_env_file_regardless_of_working_directory():
    from pathlib import Path

    from app.config import Settings

    env_file = Settings.model_config["env_file"]
    assert Path(env_file).is_absolute()
    assert Path(env_file) == Path(__file__).resolve().parents[1] / ".env"
