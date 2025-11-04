import pytest
from fastapi.testclient import TestClient


def test_basic_functionality():
    """Базовый тест для проверки работы приложения"""
    try:
        from app.main import app

        client = TestClient(app)

        # Тест health endpoint
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Тест главной страницы
        response = client.get("/")
        assert response.status_code == 200

    except Exception as e:
        pytest.fail(f"Базовые тесты не пройдены: {e}")


def test_ask_endpoint_structure():
    """Тест структуры ask endpoint"""
    try:
        from app.main import app

        client = TestClient(app)

        # Тест валидации запроса
        response = client.post("/api/ask", json={})
        assert response.status_code == 422  # Validation error

    except Exception as e:
        pytest.fail(f"Тест структуры ask endpoint не пройден: {e}")