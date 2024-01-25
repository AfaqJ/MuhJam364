
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "Open the html code to start predictions"}

def test_invalid_endpoint():
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404

def test_complete_request_cycle():
    data = {"text": "bau fahrk"}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "predicted_label" in response.json()

def test_large_input():
    data = {"text": "a" * 10000}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "predicted_label" in response.json()
