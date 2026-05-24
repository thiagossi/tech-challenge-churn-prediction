from fastapi.testclient import TestClient

from src.main import app

# Instanciando o cliente de teste do FastAPI
client = TestClient(app)

# ---------------------------------------------------------
# 1. SMOKE TEST (Teste de Fumaça)
# Objetivo: Verificar se a aplicação e o modelo carregam 
# sem crashar.
# ---------------------------------------------------------
def test_smoke_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

# ---------------------------------------------------------
# 2. SCHEMA TEST (Teste de Contrato)
# Objetivo: Garantir que o Pydantic rejeita dados malformados 
# com erro 422.
# ---------------------------------------------------------
def test_schema_invalid_payload():
    # Payload incompleto (faltando quase todos os campos obrigatórios)
    payload_incompleto = {"tenure": 10}
    response = client.post("/predict", json=payload_incompleto)
    
    # Deve retornar 422 Unprocessable Entity
    assert response.status_code == 422
    assert "detail" in response.json()

# ---------------------------------------------------------
# 3. API TEST (Teste Funcional de Predição)
# Objetivo: Simular uma requisição real completa e validar 
# o formato da resposta JSON.
# ---------------------------------------------------------
def test_predict_functional_success():
    payload_valido = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    response = client.post("/predict", json=payload_valido)
    
    assert response.status_code == 200
    data = response.json()
    assert "churn_probabilidade" in data
    assert "churn_predicao" in data
    assert data["status"] == "sucesso"
    # Garante que a probabilidade é um float entre 0 e 1
    assert 0 <= data["churn_probabilidade"] <= 1