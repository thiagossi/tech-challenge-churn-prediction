from unittest.mock import patch

import pandas as pd
import pandera as pa
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.schema import input_schema

client = TestClient(app)

# Payload válido reutilizado nos testes de API e Pandera
PAYLOAD_VALIDO = {
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
    "TotalCharges": 29.85,
}


# ---------------------------------------------------------
# 1. SMOKE TEST
# Objetivo: verificar se a aplicação e o modelo carregam
# sem crashar.
# ---------------------------------------------------------
def test_smoke_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


# ---------------------------------------------------------
# 2. SCHEMA TEST — Pydantic (contrato HTTP)
# Objetivo: garantir que o Pydantic rejeita campos faltando
# com erro 422.
# ---------------------------------------------------------
def test_schema_pydantic_payload_incompleto():
    payload_incompleto = {"tenure": 10}
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422
    assert "detail" in response.json()


# ---------------------------------------------------------
# 3. SCHEMA TEST — Pandera (validação do DataFrame)
# Objetivo: garantir que valores inválidos que passam no
# Pydantic (tipos corretos, mas domínio errado) são
# rejeitados ao nível do DataFrame.
# ---------------------------------------------------------
def test_schema_pandera_dataframe_valido():
    """Um DataFrame construído a partir do payload válido deve passar no schema."""
    df = pd.DataFrame([PAYLOAD_VALIDO])
    # Não deve lançar nenhuma exceção
    input_schema.validate(df)


def test_schema_pandera_tenure_negativo():
    """tenure negativo é um valor de domínio inválido — deve ser rejeitado."""
    payload_invalido = {**PAYLOAD_VALIDO, "tenure": -1}
    df = pd.DataFrame([payload_invalido])
    with pytest.raises(pa.errors.SchemaError):
        input_schema.validate(df)


def test_schema_pandera_senior_citizen_invalido():
    """SeniorCitizen só aceita 0 ou 1 — qualquer outro valor deve ser rejeitado."""
    payload_invalido = {**PAYLOAD_VALIDO, "SeniorCitizen": 99}
    df = pd.DataFrame([payload_invalido])
    with pytest.raises(pa.errors.SchemaError):
        input_schema.validate(df)


def test_schema_pandera_monthly_charges_negativo():
    """MonthlyCharges negativo viola a regra de negócio — deve ser rejeitado."""
    payload_invalido = {**PAYLOAD_VALIDO, "MonthlyCharges": -10.0}
    df = pd.DataFrame([payload_invalido])
    with pytest.raises(pa.errors.SchemaError):
        input_schema.validate(df)


# ---------------------------------------------------------
# 4. API TEST (Teste Funcional)
# Objetivo: simular uma requisição real completa e validar
# o formato da resposta JSON.
# ---------------------------------------------------------
def test_predict_functional_success():
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probabilidade" in data
    assert "churn_predicao" in data
    assert data["status"] == "sucesso"
    assert 0 <= data["churn_probabilidade"] <= 1


# ---------------------------------------------------------
# 5. PANDERA VIA API
# Objetivo: garantir que validate_input rejeita valores
# inválidos de domínio com 422 quando chamado via endpoint.
# ---------------------------------------------------------
def test_predict_pandera_rejeita_gender_invalido_via_api():
    """gender fora do domínio deve retornar 422 via API."""
    payload = {**PAYLOAD_VALIDO, "gender": "Outro"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Dados de entrada inválidos" in response.json()["detail"]


def test_predict_pandera_rejeita_contrato_invalido_via_api():
    """Contract fora do domínio deve retornar 422 via API."""
    payload = {**PAYLOAD_VALIDO, "Contract": "contrato_inexistente"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------
# 6. EXCEPTION HANDLER
# Objetivo: garantir que exceções inesperadas retornam 500.
# ---------------------------------------------------------
def test_predict_exception_handler_retorna_500():
    """Exceção inesperada no pipeline deve retornar 500."""
    with patch("src.main.run_inference", side_effect=RuntimeError("falha inesperada")):
        response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 500
    assert "Erro interno" in response.json()["detail"]


def test_run_inference_dimensao_errada_retorna_422():
    """Pipeline gerando shape errado deve retornar 422."""
    import numpy as np
    with patch("src.main.pipeline.transform", return_value=np.zeros((1, 1))):
        response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 422
    assert "incompatível" in response.json()["detail"]
