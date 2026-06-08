import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pandera.errors import SchemaError
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from logging_config import logger
from src.schema import input_schema
from src.train import ChurnMLP


# 1. MIDDLEWARE DE LATÊNCIA E LOG ESTRUTURADO
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000

        if request.url.path not in ("/metrics", "/health"):
            logger.info(
                "request_completed",
                extra={
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                }
            )

        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time-Ms"] = str(round(latency_ms, 2))

        return response


# 2. APP
app = FastAPI(
    title="FIAPMobile Churn Prediction API",
    description=(
        "API para predição de cancelamento de clientes (Churn) "
        "utilizando Redes Neurais."
    ),
    version="1.0.0"
)

app.add_middleware(LoggingMiddleware)


# 3. CAMINHOS
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "modelo_churn.pt"
PIPELINE_PATH = ROOT_DIR / "models" / "pipeline.joblib"


def load_artifacts():
    """
    Carrega o pipeline sklearn e o modelo MLP do disco.

    O pipeline garante transformação idêntica à do treino (substitui columns.json).
    O INPUT_DIM é lido do próprio pipeline — sem hardcode.

    Retorna (pipeline, model, input_dim).
    """
    pipeline = joblib.load(PIPELINE_PATH)
    input_dim = len(pipeline.named_steps["encoder"].columns_)

    model = ChurnMLP(input_dim=input_dim)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        logger.info(f"✅ Modelo Champion carregado de: {MODEL_PATH}")
    else:
        logger.error(f"❌ Modelo não encontrado em {MODEL_PATH}")

    return pipeline, model, input_dim


# 4. CARREGAMENTO DE ARTEFATOS (executado uma vez na startup)
pipeline, model, INPUT_DIM = load_artifacts()


# 4. SCHEMA PYDANTIC
class ClienteData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: float = Field(..., example=29.85)


# 5. FUNÇÕES DE SUPORTE AO ENDPOINT

def validate_input(df_input):
    """
    Valida o DataFrame de entrada com o schema Pandera.

    Rejeita valores fora do domínio permitido (ex: gender fora de Male/Female)
    antes de qualquer transformação, retornando 422 com detalhes do erro.
    """
    try:
        input_schema.validate(df_input)
    except SchemaError as e:
        logger.warning(f"⚠️ Schema Pandera inválido: {e.failure_cases}")
        raise HTTPException(
            status_code=422,
            detail=f"Dados de entrada inválidos: {str(e.failure_cases.to_dict())}",
        )


def run_inference(df_input):
    """
    Executa o pipeline de inferência completo:
    DataFrame → pipeline sklearn → tensor PyTorch → modelo → resposta JSON.
    """
    X_processed = pipeline.transform(df_input)

    if X_processed.shape[1] != INPUT_DIM:
        logger.error(
            f"Erro de dimensão: {X_processed.shape[1]} colunas geradas, "
            f"esperadas {INPUT_DIM}"
        )
        raise HTTPException(
            status_code=422,
            detail="Dados de entrada geraram formato incompatível.",
        )

    input_tensor = torch.tensor(X_processed, dtype=torch.float32)
    with torch.no_grad():
        probabilidade = model(input_tensor).item()

    predicao = 1 if probabilidade > 0.5 else 0
    logger.info(f"🔮 Predição realizada: Prob={probabilidade:.4f} | Churn={predicao}")

    return {
        "churn_probabilidade": round(probabilidade, 4),
        "churn_predicao": predicao,
        "status": "sucesso",
    }


# 6. ENDPOINTS
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(cliente: ClienteData):
    try:
        df_input = pd.DataFrame([cliente.model_dump()])
        validate_input(df_input)
        return run_inference(df_input)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Falha na inferência: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar predição."
        )
