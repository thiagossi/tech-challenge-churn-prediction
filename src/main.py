import json
import time
import uuid
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Importando o logger configurado
from logging_config import logger

# Importando seus módulos do pipeline
from src.preprocess import clean_data, prepare_features
from src.train import ChurnMLP


# 1. DEFINIÇÃO DO MIDDLEWARE DE LATÊNCIA E LOG ESTRUTURADO
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Registra o log estruturado em JSON para monitoramento (Aula 05/07)
        if request.url.path != "/metrics" and request.url.path != "/health":
            logger.info(
                "request_completed",
                extra={
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2)
                }
            )

        # Injeta metadados nos headers para o cliente (Frontend/CRM)
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time-Ms"] = str(round(latency_ms, 2))
        
        return response

# 2. Instanciando o App com metadados para o Swagger
app = FastAPI(
    title="FIAPMobile Churn Prediction API",
    description=(
        "API para predição de cancelamento de clientes (Churn) "
        "utilizando Redes Neurais."
    ),
    version="1.0.0"
)

# 3. ATIVANDO O MIDDLEWARE
app.add_middleware(LoggingMiddleware)


# 4. CONFIGURAÇÃO DE CAMINHOS E CARREGAMENTO DO MODELO
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "modelo_churn.pt"
COLUMNS_PATH = ROOT_DIR / "models" / "columns.json"

# Carregamento do contrato de colunas
with open(COLUMNS_PATH, "r") as f:
    expected_columns = json.load(f)

# Carregamento seguro do modelo PyTorch
INPUT_DIM = 30 # Dimensão validada no treinamento
model = ChurnMLP(input_dim=INPUT_DIM)

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logger.info(f"✅ Modelo Champion carregado com sucesso de: {MODEL_PATH}")
else:
    logger.error(f"❌ Erro Crítico: Arquivo de modelo não encontrado em {MODEL_PATH}")

# 5. Esquema Pydantic
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

# 6. Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(cliente: ClienteData):
    try:
        # Converter dados de entrada para DataFrame
        df_input = pd.DataFrame([cliente.model_dump()])
        
        # Aplicar o mesmo pré-processamento do treino (Reprodutibilidade!)
        df_clean = clean_data(df_input)
        X_processed = prepare_features(
            df_clean,
            columns_list=expected_columns
        )
        
        # Validação de Schema: Verifica se o processamento gerou as 30 colunas        
        if X_processed.shape[1] != INPUT_DIM:
            logger.error(
                f"Erro de Schema: Geradas {X_processed.shape[1]} colunas, "
                f"esperadas {INPUT_DIM}"
            )
            raise HTTPException(
                status_code=422, 
                detail="Dados de entrada geraram formato incompatível."
            )

        # Inferência com PyTorch
        input_tensor = torch.tensor(X_processed.values, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            probabilidade = output.item()
            predicao = 1 if probabilidade > 0.5 else 0            
        
        logger.info(
            f"🔮 Predição realizada: Prob={probabilidade:.4f} | Churn={predicao}"
        )
        
        return {
            "churn_probabilidade": round(probabilidade, 4),
            "churn_predicao": predicao,
            "status": "sucesso"
        }

    except Exception as e:
        logger.error(f"💥 Falha na inferência: {str(e)}")        
        raise HTTPException(
            status_code=500, 
            detail="Erro interno ao processar predição."
        )