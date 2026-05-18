import json
import logging
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Importando seus módulos do pipeline
from src.preprocess import limpar_dados, preparar_features
from src.train import ChurnMLP

# 1. Configuração Profissional de Logging e Caminhos
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "modelo_churn.pt"

# 2. Instanciando o App com metadados para o Swagger
app = FastAPI(
    title="FIAPMobile Churn Prediction API",
    description=(
        "API para predição de cancelamento de clientes (Churn) "
        "utilizando Redes Neurais."
    ),
    version="1.0.0"
)

COLUMNS_PATH = ROOT_DIR / "models" / "columns.json"
with open(COLUMNS_PATH, "r") as f:
    expected_columns = json.load(f)

# 3. Carregamento Seguro do Modelo
INPUT_DIM = 30 # Dimensão validada no treinamento
model = ChurnMLP(input_dim=INPUT_DIM)

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logger.info(f"✅ Modelo Champion carregado com sucesso de: {MODEL_PATH}")
else:
    logger.error(f"❌ Erro Crítico: Arquivo de modelo não encontrado em {MODEL_PATH}")

# 4. Esquema Pydantic
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

# 5. Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(cliente: ClienteData):
    try:
        # Converter dados de entrada para DataFrame
        df_input = pd.DataFrame([cliente.model_dump()])
        
        # Aplicar o mesmo pré-processamento do treino (Reprodutibilidade!)
        df_clean = limpar_dados(df_input)
        X_processed = preparar_features(
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