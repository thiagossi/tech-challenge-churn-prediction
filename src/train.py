import json
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from logging_config import setup_logging
from preprocess import clean_data, load_data, prepare_features

# Define o nome do experimento para o MLflow
mlflow.set_experiment("FIAPMobile_Churn_MLP")

# Configuração de caminhos dinâmicos
# __file__ é o caminho deste arquivo (train.py)
# .parent volta para 'src'
# .parent volta para a raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
DATA_PATH_PROCESSED = (
    ROOT_DIR / "data" / "processed" / "telco_customer_churn_cleaned.csv"
)
MODEL_DIR = ROOT_DIR / "models"
MODEL_FILE = MODEL_DIR / "modelo_churn.pt"

# 1. FIXANDO SEEDS PARA REPRODUTIBILIDADE
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = setup_logging() 

# 1. Definição da Classe do Modelo
class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def executar_treinamento():
    set_seeds() # Garante reprodutibilidade conforme requisito
    logger.info("🚀 Iniciando Pipeline de Treinamento...")

    # Definição de Hiperparâmetros para registro no MLflow
    params = {
        "input_dim": 30,
        "epochs": 50,
        "batch_size": 32,
        "lr": 0.001
    }

    # Início da jornada de governança no MLflow
    with mlflow.start_run(run_name="MLP_Champion_Training"):
        # 1. Registrar Parâmetros de Treino
        mlflow.log_params(params)

        try:
            # 1. Carregamento dos dados brutos
            df_raw = load_data(DATA_PATH)
            
            # 2. Limpeza e preparação dos dados
            df_clean = clean_data(df_raw)
            # Salva o dataset limpo 
            df_clean.to_csv(DATA_PATH_PROCESSED, index=False) 
            # Preparação das features (One-Hot Encoding, Normalização, etc.)
            X = prepare_features(df_clean)
            y = df_clean['Churn'].values

            # Salva o contrato de colunas para garantir que a API 
            # receba o mesmo formato de dados do treinamento
            feature_columns = X.columns.tolist()
            with open(ROOT_DIR / "models" / "columns.json", "w") as f:
                json.dump(feature_columns, f)
            logger.info("✅ Contrato de colunas salvo para a API.")
            
            # 3. DIVISÃO TREINO/TESTE
            # Importante: Separação de dados ANTES de criar os tensores 
            # para evitar Data Leakage
            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y, test_size=0.2, random_state=42, stratify=y
            )

            # 4. Preparação dos Tensores PyTorch
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)            
            
            # DataLoaders para eficiência no treino
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
            
            # 5. Instanciando e Treinando o Modelo MLP
            input_size = X_train.shape[1] 
            model = ChurnMLP(input_dim=input_size)    
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=params["lr"])
            
            logger.info(f"🧠 Treinando Rede Neural com {input_size} entradas...")
            model.train()
            for epoch in range(params["epochs"]):
                for inputs, labels in loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # 6. Avaliação Final no Conjunto de Teste (Holdout)
            model.eval()
            with torch.no_grad():
                outputs_test = model(X_test_tensor)
                y_pred = (outputs_test > 0.5).float().numpy()

            # Cálculo de Métricas Técnicas (Exigência: ≥ 4 métricas)
            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred) # Prioritária para Churn
            f1 = f1_score(y_test, y_pred)
            
            # 7. REGISTRO DE GOVERNANÇA (MLflow)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)            
           
            # Criamos a assinatura para garantir o contrato de dados
            # Isso mapeia que sua rede espera 30 colunas de entrada
            signature = infer_signature(X_test_tensor[:1].numpy(), y_pred[:1])

            # Registramos o modelo de forma estável e segura
            mlflow.pytorch.log_model(
                pytorch_model=model, 
                artifact_path="model",
                signature=signature
            )
            
            # 8. PERSISTÊNCIA PARA DEPLOY (Disco Local)
            # Salva o arquivo que a API FastAPI irá carregar no main.py
            torch.save(model.state_dict(), MODEL_FILE)
            
            logger.info(
                f"✅ Treinamento concluído. "
                f"Metrics: Recall={rec:.4f}, F1={f1:.4f}"
            )
            logger.info(f"📦 Modelo persistido localmente em: {MODEL_FILE}")

        except Exception as e:
            logger.error(f"❌ Falha crítica no pipeline: {str(e)}")
            raise e # Garante que o MLflow registre a falha da Run

if __name__ == "__main__":
    executar_treinamento()