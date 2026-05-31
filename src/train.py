import random
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from logging_config import setup_logging
from preprocess import build_pipeline, clean_data, load_data

# Define o nome do experimento para o MLflow
MLFLOW_EXPERIMENT = "FIAPMobile_Churn_MLP"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Configuração de caminhos dinâmicos
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
DATA_PATH_PROCESSED = (
    ROOT_DIR / "data" / "processed" / "telco_customer_churn_cleaned.csv"
)
MODEL_DIR = ROOT_DIR / "models"
MODEL_FILE = MODEL_DIR / "modelo_churn.pt"
PIPELINE_FILE = MODEL_DIR / "pipeline.joblib"


# FIXANDO SEEDS PARA REPRODUTIBILIDADE
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


logger = setup_logging()


# Definição da Classe do Modelo
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


def execute_training():
    set_seeds()
    logger.info("🚀 Iniciando Pipeline de Treinamento...")

    # Definição de Hiperparâmetros para registro no MLflow
    params = {
        "input_dim": 30,
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
        "early_stopping_patience": 10,
    }

    # Início da jornada de governança no MLflow
    with mlflow.start_run(run_name="MLP_Champion_Training"):
        mlflow.set_tag("model_type", "neural_network")
        mlflow.log_params(params)

        try:
            # 1. Carregamento e limpeza dos dados brutos
            df_raw = load_data(DATA_PATH)
            df_clean = clean_data(df_raw)
            df_clean.to_csv(DATA_PATH_PROCESSED, index=False)

            y = df_clean["Churn"].values
            # X como DataFrame — necessário para o Pipeline sklearn
            X_df = df_clean.drop(columns=["Churn"])

            # 2. DIVISÃO TREINO/TESTE no DataFrame bruto (antes do encoding)
            # Dividir antes do fit() do pipeline evita Data Leakage:
            # o FeatureEncoder aprende as colunas SÓ do conjunto de treino.
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_df, y, test_size=0.2, random_state=42, stratify=y
            )

            # 3. PIPELINE SKLEARN — fit apenas no treino, transform em ambos
            pipeline = build_pipeline()
            X_train = pipeline.fit_transform(X_train_df)
            X_test = pipeline.transform(X_test_df)

            # Persiste o pipeline para ser carregado pela API (substitui columns.json)
            joblib.dump(pipeline, PIPELINE_FILE)
            logger.info(f"✅ Pipeline salvo em: {PIPELINE_FILE}")

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

            # LÓGICA DE EARLY STOPPING
            best_loss = float("inf")
            counter = 0

            logger.info(
                f"🧠 Iniciando treino (Paciência: {params['early_stopping_patience']})"
            )

            for epoch in range(params["epochs"]):
                model.train()
                epoch_loss = 0.0

                for inputs, labels in loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / len(loader)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= params["early_stopping_patience"]:
                    logger.info(f"🛑 Parada antecipada acionada na época {epoch + 1}")
                    mlflow.set_tag("stop_reason", "early_stopping")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Época [{epoch + 1}/{params['epochs']}], "
                        f"Loss: {avg_train_loss:.4f}"
                    )

            # 6. Avaliação Final no Conjunto de Teste (Holdout)
            model.eval()
            with torch.no_grad():
                outputs_test = model(X_test_tensor)
                y_proba = outputs_test.numpy().flatten()
                y_pred = (y_proba > 0.5).astype(float)

            # Cálculo de Métricas Técnicas (≥ 4 métricas — exigência do Tech Challenge)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "pr_auc": average_precision_score(y_test, y_proba),
            }

            # 7. REGISTRO DE GOVERNANÇA (MLflow)
            mlflow.log_metrics(metrics)

            # Assinatura garante o contrato de dados no MLflow Model Registry
            signature = infer_signature(X_test_tensor[:1].numpy(), y_pred[:1])
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
            )

            # 8. PERSISTÊNCIA PARA DEPLOY (Disco Local)
            torch.save(model.state_dict(), MODEL_FILE)

            logger.info(
                f"✅ Treinamento concluído. "
                f"Recall={metrics['recall']:.4f} | "
                f"AUC-ROC={metrics['roc_auc']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )
            logger.info(f"📦 Modelo persistido localmente em: {MODEL_FILE}")

        except Exception as e:
            logger.error(f"❌ Falha crítica no pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    execute_training()
