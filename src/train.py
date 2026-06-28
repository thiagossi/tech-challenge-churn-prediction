import hashlib
import os
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

# Configuração de caminhos dinâmicos
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
DATA_PATH_PROCESSED = (
    ROOT_DIR / "data" / "processed" / "telco_customer_churn_cleaned.csv"
)
MODEL_DIR = ROOT_DIR / "models"
MODEL_FILE = MODEL_DIR / "modelo_churn.pt"
PIPELINE_FILE = MODEL_DIR / "pipeline.joblib"

# Define o nome do experimento para o MLflow
MLFLOW_EXPERIMENT = "FIAPMobile_Churn_MLP"
os.environ["MLFLOW_ARTIFACT_ROOT"] = str(ROOT_DIR / "mlruns")
mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/mlruns.db")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

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


def load_and_version_dataset():
    """
    Carrega e limpa o dataset, persiste a versão processada e
    registra a versão + metadados no MLflow.

    Retorna o DataFrame limpo pronto para split e pipeline.
    """
    df_raw = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(DATA_PATH_PROCESSED, index=False)

    dataset_hash = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()[:8]
    dataset_version = f"telco_churn_{dataset_hash}"

    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("dataset_rows", len(df_raw))
    mlflow.log_param("dataset_cols", len(df_raw.columns))

    mlflow_dataset = mlflow.data.from_pandas(
        df_clean,
        source=str(DATA_PATH),
        name=dataset_version,
        targets="Churn",
    )
    mlflow.log_input(mlflow_dataset, context="training")
    logger.info(f"✅ Dataset version registrada: {dataset_version}")

    return df_clean


def prepare_splits(df_clean):
    """
    Divide o dataset em treino/teste, aplica o Pipeline sklearn e persiste o pipeline.

    O split acontece ANTES do fit() do pipeline para evitar Data Leakage:
    o FeatureEncoder aprende as colunas apenas com dados de treino.

    Retorna X_train, X_test (numpy arrays) e y_train, y_test (arrays).
    """
    y = df_clean["Churn"].values
    X_df = df_clean.drop(columns=["Churn"])

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    X_train = pipeline.fit_transform(X_train_df)
    X_test = pipeline.transform(X_test_df)

    joblib.dump(pipeline, PIPELINE_FILE)
    logger.info(f"✅ Pipeline salvo em: {PIPELINE_FILE}")

    return X_train, X_test, y_train, y_test


def train_mlp(model, loader, params):
    """
    Executa o loop de treino com early stopping.

    Loga a loss por época no MLflow.
    Retorna o modelo treinado.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    best_loss = float("inf")
    counter = 0

    logger.info(f"🧠 Iniciando treino (Paciência: {params['early_stopping_patience']})")

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
            mlflow.log_param("early_stopping_epoch", epoch + 1)
            break

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Época [{epoch + 1}/{params['epochs']}], "
                f"Loss: {avg_train_loss:.4f}"
            )

    return model


def evaluate_model(model, X_test_tensor, y_test):
    """
    Avalia o modelo no conjunto de teste e retorna as métricas.

    Usa threshold 0.5 para converter probabilidade em classe binária.
    Retorna dict com accuracy, recall, precision, f1, roc_auc e pr_auc.
    """
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        y_proba = outputs_test.numpy().flatten()
        y_pred = (y_proba > 0.5).astype(float)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    logger.info(
        f"✅ Avaliação concluída. "
        f"Recall={metrics['recall']:.4f} | "
        f"AUC-ROC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )

    return metrics, y_pred, y_proba


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
            # 1. Carregamento, limpeza e versionamento do dataset
            df_clean = load_and_version_dataset()

            # 2. Split treino/teste + Pipeline sklearn
            X_train, X_test, y_train, y_test = prepare_splits(df_clean)

            # 3. Preparação dos Tensores PyTorch e DataLoader
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

            # 4. Treino com early stopping
            input_size = X_train.shape[1]
            model = ChurnMLP(input_dim=input_size)
            logger.info(f"🧠 Treinando Rede Neural com {input_size} entradas...")
            model = train_mlp(model, loader, params)

            # 5. Avaliação no conjunto de teste (holdout)
            metrics, y_pred, y_proba = evaluate_model(model, X_test_tensor, y_test)

            # 6. Registro de governança no MLflow
            mlflow.log_metrics(metrics)
            input_example = X_test_tensor[:1].numpy()
            signature = infer_signature(input_example, y_pred[:1])
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="pickle",
            )

            # 7. Persistência local para deploy
            torch.save(model.state_dict(), MODEL_FILE)
            logger.info(f"📦 Modelo persistido localmente em: {MODEL_FILE}")

        except Exception as e:
            logger.error(f"❌ Falha crítica no pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    execute_training()
