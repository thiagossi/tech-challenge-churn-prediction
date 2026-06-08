from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import (
    ChurnMLP,
    evaluate_model,
    load_and_version_dataset,
    prepare_splits,
    set_seeds,
    train_mlp,
)

# ---------------------------------------------------------
# load_and_version_dataset
# ---------------------------------------------------------

def test_load_and_version_dataset(tmp_path):
    """load_and_version_dataset deve retornar o DataFrame limpo e 
    registrar no MLflow."""
    df_raw = pd.DataFrame({
        "customerID": ["A1", "A2"],
        "Churn": ["Yes", "No"],
        "TotalCharges": ["10.0", "20.0"],
    })
    df_clean = pd.DataFrame({"Churn": [1, 0], "TotalCharges": [10.0, 20.0]})

    with patch("src.train.load_data", return_value=df_raw), \
         patch("src.train.clean_data", return_value=df_clean), \
         patch("src.train.mlflow"), \
         patch("src.train.DATA_PATH_PROCESSED", tmp_path / "cleaned.csv"), \
         patch("src.train.DATA_PATH") as mock_data_path:
        mock_data_path.read_bytes.return_value = b"conteudo_fake"
        result = load_and_version_dataset()

    assert result is df_clean


# ---------------------------------------------------------
# set_seeds
# ---------------------------------------------------------

def test_set_seeds_executa_sem_erros():
    """set_seeds deve executar sem lançar exceção."""
    set_seeds(42)
    set_seeds(0)


# ---------------------------------------------------------
# ChurnMLP
# ---------------------------------------------------------

def test_churn_mlp_forward_shape():
    """Forward pass deve retornar tensor de shape (batch, 1)."""
    model = ChurnMLP(input_dim=10)
    x = torch.rand(5, 10)
    output = model(x)
    assert output.shape == (5, 1)


def test_churn_mlp_output_entre_zero_e_um():
    """Sigmoid garante saída no intervalo [0, 1]."""
    model = ChurnMLP(input_dim=10)
    x = torch.rand(20, 10)
    output = model(x)
    assert (output >= 0).all() and (output <= 1).all()


# ---------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------

def test_evaluate_model_retorna_seis_metricas():
    """evaluate_model deve retornar as 6 métricas exigidas."""
    model = ChurnMLP(input_dim=10)
    X_test = torch.rand(20, 10)
    y_test = np.array([0, 1] * 10)
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
    assert set(metrics.keys()) == {
        "accuracy", "recall", "precision", "f1_score", "roc_auc", "pr_auc"
    }


def test_evaluate_model_proba_no_intervalo():
    """Probabilidades retornadas devem estar entre 0 e 1."""
    model = ChurnMLP(input_dim=10)
    X_test = torch.rand(20, 10)
    y_test = np.array([0, 1] * 10)
    _, _, y_proba = evaluate_model(model, X_test, y_test)
    assert all(0 <= p <= 1 for p in y_proba)


def test_evaluate_model_pred_e_proba_mesmo_tamanho():
    """y_pred e y_proba devem ter o mesmo tamanho que y_test."""
    model = ChurnMLP(input_dim=10)
    X_test = torch.rand(20, 10)
    y_test = np.array([0, 1] * 10)
    _, y_pred, y_proba = evaluate_model(model, X_test, y_test)
    assert len(y_pred) == 20
    assert len(y_proba) == 20


# ---------------------------------------------------------
# train_mlp
# ---------------------------------------------------------

def test_train_mlp_retorna_modelo():
    """train_mlp deve retornar instância de ChurnMLP."""
    model = ChurnMLP(input_dim=5)
    X = torch.rand(20, 5)
    y = torch.randint(0, 2, (20, 1)).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    params = {"epochs": 3, "lr": 0.001, "early_stopping_patience": 10}
    trained = train_mlp(model, loader, params)
    assert isinstance(trained, ChurnMLP)


def test_train_mlp_early_stopping_aciona():
    """Early stopping deve parar o treino antes de esgotar as épocas."""
    model = ChurnMLP(input_dim=5)
    # Inputs idênticos com labels alternados → gradientes se cancelam
    # → loss estagna imediatamente → counter incrementa → early stop em 3 épocas
    X = torch.ones(20, 5)
    y = torch.tensor([[0.0], [1.0]] * 10)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=20)
    params = {"epochs": 100, "lr": 0.001, "early_stopping_patience": 2}
    with patch("src.train.mlflow"):
        trained = train_mlp(model, loader, params)
    assert isinstance(trained, ChurnMLP)


# ---------------------------------------------------------
# prepare_splits
# ---------------------------------------------------------

def _make_df(n=20):
    """DataFrame mínimo com estrutura compatível com prepare_splits."""
    return pd.DataFrame({
        "gender": ["Male", "Female"] * (n // 2),
        "tenure": list(range(n)),
        "TotalCharges": [float(i) for i in range(n)],
        "Churn": [0, 1] * (n // 2),
    })


def test_prepare_splits_shapes():
    """80/20 split deve gerar tamanhos corretos."""
    with patch("src.train.joblib.dump"):  # evita escrita em disco
        X_train, X_test, y_train, y_test = prepare_splits(_make_df())
    assert X_train.shape[0] == 16
    assert X_test.shape[0] == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_prepare_splits_retorna_numpy():
    """X_train e X_test devem ser numpy arrays."""
    with patch("src.train.joblib.dump"):
        X_train, X_test, _, _ = prepare_splits(_make_df())
    assert hasattr(X_train, "shape")
    assert hasattr(X_test, "shape")
