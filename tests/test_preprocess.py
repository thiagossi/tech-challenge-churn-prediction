import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    DataCleaner,
    FeatureEncoder,
    build_pipeline,
    clean_data,
    load_data,
)

# ---------------------------------------------------------
# load_data
# ---------------------------------------------------------

def test_load_data_file_not_found():
    """Caminho inexistente deve lançar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("caminho/inexistente.csv")


def test_load_data_success(tmp_path):
    """CSV válido deve ser carregado como DataFrame."""
    csv = tmp_path / "test.csv"
    csv.write_text("col1,col2\n1,2\n3,4")
    df = load_data(str(csv))
    assert len(df) == 2
    assert list(df.columns) == ["col1", "col2"]


# ---------------------------------------------------------
# clean_data
# ---------------------------------------------------------

def test_clean_data_removes_customer_id():
    """customerID deve ser removido."""
    df = pd.DataFrame({
        "customerID": ["A1"],
        "TotalCharges": ["10.5"],
        "Churn": ["Yes"],
    })
    result = clean_data(df)
    assert "customerID" not in result.columns


def test_clean_data_converts_total_charges_empty_string():
    """TotalCharges vazio (espaço) deve virar 0."""
    df = pd.DataFrame({"TotalCharges": [" ", "29.85"], "Churn": ["No", "Yes"]})
    result = clean_data(df)
    assert result["TotalCharges"].iloc[0] == 0
    assert result["TotalCharges"].iloc[1] == 29.85


def test_clean_data_converts_churn_to_binary():
    """Churn Yes/No deve virar 1/0."""
    df = pd.DataFrame({"TotalCharges": ["10.0", "20.0"], "Churn": ["Yes", "No"]})
    result = clean_data(df)
    assert list(result["Churn"]) == [1, 0]


def test_clean_data_sem_churn_nao_falha():
    """clean_data não deve falhar se coluna Churn não existir."""
    df = pd.DataFrame({"TotalCharges": ["10.0"]})
    result = clean_data(df)
    assert "TotalCharges" in result.columns


# ---------------------------------------------------------
# DataCleaner (transformer sklearn)
# ---------------------------------------------------------

def test_data_cleaner_remove_colunas_extras():
    """DataCleaner deve remover customerID e Churn se presentes."""
    df = pd.DataFrame({
        "customerID": ["A1"],
        "Churn": ["Yes"],
        "TotalCharges": ["29.85"],
        "gender": ["Male"],
    })
    cleaner = DataCleaner()
    result = cleaner.fit_transform(df)
    assert "customerID" not in result.columns
    assert "Churn" not in result.columns
    assert "gender" in result.columns


def test_data_cleaner_converte_total_charges():
    """TotalCharges inválido deve virar 0."""
    df = pd.DataFrame({"TotalCharges": [" "], "gender": ["Male"]})
    cleaner = DataCleaner()
    result = cleaner.transform(df)
    assert result["TotalCharges"].iloc[0] == 0


def test_data_cleaner_fit_retorna_self():
    """fit() deve retornar self para compatibilidade com Pipeline."""
    cleaner = DataCleaner()
    assert cleaner.fit(pd.DataFrame()) is cleaner


# ---------------------------------------------------------
# FeatureEncoder (transformer sklearn)
# ---------------------------------------------------------

def test_feature_encoder_fit_transform():
    """FeatureEncoder deve retornar array numérico após fit_transform."""
    df = pd.DataFrame({"gender": ["Male", "Female"], "tenure": [1, 2]})
    encoder = FeatureEncoder()
    result = encoder.fit_transform(df)
    assert result.shape[0] == 2
    assert result.dtype in [np.int32, np.int64, np.uint8]


def test_feature_encoder_alinha_colunas_na_inferencia():
    """Coluna ausente na inferência deve aparecer com valor 0."""
    train_df = pd.DataFrame({"gender": ["Male", "Female"]})
    infer_df = pd.DataFrame({"gender": ["Male"]})
    encoder = FeatureEncoder()
    encoder.fit(train_df)
    result = encoder.transform(infer_df)
    assert result.shape[1] == len(encoder.columns_)


# ---------------------------------------------------------
# prepare_features (função standalone legacy)
# ---------------------------------------------------------

def test_prepare_features_sem_columns_list():
    """Sem columns_list deve aplicar get_dummies e retornar int."""
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "Churn": [1, 0],
        "tenure": [1, 2],
    })
    from src.preprocess import prepare_features
    result = prepare_features(df)
    assert "Churn" not in result.columns
    assert result.shape[0] == 2


def test_prepare_features_com_columns_list():
    """columns_list deve alinhar colunas ao schema fornecido."""
    df = pd.DataFrame({"gender": ["Male", "Female"], "tenure": [1, 2]})
    from src.preprocess import prepare_features
    encoded = pd.get_dummies(df, drop_first=True)
    result = prepare_features(df, columns_list=list(encoded.columns))
    assert list(result.columns) == list(encoded.columns)


# ---------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------

def test_build_pipeline_executa_fit_transform():
    """Pipeline completo deve transformar DataFrame sem erros."""
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "tenure": [1, 2],
        "TotalCharges": ["29.85", "50.00"],
    })
    pipeline = build_pipeline()
    result = pipeline.fit_transform(df)
    assert result.shape[0] == 2
    assert result.ndim == 2
