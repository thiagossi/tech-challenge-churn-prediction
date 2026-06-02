import logging
import os

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger("api")


# ── Funções standalone (usadas no notebook) ───────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um caminho e valida sua existência."""
    if not os.path.exists(path):
        logger.error(f"Arquivo não encontrado: {path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def clean_data(df) -> pd.DataFrame:
    """Remove colunas irrelevantes e converte o alvo para binário."""
    df_clean = df.copy()

    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    ).fillna(0)

    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == "object":
            df_clean["Churn"] = df_clean["Churn"].apply(
                lambda x: 1 if x == "Yes" else 0
            )

    logger.info("Saneamento de dados (Data Readiness) concluído.")
    return df_clean


def prepare_features(df: pd.DataFrame, columns_list: list = None) -> pd.DataFrame:
    """Aplica One-Hot Encoding e garante que todos os dados sejam numéricos."""
    X = df.drop(columns=["Churn"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    if columns_list:
        X = X.reindex(columns=columns_list, fill_value=0)

    logger.info(f"Features preparadas. Total de colunas: {X.shape[1]}")
    return X.astype(int)


# ── Transformadores sklearn (usados no Pipeline de produção) ──────────────────

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Passo 1 do Pipeline: limpeza de dados brutos.

    - Remove customerID e Churn (se presentes)
    - Converte TotalCharges para numérico, preenchendo NaN com 0

    Não tem parâmetros aprendidos — fit() não faz nada além de retornar self.
    Isso é intencional: limpeza é determinística, não depende dos dados de treino.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        for col in ["customerID", "Churn"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        ).fillna(0)

        return df


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Passo 2 do Pipeline: One-Hot Encoding com alinhamento de colunas.

    fit(): aprende quais colunas existem no conjunto de TREINO após get_dummies.
    transform(): aplica get_dummies e alinha ao schema aprendido.

    Isso garante que dados de inferência (API) sempre tenham exatamente
    as mesmas colunas que o modelo viu durante o treino — sem columns.json externo.
    """

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = pd.get_dummies(df, drop_first=True).columns.tolist()
        logger.info(f"FeatureEncoder: {len(self.columns_)}"
            f" colunas aprendidas no treino."
        )
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        encoded = pd.get_dummies(df, drop_first=True)
        aligned = encoded.reindex(columns=self.columns_, fill_value=0).astype(int)
        return aligned.values


# ── Fábrica do Pipeline ───────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Constrói o pipeline reprodutível de pré-processamento.

    Uso em treino:
        pipeline = build_pipeline()
        X_train = pipeline.fit_transform(X_train_df)
        X_test  = pipeline.transform(X_test_df)
        joblib.dump(pipeline, 'models/pipeline.joblib')

    Uso em produção (API):
        pipeline = joblib.load('models/pipeline.joblib')
        X = pipeline.transform(df_input)
    """
    return Pipeline([
        ("cleaner", DataCleaner()),
        ("encoder", FeatureEncoder()),
    ])
