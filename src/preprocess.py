import logging
import os

import pandas as pd

# Configuração básica de logging
logger = logging.getLogger(__name__)

def carregar_dados(path: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um caminho e valida sua existência.
    """
    if not os.path.exists(path):
        logger.error(f"Arquivo não encontrado: {path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def limpar_dados(df) -> pd.DataFrame:
    """
    Remove colunas irrelevantes e converte o alvo para binário.
    """
    df_clean = df.copy()
    # Remove customerID se existir
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])        

    # Transforma texto em número. Espaços vazios viram NaN.
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    # Clientes novos (tenure 0) não têm cobrança, preenchemos com 0 conforme a EDA.
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
    
    # Converte Churn para 0 e 1 se for texto
    # para api não preciso da coluna Churn, mas para treino sim
    if 'Churn' in df_clean.columns: 
        if df_clean['Churn'].dtype == 'object':
            df_clean['Churn'] = df_clean['Churn'].apply(
                lambda x: 1 if x == 'Yes' else 0
            )
        
    logger.info("Saneamento de dados (Data Readiness) concluído.")
    return df_clean

def preparar_features(df: pd.DataFrame, columns_list: list = None) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding e garante que todos os dados sejam numéricos.
    """
    # X é tudo exceto o alvo
    X = df.drop(columns=['Churn'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    
    # Se uma lista de colunas for fornecida, alinhamos o DataFrame a ela
    if columns_list:
        # Cria as colunas que faltam com 0 e remove as que não deveriam estar lá
        X = X.reindex(columns=columns_list, fill_value=0)
        
    logger.info(f"Features preparadas. Total de colunas: {X.shape[1]}")
    return X.astype(int)

