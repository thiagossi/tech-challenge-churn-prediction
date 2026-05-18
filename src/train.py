import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from preprocess import carregar_dados, limpar_dados, preparar_features

# Configuração de caminhos dinâmicos
# __file__ é o caminho deste arquivo (train.py)
# .parent volta para 'src'
# .parent volta para a raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_FILE = MODEL_DIR / "modelo_churn.pt"

# 1. FIXANDO SEEDS PARA REPRODUTIBILIDADE
def fixar_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    fixar_seeds() # Chamada obrigatória
    logger.info("🚀 Iniciando Pipeline de Treinamento...")

    try:
        # 1. Carregar o dataset        
        df_raw = carregar_dados(DATA_PATH)
        
        # 2. Pré-processamento Modularizado
        df_clean = limpar_dados(df_raw)
        X = preparar_features(df_clean)
        y = df_clean['Churn'].values

        feature_columns = X.columns.tolist()
        
        with open(ROOT_DIR / "models" / "columns.json", "w") as f:
            json.dump(feature_columns, f)
        logger.info("✅ Lista de colunas salva para a API.")
        
        # 3. Preparação para PyTorch
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 4. Instanciando o Modelo Champion
        input_size = X.shape[1] 
        model = ChurnMLP(input_dim=input_size)    
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        logger.info(f"Estrutura da rede criada com {input_size} entradas.")
        
        # 5. Loop de Treinamento com logging
        logger.info("🧠 Treinando a Rede Neural...")
        model.train()
        for epoch in range(50): # 50 épocas são suficientes para este exemplo
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # 6. Salvamento do Artefato
        torch.save(model.state_dict(), MODEL_FILE)        
        
        model_path = MODEL_FILE.absolute()
        logger.info(f"✅ Treinamento concluído e modelo persistido em: {model_path}")
    except Exception as e:
        logger.error(f"Falha no pipeline: {str(e)}")

if __name__ == "__main__":
    executar_treinamento()