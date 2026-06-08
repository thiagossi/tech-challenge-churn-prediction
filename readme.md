# 🚀 Tech Challenge: Previsão de Churn com Redes Neurais

Este repositório contém a solução para o **Tech Challenge da Fase 1** da Pós-Graduação em Machine Learning Engineering da FIAP.

O projeto consiste no desenvolvimento de um pipeline **end-to-end** para prever a rotatividade de clientes (**churn**) de uma operadora de telecomunicações utilizando **Redes Neurais**.

---

# 📋 Contexto de Negócio

A FIAPMobile enfrenta uma alta taxa de cancelamento de clientes. O objetivo técnico é desenvolver um modelo de **classificação binária** capaz de identificar clientes propensos ao churn, permitindo ações preventivas de retenção.

## 🎯 Métricas de Sucesso

### Técnica
Maximizar o **Recall** para capturar o maior número possível de clientes em risco.

### Negócio
Reduzir o custo operacional do churn através de predições antecipadas.

---

# 📁 Estrutura do Projeto

Seguindo boas práticas de organização profissional para **MLOps**:

```text
tech-challenge-churn-prediction/
├── data/               # Dados do projeto (ignorado pelo Git)
│   ├── raw/            # Dados brutos imutáveis
│   └── processed/      # Dados após limpeza e transformação
├── docs/               # Documentação técnica e governança (ML Canvas, Model Card)
├── models/             # Artefatos de modelos serializados
├── notebooks/          # Jupyter Notebooks para EDA e experimentação
├── src/                # Código-fonte modularizado (Scripts de treino e inferência)
├── tests/              # Testes unitários e de integração (Pytest)
├── .gitignore          # Proteção de dados sensíveis e binários
├── Makefile            # Automação de comandos
├── pyproject.toml      # Fonte única de verdade para dependências e linting
└── README.md           # Instruções de uso (este arquivo)
```

# 🚀 Instalação e Setup

Para esse projeto foi utilizada a linguagem **Python 3.11.9**. \
Caso você ainda não possua o Python, baixe-o acessando a seção de downloads do Python nesse [link](https://www.python.org/downloads/) e instale-o no seu sistema operacional. \
Opte por baixar a mesma versão ou superior. \
Obs.: Versões diferentes da qual foi utilizada nesse projeto, podem ocorrer falhas ou instabilidades.

## 1️⃣ Clonar o repositório no GitHub

```bash
git clone https://github.com/thiagossi/tech-challenge-churn-prediction
cd tech-challenge-churn-prediction
```
---

## 2️⃣ 📥 Download do Dataset

### ⚙️ Configuração do Dataset

Por questões de governança e segurança, o arquivo de dados não está incluído no repositório. \
Após clonar o repositório em sua máquina, siga os passos abaixos.

1. Acesse o dataset **Telco Customer Churn (IBM)** no Kaggle através do link abaixo:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

2. Baixe o arquivo:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

3. Coloque o arquivo na pasta:

```text
data/raw/
```

4. Renomeie o arquivo para:

```text
telco_customer_churn.csv
```

## 3️⃣ Criar o ambiente virtual

```bash
python -m venv .venv
```

## 4️⃣ Ativar o ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate 
```
ou

```bash
source .venv/scripts/activate 
```

## 4️⃣ Instalar dependências

### Instalação e Setup

Este projeto utiliza o arquivo `pyproject.toml` como **Single Source of Truth** para dependências e um `Makefile` para automação de tarefas.

### 🐧 Linux e macOS

O utilitário `make` é nativo. Para configurar o ambiente e instalar as dependências, basta rodar:

```bash
make install
```

### 🪟 Windows

#### Configurando o Make no Windows

Para utilizar os comandos acima no Windows, siga este setup rápido (PowerShell como Admin):

### 1. Instalar

```powershell
winget install GnuWin32.Make
```

### 2. Variável de Ambiente

Adicione o caminho abaixo ao seu `PATH`:

```text
C:\Program Files (x86)\GnuWin32\bin
```

### 3. Reiniciar

Feche e abra novamente seu terminal ou VS Code.

Conforme definido no arquivo `pyproject.toml`.

---

Se você o Make foi instalado corretamente no seu windows o comando `make` já poderá ser usado.

```bash
make install
```

Caso contrário, use o comando manual do `pip` abaixo:

```bash
pip install -e .
```
---

# 🛠️ Automação com Makefile

O `Makefile` centraliza os comandos principais do ciclo de vida do projeto, garantindo que a execução seja idêntica em qualquer ambiente.

## 🚀 Comandos Principais

| Comando | Descrição |
|---|---|
| `make install` | Instala o projeto em modo editável e todas as dependências |
| `make lint` | Executa a verificação de estilo e erros com o Ruff |
| `make train` | Executa o script de treinamento do modelo Churn |
| `make test` | Roda a suíte de testes automatizados com relatório de cobertura (≥90%) |
| `make run` | Inicia a API FastAPI (Uvicorn) em modo reload |

---

# ▶️ Executando a API

Após instalar as dependências, execute o comando abaixo para iniciar a aplicação:

```bash
make run
```

A API normalmente ficará disponível no endereço local:

```text
http://127.0.0.1:8000/
```

Com a aplicação em execução, você poderá utilizar ferramentas como:

- Insomnia
- Postman
- Swagger UI
- Navegador Web

para testar e validar as rotas da API.

---

# 🌐 Principais Rotas da API

## 📘 Documentação Interativa

A documentação automática gerada pelo FastAPI pode ser acessada pela rota:

```http
GET /docs
```

Exemplo:

```text
http://127.0.0.1:8000/docs
```

Nesta página será possível:

- Visualizar todas as rotas disponíveis
- Testar requisições diretamente pelo navegador
- Ver exemplos de payloads
- Validar respostas da API

---

## ❤️ Health Check

Rota utilizada para verificar a saúde da aplicação.

```http
GET /health
```

Exemplo:

```text
http://127.0.0.1:8000/health
```

Essa rota é útil para:

- Monitoramento da aplicação
- Testes automatizados
- Validação em pipelines CI/CD
- Verificação rápida se a API está online

---

## 🔮 Predição de Churn

Rota principal responsável por receber os dados de um cliente e retornar a probabilidade de churn.

```http
POST /predict
```

Exemplo:

```text
http://127.0.0.1:8000/predict
```

Nessa rota, enviamos um JSON contendo os dados do cliente para que o modelo de Machine Learning consiga realizar a previsão.

Exemplo simplificado de payload:

```json
{
	"gender": "Female",
	"SeniorCitizen": 0,
	"Partner": "Yes",
	"Dependents": "No",
	"tenure": 1,
	"PhoneService": "No",
	"MultipleLines": "No phone service",
	"InternetService": "DSL",
	"OnlineSecurity": "No",
	"OnlineBackup": "Yes",
	"DeviceProtection": "No",
	"TechSupport": "No",
	"StreamingTV": "No",
	"StreamingMovies": "No",
	"Contract": "Month-to-month",
	"PaperlessBilling": "Yes",
	"PaymentMethod": "Electronic check",
	"MonthlyCharges": 29.85,
	"TotalCharges": 29.85
}
```

A API retornará informações como:

- Probabilidade de churn
- Status da requisição

---

# 🛠️ Tecnologias Utilizadas

## 🧠 Machine Learning

- **PyTorch**  
  Desenvolvimento da Rede Neural MLP.

- **Scikit-Learn**  
  Pipeline reprodutível de pré-processamento (`DataCleaner` + `FeatureEncoder`) e modelos baseline (*Dummy Classifier*, *Regressão Logística* e *Random Forest*).

## 📊 Observabilidade e MLOps

- **MLflow**  
  Rastreamento de experimentos e registro de modelos.

## 🌐 API

- **FastAPI**  
  Disponibilização do modelo via API REST.

## ✅ Qualidade de Código

- **Ruff**  
  Garantia de qualidade e padronização do código.

- **Pytest**  
  Execução de testes automatizados (smoke, schema Pydantic, schema Pandera e API).

- **Pandera**  
  Validação de schema do DataFrame de entrada — garante domínio correto dos valores além da tipagem.

- **Joblib**  
  Serialização do pipeline sklearn (`pipeline.joblib`) para garantir transformação idêntica entre treino e inferência.

---

# 🧪 Execução do Projeto

## 1️⃣ Análise Exploratória (EDA)

Acesse a pasta `notebooks/` para visualizar:

- Qualidade dos dados
- Identificação de outliers
- Correlação entre features
- Distribuição das variáveis

---

## 2️⃣ Treinamento

O `make train` treina a **Rede Neural MLP** (modelo campeão), salva o `pipeline.joblib` e registra métricas e dataset version no MLflow.

Os **modelos baseline** (Dummy, Regressão Logística e Random Forest) são treinados e comparados no notebook `notebooks/eda_churn.ipynb` — execute as células da seção "Treinamento das baselines" para registrá-los no MLflow.

com uso do make
```bash
make train
```

Sem uso do make
```bash
python src/train.py
```

---

## 3️⃣ Qualidade de Código (Linting)

com uso do make
```bash
make lint
```

Sem uso do make
```bash
ruff check .
```

---

## 4️⃣ Testes

A suíte cobre **90%** do código-fonte e está organizada em três arquivos:

- `tests/test_main.py` — smoke test, validação Pydantic, validação Pandera e testes funcionais da API
- `tests/test_preprocess.py` — testes unitários de `load_data`, `clean_data`, `DataCleaner`, `FeatureEncoder` e `build_pipeline`
- `tests/test_train.py` — testes unitários de `ChurnMLP`, `evaluate_model`, `train_mlp`, `prepare_splits` e `load_and_version_dataset`

com uso do make
```bash
make test
```

Sem uso do make
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

# 🛡️ Governança e Ética

## 📄 Model Card

Documentação detalhada de:

- Limitações do modelo
- Vieses conhecidos
- Contexto de uso
- Métricas de avaliação

Arquivo `model_card.md` disponível na pasta `docs/`.

---
## 📄 Plano de Monitoração

Documentação detalhada para demonstrar a estratégia de observabilidade do modelo de predição

Arquivo `monitoring_plan.md` disponível na pasta `docs/`.

## ⚖️ Fairness

Monitoramento de disparidade entre grupos sensíveis:

- `gender`
- `SeniorCitizen`

Utilizando Fairlearn.

---

## 🔍 Rastreabilidade

Cada modelo treinado está vinculado a:

- **Dataset version** via hash MD5 do arquivo CSV (ex: `telco_churn_0f9de68e`), registrado no MLflow via `mlflow.log_input()`
- Parâmetros e hiperparâmetros utilizados
- Métricas de avaliação (Recall, F1, AUC-ROC, PR-AUC, Accuracy, Precision)

## 📐 Arquitetura de Deploy e Justificativa

Para o projeto da FIAPMobile, a arquitetura de implantação escolhida foi a **Inferência em Tempo Real (Online Inference)**, expondo o modelo através de uma **API REST** utilizando o framework **FastAPI**.

### 🛠️ Escolha Técnica: FastAPI
A escolha do **FastAPI** baseia-se em sua alta performance (baseada Uvicorn) e na capacidade de garantir um "contrato" robusto entre o modelo e os sistemas consumidores através da validação automática de dados com **Pydantic**.

### 💡 Justificativa: Real-Time vs. Batch
A decisão de não utilizar o processamento em lote (Batch) e optar pela API em tempo real fundamenta-se nos seguintes pontos:

1. **Retenção Proativa (Valor de Negócio):** O Churn em telecomunicações exige rapidez. Ter um endpoint disponível permite que, no momento em que um cliente solicita o cancelamento ou interage com o suporte, o sistema de CRM consulte o modelo e ofereça uma oferta de retenção personalizada instantaneamente.
2. **Desacoplamento de Sistemas:** Ao servir o modelo via API, isolamos o ciclo de vida da inteligência (equipe de ML) das aplicações móveis e web da FIAPMobile. Isso permite atualizar os pesos do modelo (`.pt`) de forma independente, sem necessidade de alterar o código de outros sistemas da operadora.
3. **Interoperabilidade:** O uso de **JSON via HTTP** torna o modelo de Churn acessível para qualquer linguagem de programação ou plataforma utilizada pela FIAPMobile, garantindo uma integração padronizada e simples.

### 🚀 Reprodutibilidade e Automação
A consistência do ambiente é garantida através de:
- **Single Source of Truth:** O arquivo `pyproject.toml` centraliza todas as dependências e versões das bibliotecas (PyTorch, Scikit-Learn, etc.), garantindo que o ambiente de execução seja idêntico ao de desenvolvimento.
- **Interface de Comando Única:** O uso do **Makefile** automatiza os processos de instalação, testes e execução, eliminando erros manuais e simplificando a operação do serviço.