# Machine Learning Canvas: Previsão de Churn (Tech Challenge - Fase 1)

## 1. Problema de Negócio e Proposta de Valor
*   **O Problema:** A operadora de telecomunicações está enfrentando uma alta taxa de cancelamento de clientes (churn), o que impacta diretamente a receita e a sustentabilidade do negócio.
*   **Objetivo de Negócio:** Identificar com antecedência clientes com alta propensão de cancelamento para que a equipe de retenção possa realizar ações preventivas personalizadas.
*   **Impacto Esperado:** Reduzir a taxa de churn trimestral (ex: meta de 15%) e aumentar o tempo de vida do cliente (LTV).

## 2. Partes Interessadas (Stakeholders)
*   **Equipe de CRM:** Detentores do histórico de interação e perfil do cliente.
*   **Marketing:** Responsáveis por criar ofertas de fidelização baseadas nas predições.
*   **Suporte Técnico:** Fornecem dados sobre reclamações e qualidade do sinal.
*   **Usuários Finais:** Equipe de retenção que utilizará o score de risco gerado pelo modelo para priorizar contatos.

## 3. Métricas de Sucesso
*   **KPIs de Negócio:** Redução percentual na taxa de churn mensal e cálculo do "Custo de Churn Evitado".
*   **Métricas Técnicas (Data Science):** Como o churn é um problema de classificação binária, utilizaremos: **AUC-ROC**, **PR-AUC** e **F1-Score**.
*   **Métrica Prioritária:** O **Recall** será priorizado para garantir que identifiquemos o maior número possível de clientes em risco, minimizando os falsos negativos.

## 4. Requisitos Não Funcionais e SLOs (Service Level Objectives)
*   **Latência:** A API de inferência deve responder em menos de 200ms para não impactar o sistema de atendimento em tempo real.
*   **Disponibilidade:** O serviço de predição deve estar disponível 99.9% do tempo (SLA).
*   **Escalabilidade:** O sistema deve suportar picos de até 100 requisições por segundo durante campanhas de marketing.

## 5. Tarefa de Machine Learning
*   **Tipo de Tarefa:** Classificação Binária.
*   **Variável Alvo (Target):** Coluna `Churn`, que será convertida de categorias textuais ("Yes", "No") para valores numéricos (1 e 0), onde 1 indica a perda do cliente.
*   **Abordagem Central:** Rede Neural MLP em **PyTorch**, comparada com baselines de **Regressão Logística** e **Dummy Classifier** (Scikit-Learn).

## 6. Dados e Variáveis (Features)
*   **Fontes de Dados:** Dataset Telco Customer Churn (IBM), contendo 7.043 registros e 21 variáveis.
*   **Variáveis Relevantes (Features):** 
    *   **Demográficas:** `gender` (Gênero), `SeniorCitizen` (Idoso), `Partner` (Parceiro) e `Dependents` (Dependentes).
    *   **Serviços:** `tenure` (meses de permanência), `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV` e `StreamingMovies`.
    *   **Contrato e Financeiro:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges` e `TotalCharges`.
*   **Data Readiness:** 
    *   Remoção do `customerID` (sem valor preditivo).
    *   Tratamento de espaços vazios na coluna `TotalCharges`, que será convertida de texto para numérico.
    *   Codificação de variáveis categóricas via One-Hot Encoding.

## 7. Governança e Ética
*   **Atributos Sensíveis:** Monitoraremos o viés do modelo em relação ao gênero (`gender`) e à faixa etária (`SeniorCitizen`).
*   **Justiça (Fairness):** Utilizaremos o **Fairlearn** para garantir que a taxa de erro (especialmente o **Recall**) seja equitativa entre homens/mulheres e jovens/idosos.
*   **Compliance:** Manteremos a linhagem de dados (data lineage) via **MLflow**, registrando qual versão do CSV gerou cada modelo para fins de auditabilidade e LGPD.
