# 📈 Plano de Monitoramento e Manutenção - FIAPMobile Churn

## 1. Introdução
Este plano define a estratégia de observabilidade para o modelo de predição de Churn da FIAPMobile em produção. O objetivo é garantir que a API permaneça estável e que o modelo continue generalizando corretamente, detectando degradações antes que impactem o negócio.

---

## 2. Métricas de Monitoramento

Dividimos o monitoramento em três camadas essenciais, conforme as boas práticas de MLOps:

### 2.1. Métricas de Serviço (SRE Golden Signals)
Focam na saúde da API FastAPI.
*   **Latência (p95 e p99):** Tempo de resposta do endpoint `/predict`. Meta: P99 < 300ms.
*   **Taxa de Erros:** Percentual de requisições com status HTTP 5xx (falhas internas).
*   **Throughput (Vazão):** Número de requisições por segundo (RPS) para medir a carga.
*   **Disponibilidade (Uptime):** Verificação constante via endpoint `/health`.

### 2.2. Métricas de Qualidade de Dados (Data Drift)
Detectam mudanças na distribuição dos dados de entrada.
*   **Distribuição das Features:** Comparação das variáveis (ex: `MonthlyCharges`, `Tenure`) entre o treino e a produção usando o teste estatístico **Kolmogorov-Smirnov (KS)** ou **PSI**.
*   **Integridade dos Dados:** Monitoramento de valores nulos inesperados ou tipos de dados fora do contrato definido no `columns.json`.

### 2.3. Métricas de Performance do Modelo (Model Drift)
Acompanham a eficácia preditiva real conforme o *ground truth* (resultado real) é coletado.
*   **Recall (Revocação):** Métrica prioritária para garantir que não estamos perdendo clientes em risco.
*   **F1-Score:** Equilíbrio entre precisão e sensibilidade.
*   **Prediction Drift:** Mudança na proporção de clientes classificados como "Churn" pela API.

---

## 3. Configuração de Alertas

Os alertas são configurados via Prometheus/Grafana para notificar a equipe proativamente.

| Alerta | Gatilho (Threshold) | Severidade |
| :--- | :--- | :--- |
| **API Down** | Endpoint `/health` fora do ar por > 1 min | Crítica |
| **Alta Latência** | p99 de `/predict` > 500ms por 5 min | Alta |
| **Taxa de Erro** | > 2% das requisições com erro 5xx em 10 min | Alta |
| **Data Drift Detectado** | Teste KS com p-valor < 0.05 em features chave | Média |
| **Queda de Performance**| Recall abaixo de 75% na avaliação mensal | Média |

---

## 4. Playbook de Resposta (Incidência e Manutenção)

Fluxo de ação para quando um alerta for disparado ou uma falha for detectada:

### Passo 1: Triagem e Identificação
*   Verificar se a falha é de **infraestrutura** (ex: servidor sem memória) ou de **modelo** (ex: dados mudaram).
*   Consultar os **Logs Estruturados** no formato JSON para rastrear as requisições afetadas.

### Passo 2: Mitigação Imediata
*   **Rollback de Versão:** Se uma nova atualização causou o erro, reverter para a versão estável anterior (Champion) via **Model Registry** do MLflow.
*   **Modelo de Backup:** Em caso de erro crítico na rede neural, ativar regras de negócio simples ou modelo linear de fallback para não interromper o serviço.

### Passo 3: Resolução (Manutenção Evolutiva)
*   **Retreinamento Automático (CT):** Se o alerta for de Drift, acionar o pipeline de retreino via **Makefile** (`make train`) com os dados mais recentes coletados.
*   **Análise de Causa Raiz:** Realizar um *Post-mortem* para entender se o drift é abrupto (mudança de mercado) ou gradual.

### Passo 4: Validação e Promoção
*   O novo modelo retreinado deve superar a performance do modelo atual em um conjunto de teste "hold-out" antes de ser promovido para produção novamente.