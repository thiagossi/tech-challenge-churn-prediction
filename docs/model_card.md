# 📋 Model Card - FIAPMobile Churn Prediction

## 1. Detalhes do Modelo
*   **Desenvolvedor:** Thiago Soares Simões (RM374443)
*   **Data:** Maio de 2026
*   **Versão:** 1.0.0 (Champion Model)
*   **Tipo de Modelo:** Rede Neural MLP (Multilayer Perceptron) via **PyTorch**
*   **Dataset Version:** `telco_churn_0f9de68e` (Telco Customer Churn — IBM, 7.043 registros)
*   **Rastreabilidade:** Experimento `FIAPMobile_Churn_MLP`, run `MLP_Champion_Training` no MLflow

## 2. Uso Pretendido
*   **Objetivo:** Classificar clientes com propensão ao cancelamento (Churn) para ações de retenção proativa.
*   **Usuários:** Equipes de Marketing e CRM da FIAPMobile.
*   **Limitações:** Modelo validado apenas para clientes pessoa física. Não deve ser utilizado para decisões automáticas de crédito ou políticas discriminatórias de preço.

## 3. Performance (Métricas Técnicas)

Métricas validadas no conjunto de teste holdout (20% — 1.409 registros), registradas no MLflow.

### MLP PyTorch (Champion) — Early Stopping na época 71/100

| Métrica | Valor |
|---------|-------|
| **Recall** | **0.5936** |
| **AUC-ROC** | **0.8405** |
| **PR-AUC** | **0.6323** |
| F1-Score | **0.6167** |
| Accuracy | **0.8041** |
| Precision | **0.6416** |

### Comparativo com Baselines (Holdout 20%)

| Modelo | Recall | AUC-ROC | PR-AUC |
|--------|--------|---------|--------|
| DummyClassifier (estratificado) | ~0.26 | ~0.50 | ~0.27 |
| Regressão Logística | 0.5615 | 0.8427 | 0.6361 |
| Random Forest | 0.4973 | 0.8253 | 0.6264 |
| **MLP PyTorch (Champion)** | **0.5936** | **0.8405** | **0.6323** |

**Conclusão:** A MLP superou todos os baselines em Recall (+5,7% vs. Regressão Logística), a métrica prioritária para o negócio — minimizar clientes que cancelam sem serem detectados.

### Threshold de Decisão
*   **Padrão:** 0.5 (probabilidade ≥ 0.5 → prediz churn)
*   **Threshold ótimo de negócio:** determinado pela análise de custo FP/FN (ver notebook, célula de trade-off)

## 4. Avaliação de Vieses (Fairness)
Seguindo as normas de governança, o modelo foi auditado com **Fairlearn**:
*   **Atributos Sensíveis:** Gênero (`gender`) e Senioridade (`SeniorCitizen`).
*   **Critério:** Verificação de disparidade na Taxa de Falso Negativo entre grupos demográficos.
*   **Resultado:** O modelo mantém paridade estatística aceitável, não penalizando subgrupos específicos.

## 5. Cenários de Falha (Failure Scenarios)
*   **Data Drift Abrupto:** Mudanças repentinas no mercado (ex: crise econômica) que alteram o comportamento de consumo histórico.
*   **Training-Serving Skew:** Divergência entre o formato dos dados de treino e os dados reais recebidos pela API — mitigado pelo `pipeline.joblib` que garante transformação idêntica.
*   **Degradação Silenciosa (Model Drift):** Queda gradual de performance devido ao envelhecimento do modelo frente a novos planos de competidores.
*   **Latência Crítica:** Sobrecarga de infraestrutura que eleva o tempo de resposta acima do SLO de 200ms.

## 6. Fatores de Risco e Mitigação
*   **Atraso de Ground Truth:** O churn real só é confirmado após 30 dias, gerando um gap de feedback para retreino.
*   **Ação:** Monitoramento mensal e plano de contingência para rollback via **Model Registry** do MLflow (ver `docs/monitoring_plan.md`).
