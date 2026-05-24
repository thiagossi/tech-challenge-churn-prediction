# 📋 Model Card - FIAPMobile Churn Prediction

## 1. Detalhes do Modelo
*   **Desenvolvedor:** Thiago Soares Simões (RM374443)
*   **Data:** Maio de 2026
*   **Versão:** 1.0.0 (Champion Model)
*   **Tipo de Modelo:** Rede Neural MLP (Multilayer Perceptron) via **PyTorch**.

## 2. Uso Pretendido
*   **Objetivo:** Classificar clientes com propensão ao cancelamento (Churn) para ações de retenção proativa.
*   **Usuários:** Equipes de Marketing e CRM da FIAPMobile.
*   **Limitações:** Modelo validado apenas para clientes pessoa física. Não deve ser utilizado para decisões automáticas de crédito ou políticas discriminatórias de preço.

## 3. Performance (Métricas Técnicas)
As métricas foram validadas em conjunto de teste (20%) e registradas no **MLflow**:
*   **Recall (Sensibilidade):** Priorizado para minimizar Falsos Negativos (clientes que saem sem detecção).
*   **Acurácia:** Baseline de ~80% superado em relação à Regressão Logística.
*   **F1-Score:** Equilíbrio estatístico entre precisão e sensibilidade.

## 4. Avaliação de Vieses (Fairness)
Seguindo as normas de governança, o modelo foi auditado com **Fairlearn**:
*   **Atributos Sensíveis:** Gênero e Senioridade (Idosos).
*   **Critério:** Verificação de disparidade na Taxa de Falso Negativo entre grupos demográficos.
*   **Resultado:** O modelo mantém paridade estatística aceitável, não penalizando subgrupos específicos.

## 5. Cenários de Falha (Failure Scenarios)
Os seguintes cenários podem invalidar as predições:
*   **Data Drift Abrupto:** Mudanças repentinas no mercado (ex: crise econômica ou pandemia) que alteram o comportamento de consumo histórico.
*   **Training-Serving Skew:** Divergência entre o formato dos dados de treino e os dados reais recebidos pela API (protegido pelo contrato `columns.json`).
*   **Degradação Silenciosa (Model Drift):** Queda gradual de performance técnica devido ao envelhecimento do modelo frente a novos planos de competidores.
*   **Latência Crítica:** Sobrecarga de infraestrutura que eleva o tempo de resposta acima dos SLOs definidos, inviabilizando a triagem em tempo real.

## 6. Fatores de Risco e Mitigação
*   **Atraso de Ground Truth:** O churn real só é confirmado após 30 dias, gerando um gap de feedback para retreino.
*   **Ação:** Implementação de monitoramento mensal e plano de contingência para rollback de versão via **Model Registry** do MLflow.