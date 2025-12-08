# Análise Comparativa Consolidada - Detecção de Fraude

**Data da última execução:** 07/12/2025
**Dataset:** Credit Card Fraud Detection (Kaggle)
**Total de transações:** 284,807
**Fraudes:** 492 (0.173%)
**Desbalanceamento:** 577.9:1

---

## Resumo Executivo

Este projeto implementou e comparou **8 modelos diferentes** para detecção de fraude em transações de cartão de crédito, incluindo métodos tradicionais, deep learning e ensemble. O **Random Forest** obteve o melhor resultado geral com F1-Score de 0.822.

---

## Resultados Consolidados - Todos os Modelos

**Fonte:** Script `fraud_detection_comparison_all.py` (execução 07/12/2025)

| Modelo | Categoria | F1-Score | Precision | Recall | ROC-AUC | PR-AUC | Tempo (s) |
|--------|-----------|----------|-----------|--------|---------|--------|-----------|
| **Random Forest** | Tradicional | **0.822** | **0.818** | 0.827 | **0.977** | 0.818 | 10.1 |
| **XGBoost** | Tradicional | **0.656** | 0.532 | **0.857** | 0.976 | **0.845** | 0.9 |
| Isolation Forest | Não Supervisionado | 0.322 | 0.308 | 0.337 | 0.954 | 0.218 | 1.1 |
| MLP Wide | Deep Learning | 0.285 | 0.169 | 0.898 | 0.970 | 0.721 | 43.0 |
| Gradient Boosting | Tradicional | 0.273 | 0.529 | 0.184 | 0.347 | 0.157 | 129.4 |
| MLP Deep | Deep Learning | 0.171 | 0.095 | 0.898 | 0.973 | 0.713 | 36.0 |
| Logistic Regression | Tradicional | 0.114 | 0.061 | **0.918** | 0.972 | 0.716 | 1.2 |
| Autoencoder | Deep Learning | 0.056 | 0.029 | 0.888 | 0.963 | 0.193 | 45.6 |

### Destaques

- **Melhor F1-Score:** Random Forest (0.822)
- **Melhor Precision:** Random Forest (0.818)
- **Melhor Recall:** Logistic Regression (0.918)
- **Melhor ROC-AUC:** Random Forest (0.977)
- **Melhor PR-AUC:** XGBoost (0.845)
- **Mais Rápido:** XGBoost (0.9s)

---

## Resultados de Ensemble

**Fonte:** Script `fraud_detection_ensemble.py` (execução 07/12/2025)

| Método | Tipo | F1-Score | Precision | Recall | ROC-AUC | PR-AUC |
|--------|------|----------|-----------|--------|---------|--------|
| Voting Hard | Ensemble | 0.814 | 0.783 | 0.847 | - | - |
| Weighted Average | Ensemble | 0.776 | 0.716 | 0.847 | 0.978 | 0.823 |
| Voting Soft | Ensemble | 0.767 | 0.694 | 0.857 | 0.974 | 0.752 |
| Stacking | Ensemble | 0.135 | 0.073 | 0.908 | 0.978 | 0.833 |

---

## Comparação com Estado da Arte

| Métrica | Nossa Melhor | Estado da Arte | Gap | Status |
|---------|--------------|----------------|-----|---------|
| F1-Score | **0.822** (RF) | 0.87-0.90 | -5.5% | Muito Próximo |
| ROC-AUC | **0.977** (RF) | 0.98-0.99 | -0.3% | Praticamente Alcançado |
| PR-AUC | **0.845** (XGB) | 0.88-0.92 | -4.0% | Muito Próximo |
| Precision | **0.818** (RF) | 0.90+ | -9.1% | Bom |

**Conclusão:** Estamos a **3-6%** do estado da arte!

---

## Trade-offs e Cenários de Uso

### Cenário 1: Minimizar Falsos Positivos (Melhor UX)

**Recomendação:** Random Forest (Precision=0.818)
- Apenas 18.2% de transações legítimas bloqueadas
- Detecta 82.7% das fraudes
- ROC-AUC=0.977 (excelente separação)

### Cenário 2: Maximizar Detecção de Fraudes

**Recomendação:** Logistic Regression (Recall=0.918)
- Captura 91.8% das fraudes
- Muitos falsos positivos (Precision=0.061)

### Cenário 3: Balanço Otimizado (Produção)

**Recomendação:** Random Forest (F1=0.822)
- Melhor compromisso entre precisão e recall
- **Solução recomendada para produção**

### Cenário 4: Performance e Velocidade

**Recomendação:** XGBoost (F1=0.656, Tempo=0.9s)
- Muito rápido (0.9s vs 10.1s do RF)
- PR-AUC mais alto (0.845)

---

## Ranking Final por F1-Score

1. Random Forest: 0.822
2. Voting Hard (Ensemble): 0.814
3. Weighted Average (Ensemble): 0.776
4. Voting Soft (Ensemble): 0.767
5. XGBoost: 0.656
6. Isolation Forest: 0.322
7. MLP Wide: 0.285
8. Gradient Boosting: 0.273

---

## Conclusão

**Modelo Recomendado:** Random Forest
- F1-Score: 0.822
- Precision: 0.818
- Recall: 0.827
- ROC-AUC: 0.977
- Tempo: 10.1s

**Melhoria sobre baseline:** +155% em F1-Score (0.322 → 0.822)

**Proximidade ao estado da arte:** 94-99.7% dos melhores resultados publicados
