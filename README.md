# Detecção de Fraude em Cartões de Crédito
## Comparação de Algoritmos de Machine Learning

**Dataset:** Credit Card Fraud Detection (Kaggle)
**Total:** 284,807 transações | 492 fraudes (0.173%)
**Desbalanceamento:** 577.9:1

---

## Resultados Principais

| Modelo | F1-Score | Precision | Recall | ROC-AUC |
|--------|----------|-----------|--------|---------|
| **Random Forest** | **0.822** | **0.818** | 0.827 | **0.977** |
| XGBoost | 0.656 | 0.532 | 0.857 | 0.976 |
| Voting Hard | 0.814 | 0.783 | 0.847 | - |

**Proximidade ao estado da arte:** 94-99.7%

---

## Execução Rápida

### Opção 1: Script Automatizado (Recomendado)

```bash
./scripts/run_all_experiments.sh
```

Executa todos os experimentos em sequência:
1. Modelos Principais (RF, XGBoost, GB, LR, Isolation Forest)
2. Deep Learning (Autoencoder, MLP Deep, MLP Wide)
3. Ensemble (Voting, Stacking, Weighted Average)
4. SMOTE Comparison

Tempo total: ~10-15 minutos

### Opção 2: Execução Manual

```bash
# Modelos principais
python src/models/fraud_detection_complete_with_plots.py

# Deep Learning
python src/deep_learning/fraud_detection_deep_learning.py

# Ensemble
python src/ensemble/fraud_detection_ensemble.py

# SMOTE Comparison (opcional)
python src/models/fraud_detection_smote_comparison.py
```

---

## Estrutura do Projeto

```
ML_seminario_1/
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências
├── .gitignore                      # Arquivos ignorados
│
├── src/                            # Código fonte
│   ├── models/                     # Modelos tradicionais
│   │   ├── fraud_detection_complete_with_plots.py
│   │   ├── fraud_detection_smote_comparison.py
│   │   └── isolation_forest_fraud_detection.py
│   │
│   ├── deep_learning/              # Modelos de deep learning
│   │   └── fraud_detection_deep_learning.py
│   │
│   ├── ensemble/                   # Métodos de ensemble
│   │   └── fraud_detection_ensemble.py
│   │
│   └── utils/                      # Utilitários (vazio)
│
├── scripts/                        # Scripts de automação
│   └── run_all_experiments.sh      # Executa todos os experimentos
│
├── docs/                           # Documentação
│   ├── ANALISE_COMPARATIVA_CONSOLIDADA.md
│   ├── DOCUMENTACAO_TECNICA_IMPLEMENTACAO.md
│   └── APRESENTACAO_RESULTADOS.md
│
├── results/                        # Resultados (git ignored)
│   ├── complete_analysis_*/
│   ├── deep_learning_*/
│   ├── ensemble_*/
│   └── smote_comparison_*/
│
└── data/                           # Dados (git ignored, baixado via kagglehub)
```

---

## Instalação

### Pré-requisitos

```bash
Python >= 3.10
```

### Dependências

```bash
pip install -r requirements.txt
```

### Principais bibliotecas:
- scikit-learn >= 1.3.0
- tensorflow >= 2.15.0
- xgboost >= 2.0.0
- imbalanced-learn >= 0.11.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- kagglehub >= 0.2.0

---

## Modelos Implementados

### Métodos Tradicionais (5)
1. **Random Forest** (Melhor F1: 0.822)
2. **XGBoost** (Mais rápido: 0.9s)
3. Gradient Boosting
4. Logistic Regression (Melhor Recall: 0.918)
5. Isolation Forest (Baseline)

### Deep Learning (3)
6. Autoencoder (Detecção de anomalias)
7. MLP Deep (Rede profunda)
8. MLP Wide (Rede larga)

### Ensemble (4)
9. Voting Hard (F1: 0.814)
10. Voting Soft
11. Weighted Average
12. Stacking

### Balanceamento Sintético
13. SMOTE Comparison (Original vs Balanceado)

**Total:** 13 abordagens diferentes

---

## Resultados Detalhados

### Ranking por F1-Score

1. Random Forest: 0.822
2. Voting Hard: 0.814
3. Weighted Average: 0.776
4. Voting Soft: 0.767
5. XGBoost: 0.656
6. Isolation Forest: 0.322
7. MLP Wide: 0.285
8. Gradient Boosting: 0.273

### Comparação SMOTE

| Modelo | Original F1 | SMOTE F1 | Melhoria |
|--------|-------------|----------|----------|
| Random Forest | 0.816 | 0.710 | -13.0% |
| XGBoost | 0.697 | 0.640 | -8.2% |
| Logistic Regression | 0.101 | 0.310 | +208.1% |
| Gradient Boosting | 0.176 | 0.344 | +96.0% |

**Conclusão:** SMOTE piora modelos fortes, melhora modelos fracos.

---

## Documentação Completa

Acesse a pasta `docs/` para análises detalhadas:

- **ANALISE_COMPARATIVA_CONSOLIDADA.md:** Resultados, comparações e análise SMOTE
- **DOCUMENTACAO_TECNICA_IMPLEMENTACAO.md:** Implementação detalhada de cada algoritmo
- **APRESENTACAO_RESULTADOS.md:** Apresentação focada em decisões e impactos

---

## Cenários de Uso Recomendados

### Produção (Balanço)
- **Modelo:** Random Forest
- **F1:** 0.822 | **Precision:** 0.818 | **Recall:** 0.827
- **Por quê:** Melhor compromisso geral

### Alta Performance / Real-time
- **Modelo:** XGBoost
- **F1:** 0.656 | **Tempo:** 0.9s (11x mais rápido)
- **Por quê:** Velocidade + PR-AUC alto (0.845)

### Máxima Detecção
- **Modelo:** Logistic Regression
- **F1:** 0.114 | **Recall:** 0.918
- **Por quê:** Captura 91.8% das fraudes

---

## Dataset

**Fonte:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transações (setembro/2013)
- 492 fraudes (0.172%)
- 30 features (V1-V28 via PCA, Time, Amount)
- Download automático via `kagglehub`

---

## Licença

Projeto acadêmico. Dataset sob licença DbCL v1.0.

---

## Autor

Projeto de Mestrado - Detecção de Fraude em Cartões de Crédito
Data: Dezembro/2025
