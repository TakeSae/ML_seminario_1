# Detecção de Fraude - Comparação de Algoritmos de Machine Learning

## Resumo dos Resultados

**Melhor Modelo:** Random Forest (F1-Score: 0.822)  
**Dataset:** 284,807 transações, 492 fraudes (0.173%)  
**Modelos Testados:** 8 modelos + 4 ensembles

| Modelo | F1-Score | Precision | Recall | ROC-AUC | Tempo |
|--------|----------|-----------|--------|---------|-------|
| Random Forest | 0.822 | 0.818 | 0.827 | 0.977 | 10.1s |
| XGBoost | 0.656 | 0.532 | 0.857 | 0.976 | 0.9s |
| Isolation Forest | 0.322 | 0.308 | 0.337 | 0.954 | 1.1s |

**Comparação com Estado da Arte:** 94-99.7% dos melhores resultados publicados

---

## Execução Rápida

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar comparação completa
python fraud_detection_comparison_all.py
```

**Resultados em:** `results/comparison_all/run_TIMESTAMP/`

---

## Scripts Disponíveis

### 1. Comparação Completa (RECOMENDADO)
```bash
python fraud_detection_comparison_all.py
```
- Testa 8 modelos
- Gera tabela consolidada
- Tempo: ~5 minutos

### 2. Métodos de Ensemble
```bash
python fraud_detection_ensemble.py
```
- Voting, Stacking, Weighted Average
- Tempo: ~20 minutos

### 3. Deep Learning
```bash
python fraud_detection_deep_learning.py
```
- Autoencoder, MLP Deep, MLP Wide
- Tempo: ~15-30 minutos

### 4. Baseline (Isolation Forest)
```bash
python isolation_forest_fraud_detection.py
```
- Isolation Forest + Baselines
- Tempo: ~3-5 minutos

---

## Dependências

```
Python >= 3.10
numpy >= 1.26.0
pandas >= 2.1.0
scikit-learn >= 1.3.0
tensorflow >= 2.15.0
xgboost >= 2.0.0
matplotlib >= 3.8.0
seaborn >= 0.13.0
kagglehub >= 0.2.0
tqdm >= 4.65.0
```

---

## Dataset

**Fonte:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transações (setembro/2013)
- 492 fraudes (0.172%)
- 30 features (V1-V28 via PCA, Time, Amount)

---

## Estrutura do Projeto

```
ML_seminario_1/
├── README.md                                  # Este arquivo
├── requirements.txt                            # Dependências
├── .gitignore                                 # Arquivos ignorados
│
├── fraud_detection_comparison_all.py          # Script principal (RECOMENDADO)
├── fraud_detection_ensemble.py                # Métodos de ensemble
├── fraud_detection_deep_learning.py           # Deep Learning
├── fraud_detection_hyperparameter_tuning.py   # Otimização (não executado)
├── fraud_detection_comparison.py              # Comparação antiga (deprecado)
├── isolation_forest_fraud_detection.py        # Baseline
│
├── ANALISE_COMPARATIVA_CONSOLIDADA.md         # Análise detalhada
│
└── results/                                   # Resultados (ignorado pelo Git)
    ├── comparison_all/
    │   └── run_2025-12-07_20-23-49/           # Comparação completa
    │       ├── all_models_comparison.csv
    │       └── SUMMARY.txt
    │
    ├── baseline_2025-12-05/                   # Isolation Forest + baselines
    │   ├── comparison_results.csv
    │   ├── isolation_forest_metrics.csv
    │   └── *.png (4 gráficos)
    │
    ├── deep_learning_2025-12-05/              # Deep Learning
    │   ├── deep_learning_comparison.csv
    │   ├── SUMMARY.txt
    │   ├── *.png (6 gráficos)
    │   └── *.keras (3 modelos salvos)
    │
    └── ensemble_2025-12-07/                   # Ensemble methods
        ├── ensemble_comparison.csv
        ├── SUMMARY.txt
        └── *.png (4 gráficos)
```

---

## Resultados Detalhados

Ver arquivo [ANALISE_COMPARATIVA_CONSOLIDADA.md](ANALISE_COMPARATIVA_CONSOLIDADA.md)

**Contém:**
- Tabela completa de resultados (8 modelos)
- Análise por categoria (Não Supervisionado, Tradicional, Deep Learning)
- Resultados de ensemble (4 métodos)
- Comparação com estado da arte
- Trade-offs e cenários de uso
- Rankings por métrica
- Conclusões e recomendações

---

## Licença

Projeto acadêmico. Dataset sob licença DbCL v1.0.
