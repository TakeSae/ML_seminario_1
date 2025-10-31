# Detecção de Fraude com Algoritmos de Detecção de Anomalias

## Descrição do Projeto

Este projeto implementa e compara diferentes algoritmos de detecção de anomalias para identificação de fraudes em transações de cartão de crédito, com foco em métodos não supervisionados.

### Algoritmos Implementados

**Métodos Não Supervisionados (Detecção de Anomalias):**
- **Isolation Forest** - Método baseado em árvores de isolamento
- **HBOS** (Histogram-based Outlier Score) - Detecção baseada em histogramas
- **COPOD** (Copula-based Outlier Detection) - Método baseado em cópulas
- **LOF** (Local Outlier Factor) - Fator de outlier local

**Métodos Supervisionados (Baselines):**
- **Random Forest** - Ensemble de árvores de decisão
- **Logistic Regression** - Regressão logística regularizada

## Dataset

**Fonte:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Características:**
- **284.807 transações** realizadas por cartões de crédito europeus
- **Período:** Setembro de 2013 (2 dias)
- **492 fraudes** (0.172% do total)
- **Desbalanceamento:** ~578:1 (classe positiva extremamente rara)
- **30 features:**
  - V1-V28: Componentes principais (PCA) - anonimizadas por privacidade
  - Time: Segundos decorridos desde a primeira transação
  - Amount: Valor da transação em euros

**Importante:** As features V1-V28 são resultado de transformação PCA aplicada aos dados originais por questões de confidencialidade. Apenas Time e Amount não foram transformadas.

**Licença:** Database Contents License (DbCL) v1.0

**Citação:**
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
```

**Recursos Adicionais:**
- **Simulador de Transações:** Para quem deseja gerar dados sintéticos de fraude, consulte o [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html)
- **Machine Learning for Credit Card Fraud Detection:** Guia prático disponível em https://fraud-detection-handbook.github.io/

## Requisitos

### Dependências

```bash
Python >= 3.10
numpy >= 1.26.0
pandas >= 2.1.0
scikit-learn >= 1.3.0
pyod >= 1.1.0
matplotlib >= 3.8.0
seaborn >= 0.13.0
kagglehub >= 0.2.0
tqdm >= 4.65.0
```

### Instalação

1. Clone o repositório:
```bash
git clone <URL_DO_REPOSITORIO>
cd ML_seminario_1
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Executar

### Script 1: Isolation Forest (Análise Detalhada)

```bash
python isolation_forest_fraud_detection.py
```

**Saída:**
- Pipeline completo com Isolation Forest
- Comparação com Random Forest e Logistic Regression
- Resultados salvos em `results/YYYY-MM-DD_HH-MM-SS/`

**Arquivos gerados:**
- `01_distribuicao_classes.png` - Distribuição das classes
- `02_confusion_matrix.png` - Matriz de confusão
- `03_roc_pr_curves.png` - Curvas ROC e Precision-Recall
- `04_model_comparison.png` - Comparação de métricas
- `comparison_results.csv` - Métricas em formato CSV
- `isolation_forest_metrics.csv` - Métricas do Isolation Forest

**Tempo estimado:** ~3-5 minutos

---

### Script 2: Comparação Completa de Algoritmos

```bash
python fraud_detection_comparison.py
```

**Saída:**
- Comparação entre 6 algoritmos diferentes
- Análise detalhada com múltiplas visualizações
- Sumário executivo em texto
- Resultados salvos em `results/YYYY-MM-DD_HH-MM-SS/`

**Arquivos gerados:**
- `01_distribuicao_classes.png` - Distribuição das classes
- `02_comparison_metrics.png` - Comparação de métricas (barras)
- `03_roc_curves_all.png` - Curvas ROC de todos os modelos
- `04_pr_curves_all.png` - Curvas Precision-Recall
- `05_confusion_matrices.png` - Matrizes de confusão (todos)
- `06_metrics_heatmap.png` - Heatmap de métricas
- `comparison_all_models.csv` - Métricas completas em CSV
- `SUMMARY.txt` - Sumário executivo
- `report_*.txt` - Relatórios individuais por modelo

**Tempo estimado:** ~8-15 minutos (LOF é lento com dataset grande)

**Nota:** O algoritmo LOF pode demorar devido ao cálculo de vizinhanças em ~227k amostras.

## Estrutura do Projeto

```
ML_seminario_1/
├── README.md                              # Este arquivo
├── requirements.txt                        # Dependências Python
├── .gitignore                             # Arquivos ignorados pelo Git
├── isolation_forest_fraud_detection.py    # Script 1: Isolation Forest
├── fraud_detection_comparison.py          # Script 2: Comparação completa
└── results/                               # Resultados (ignorado pelo Git)
    └── YYYY-MM-DD_HH-MM-SS/              # Pasta por execução
        ├── *.png                          # Visualizações
        ├── *.csv                          # Métricas
        └── *.txt                          # Relatórios
```

## Metodologia

### Pipeline de Processamento

1. **Carregamento de Dados**
   - Download automático via kagglehub
   - Fallback para arquivo local se disponível

2. **Análise Exploratória**
   - Estatísticas descritivas
   - Análise de desbalanceamento
   - Visualização da distribuição

3. **Pré-processamento**
   - Split estratificado (80/20)
   - Normalização de features `Time` e `Amount` (StandardScaler)
   - Demais features já são componentes PCA

4. **Treinamento**
   - Taxa de contaminação baseada na proporção real de fraudes
   - Todos os modelos usam RANDOM_STATE=42 para reprodutibilidade

5. **Avaliação**
   - Métricas apropriadas para classes desbalanceadas
   - Análise de trade-offs (Precision vs Recall)

### Métricas Utilizadas

**Justificativa para Classes Desbalanceadas:**

- **ROC-AUC:** Avalia capacidade de separação entre classes
- **PR-AUC (Average Precision):** Mais informativo que ROC-AUC em desbalanceamento extremo
- **F1-Score:** Balanço entre Precision e Recall
- **Precision:** Proporção de fraudes corretamente identificadas
- **Recall (TPR):** Proporção de fraudes detectadas do total de fraudes
- **FPR:** Taxa de falsos alarmes em transações legítimas

### Hiperparâmetros

**Isolation Forest:**
- `n_estimators=100` - Número de árvores
- `contamination=taxa_real` - Baseado na proporção de fraudes (0.00173)
- `max_samples='auto'` - Tamanho da amostra por árvore

**LOF:**
- `n_neighbors=35` - Número de vizinhos para cálculo local
- `novelty=True` - Modo para predição em novos dados
- `n_jobs=-1` - Paralelização

**Random Forest:**
- `n_estimators=100`
- `class_weight='balanced'` - Compensação automática de desbalanceamento

**Logistic Regression:**
- `class_weight='balanced'`
- `max_iter=1000`

## Resultados Esperados

### Performance Típica

| Modelo                | F1-Score | Precision | Recall | ROC-AUC |
|-----------------------|----------|-----------|--------|---------|
| Random Forest         | ~0.84    | ~0.96     | ~0.74  | ~0.95   |
| Logistic Regression   | ~0.11    | ~0.06     | ~0.92  | ~0.97   |
| Isolation Forest      | ~0.32    | ~0.31     | ~0.34  | ~0.95   |
| HBOS                  | Varia    | Varia     | Varia  | ~0.90+  |
| COPOD                 | Varia    | Varia     | Varia  | ~0.90+  |
| LOF                   | Varia    | Varia     | Varia  | ~0.90+  |

### Interpretação

- **Random Forest:** Melhor F1-Score, alta precisão (poucos falsos positivos)
- **Logistic Regression:** Recall alto (detecta mais fraudes) mas muitos falsos alarmes
- **Isolation Forest:** Balanço intermediário, bom para detecção não supervisionada
- **LOF/HBOS/COPOD:** Variam conforme características locais dos dados

## Limitações e Considerações

### Limitações Técnicas

1. **Features PCA:** Dados já transformados impedem interpretabilidade
2. **Desbalanceamento Extremo:** Dificulta aprendizado de padrões de fraude
3. **LOF Escalabilidade:** Lento em datasets grandes (O(n²))
4. **Threshold Fixo:** Contamination rate fixo pode não ser ideal

### Considerações Éticas

1. **Falsos Positivos:** Impacto negativo em usuários legítimos
   - Bloqueio de transações válidas
   - Experiência do usuário prejudicada

2. **Falsos Negativos:** Custo financeiro de fraudes não detectadas
   - Prejuízo para instituição financeira
   - Possível responsabilização do cliente

3. **Trade-off:** Definir threshold baseado em custo de negócio
   - Custo de investigação manual vs custo de fraude

4. **Viés:** Possível viés em padrões de transação de grupos específicos

## Referências

### Dataset
- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE.

### Algoritmos
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 eighth ieee international conference on data mining (pp. 413-422). IEEE.
- Goldstein, M., & Dengel, A. (2012). Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. KI-2012: poster and demo track, 1029, 59-63.
- Li, Z., Zhao, Y., Botta, N., Ionescu, C., & Hu, X. (2020). COPOD: copula-based outlier detection. In 2020 IEEE International Conference on Data Mining (ICDM) (pp. 1118-1123). IEEE.
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international conference on Management of data (pp. 93-104).

### Bibliotecas
- Scikit-learn: Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- PyOD: Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. JMLR, 20(96), pp.1-7.

## Licença

Este projeto é para fins acadêmicos. O dataset utilizado segue a licença DbCL v1.0.
