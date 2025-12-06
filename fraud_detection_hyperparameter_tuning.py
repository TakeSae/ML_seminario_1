#!/usr/bin/env python3
"""
Otimização de Hiperparâmetros para Detecção de Fraude
Dataset: Credit Card Fraud Detection
Métodos: GridSearchCV, RandomizedSearchCV
Modelos: Isolation Forest, Random Forest, Logistic Regression, XGBoost
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time

# Sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, precision_score, recall_score, make_scorer
)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost não instalado. Instale com: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Criar diretório de saída com timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path(f"results/hyperparameter_tuning_{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("OTIMIZAÇÃO DE HIPERPARÂMETROS - DETECÇÃO DE FRAUDE")
print("="*80)
print(f"\nDiretório de saída: {OUTPUT_DIR}\n")

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
print("\n[1/8] Carregando dados...")

try:
    import kagglehub
    print("  Baixando dataset do Kaggle...")
    path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
    print(f"  Dataset baixado para: {path}")
    csv_file = os.path.join(path, 'creditcard.csv')
    df = pd.read_csv(csv_file)
    print(f"  Dataset carregado: {df.shape}")
except Exception as e:
    print(f"  Erro ao baixar via kagglehub: {e}")
    print("  Tentando carregar arquivo local 'creditcard.csv'...")
    df = pd.read_csv('creditcard.csv')
    print(f"  Dataset carregado: {df.shape}")

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# ============================================================================
print("\n[2/8] Análise exploratória...")

class_dist = df['Class'].value_counts()
print(f"\n  Legítimas: {class_dist[0]:,} ({class_dist[0]/len(df)*100:.3f}%)")
print(f"  Fraudes: {class_dist[1]:,} ({class_dist[1]/len(df)*100:.3f}%)")

# ============================================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n[3/8] Pré-processamento...")

X = df.drop('Class', axis=1)
y = df['Class']

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Treino: {X_train.shape}")
print(f"  Teste: {X_test.shape}")

# Normalização
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

print("  Dados normalizados")

contamination_rate = y_train.sum() / len(y_train)
print(f"  Taxa de contaminação: {contamination_rate:.5f}")

# ============================================================================
# 4. OTIMIZAÇÃO: RANDOM FOREST
# ============================================================================
print("\n[4/8] Otimizando Random Forest...")

# Scorer customizado para F1 (melhor para dados desbalanceados)
f1_scorer = make_scorer(f1_score)

# Grid de hiperparâmetros
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

print(f"  Grid de parâmetros: {len(rf_param_grid['n_estimators']) * len(rf_param_grid['max_depth']) * len(rf_param_grid['min_samples_split']) * len(rf_param_grid['min_samples_leaf']) * len(rf_param_grid['class_weight'])} combinações")

# RandomizedSearchCV (mais rápido que GridSearchCV)
print("  Usando RandomizedSearchCV (30 iterações, 3-fold CV)...")
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_grid,
    n_iter=30,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE
)

start_time = time.time()
rf_search.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time

print(f"\n  Tempo de treinamento: {rf_time/60:.2f} minutos")
print(f"  Melhores parâmetros: {rf_search.best_params_}")
print(f"  Melhor F1-Score (CV): {rf_search.best_score_:.4f}")

# Avaliar no conjunto de teste
best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
y_proba_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

rf_results = {
    'F1-Score': f1_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'ROC-AUC': roc_auc_score(y_test, y_proba_rf),
    'PR-AUC': average_precision_score(y_test, y_proba_rf)
}

print(f"\n  Resultados no teste:")
for metric, value in rf_results.items():
    print(f"    {metric}: {value:.4f}")

# ============================================================================
# 5. OTIMIZAÇÃO: LOGISTIC REGRESSION
# ============================================================================
print("\n[5/8] Otimizando Logistic Regression...")

lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', None],
    'max_iter': [1000, 2000]
}

print(f"  Grid de parâmetros: {len(lr_param_grid['C']) * len(lr_param_grid['penalty']) * len(lr_param_grid['solver']) * len(lr_param_grid['class_weight']) * len(lr_param_grid['max_iter'])} combinações")

print("  Usando RandomizedSearchCV (30 iterações, 3-fold CV)...")
lr_search = RandomizedSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    lr_param_grid,
    n_iter=30,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE
)

start_time = time.time()
lr_search.fit(X_train_scaled, y_train)
lr_time = time.time() - start_time

print(f"\n  Tempo de treinamento: {lr_time/60:.2f} minutos")
print(f"  Melhores parâmetros: {lr_search.best_params_}")
print(f"  Melhor F1-Score (CV): {lr_search.best_score_:.4f}")

# Avaliar no conjunto de teste
best_lr = lr_search.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
y_proba_lr = best_lr.predict_proba(X_test_scaled)[:, 1]

lr_results = {
    'F1-Score': f1_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'ROC-AUC': roc_auc_score(y_test, y_proba_lr),
    'PR-AUC': average_precision_score(y_test, y_proba_lr)
}

print(f"\n  Resultados no teste:")
for metric, value in lr_results.items():
    print(f"    {metric}: {value:.4f}")

# ============================================================================
# 6. OTIMIZAÇÃO: XGBOOST (se disponível)
# ============================================================================
if XGBOOST_AVAILABLE:
    print("\n[6/8] Otimizando XGBoost...")

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }

    print(f"  Grid de parâmetros: muitas combinações (usando RandomizedSearch)")

    print("  Usando RandomizedSearchCV (30 iterações, 3-fold CV)...")
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, tree_method='hist', n_jobs=-1),
        xgb_param_grid,
        n_iter=30,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE
    )

    start_time = time.time()
    xgb_search.fit(X_train_scaled, y_train)
    xgb_time = time.time() - start_time

    print(f"\n  Tempo de treinamento: {xgb_time/60:.2f} minutos")
    print(f"  Melhores parâmetros: {xgb_search.best_params_}")
    print(f"  Melhor F1-Score (CV): {xgb_search.best_score_:.4f}")

    # Avaliar no conjunto de teste
    best_xgb = xgb_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test_scaled)
    y_proba_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]

    xgb_results = {
        'F1-Score': f1_score(y_test, y_pred_xgb),
        'Precision': precision_score(y_test, y_pred_xgb),
        'Recall': recall_score(y_test, y_pred_xgb),
        'ROC-AUC': roc_auc_score(y_test, y_proba_xgb),
        'PR-AUC': average_precision_score(y_test, y_proba_xgb)
    }

    print(f"\n  Resultados no teste:")
    for metric, value in xgb_results.items():
        print(f"    {metric}: {value:.4f}")
else:
    print("\n[6/8] Pulando XGBoost (não instalado)")
    xgb_results = None

# ============================================================================
# 7. COMPARAÇÃO DE MODELOS OTIMIZADOS
# ============================================================================
print("\n[7/8] Comparando modelos otimizados...")

# Montar DataFrame de comparação
comparison_data = {
    'Random Forest (Otimizado)': rf_results,
    'Logistic Regression (Otimizado)': lr_results,
}

if XGBOOST_AVAILABLE:
    comparison_data['XGBoost (Otimizado)'] = xgb_results

comparison_df = pd.DataFrame(comparison_data).T
comparison_df = comparison_df.reset_index().rename(columns={'index': 'Modelo'})
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("COMPARAÇÃO DE MODELOS OTIMIZADOS")
print("="*80)
print(comparison_df.to_string(index=False))

# Salvar resultados
comparison_df.to_csv(OUTPUT_DIR / 'optimized_models_comparison.csv', index=False)
print(f"\n  Resultados salvos: {OUTPUT_DIR}/optimized_models_comparison.csv")

# Salvar detalhes dos melhores parâmetros
best_params_df = pd.DataFrame({
    'Random Forest': [str(rf_search.best_params_)],
    'Logistic Regression': [str(lr_search.best_params_)],
})

if XGBOOST_AVAILABLE:
    best_params_df['XGBoost'] = [str(xgb_search.best_params_)]

best_params_df.to_csv(OUTPUT_DIR / 'best_hyperparameters.csv', index=False)
print(f"  Melhores parâmetros salvos: {OUTPUT_DIR}/best_hyperparameters.csv")

# ============================================================================
# 8. VISUALIZAÇÕES
# ============================================================================
print("\n[8/8] Gerando visualizações...")

# 8.1 Comparação de métricas
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparação de Modelos Otimizados', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    x_pos = np.arange(len(comparison_df))
    bars = ax.bar(x_pos, comparison_df[metric], alpha=0.8, color='#16a085')

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_df['Modelo'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# Esconder subplot extra
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_optimized_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/01_optimized_comparison.png")
plt.close()

# 8.2 Curvas ROC
fig, ax = plt.subplots(figsize=(12, 8))

all_predictions = [
    ('Random Forest', y_proba_rf),
    ('Logistic Regression', y_proba_lr),
]

if XGBOOST_AVAILABLE:
    all_predictions.append(('XGBoost', y_proba_xgb))

colors = ['#e74c3c', '#3498db', '#2ecc71']

for (name, proba), color in zip(all_predictions, colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5000)', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Curvas ROC - Modelos Otimizados', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_roc_curves.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/02_roc_curves.png")
plt.close()

# 8.3 Curvas Precision-Recall
fig, ax = plt.subplots(figsize=(12, 8))

baseline = y_test.sum() / len(y_test)

for (name, proba), color in zip(all_predictions, colors):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    ax.plot(recall_curve, precision_curve, linewidth=2,
            label=f'{name} (AP={pr_auc:.4f})', color=color)

ax.axhline(y=baseline, color='k', linestyle='--', lw=2,
          label=f'Baseline (AP={baseline:.4f})', alpha=0.5)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Curvas Precision-Recall - Modelos Otimizados', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_pr_curves.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/03_pr_curves.png")
plt.close()

# 8.4 Matrizes de confusão
n_models = len(all_predictions)
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

if n_models == 1:
    axes = [axes]

all_preds = [
    ('Random Forest', y_pred_rf),
    ('Logistic Regression', y_pred_lr),
]

if XGBOOST_AVAILABLE:
    all_preds.append(('XGBoost', y_pred_xgb))

for idx, (name, preds) in enumerate(all_preds):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
               xticklabels=['Legítima', 'Fraude'],
               yticklabels=['Legítima', 'Fraude'],
               ax=axes[idx], cbar=True)
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Classe Real', fontsize=10)
    axes[idx].set_xlabel('Classe Predita', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_confusion_matrices.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/04_confusion_matrices.png")
plt.close()

# 8.5 Histórico de busca (Random Forest)
print("  Gerando visualização do histórico de busca...")
cv_results = pd.DataFrame(rf_search.cv_results_)
cv_results = cv_results.sort_values('rank_test_score')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Random Forest - Análise de Hiperparâmetros', fontsize=14, fontweight='bold')

# Top 10 configurações
top_10 = cv_results.head(10)
axes[0, 0].barh(range(10), top_10['mean_test_score'], color='#16a085')
axes[0, 0].set_yticks(range(10))
axes[0, 0].set_yticklabels([f"Config {i+1}" for i in range(10)])
axes[0, 0].set_xlabel('F1-Score (CV)')
axes[0, 0].set_title('Top 10 Configurações')
axes[0, 0].invert_yaxis()

# Distribuição de scores
axes[0, 1].hist(cv_results['mean_test_score'], bins=20, color='#16a085', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(rf_search.best_score_, color='red', linestyle='--', linewidth=2, label='Melhor Score')
axes[0, 1].set_xlabel('F1-Score (CV)')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].set_title('Distribuição de Scores')
axes[0, 1].legend()

# Tempo de treinamento
axes[1, 0].scatter(cv_results['mean_fit_time'], cv_results['mean_test_score'], alpha=0.6, color='#16a085')
axes[1, 0].set_xlabel('Tempo de Treinamento (s)')
axes[1, 0].set_ylabel('F1-Score (CV)')
axes[1, 0].set_title('Tempo vs Performance')

# Impacto de n_estimators
if 'param_n_estimators' in cv_results.columns:
    grouped = cv_results.groupby('param_n_estimators')['mean_test_score'].mean().sort_index()
    axes[1, 1].plot(grouped.index, grouped.values, marker='o', color='#16a085', linewidth=2)
    axes[1, 1].set_xlabel('n_estimators')
    axes[1, 1].set_ylabel('F1-Score Médio (CV)')
    axes[1, 1].set_title('Impacto de n_estimators')
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/05_hyperparameter_analysis.png")
plt.close()

# ============================================================================
# 9. RELATÓRIO FINAL
# ============================================================================
print("\nGerando relatório final...")

summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO - OTIMIZAÇÃO DE HIPERPARÂMETROS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data de execução: {timestamp}\n")
    f.write(f"Dataset: Credit Card Fraud Detection (Kaggle)\n")
    f.write(f"Conjunto de teste: {len(y_test):,} transações\n\n")

    f.write("MÉTODO DE OTIMIZAÇÃO\n")
    f.write("-"*80 + "\n")
    f.write("- RandomizedSearchCV com 30 iterações\n")
    f.write("- 3-Fold Stratified Cross-Validation\n")
    f.write("- Métrica de otimização: F1-Score\n\n")

    f.write("="*80 + "\n")
    f.write("MELHORES HIPERPARÂMETROS\n")
    f.write("="*80 + "\n\n")

    f.write("Random Forest:\n")
    for param, value in rf_search.best_params_.items():
        f.write(f"  - {param}: {value}\n")

    f.write("\nLogistic Regression:\n")
    for param, value in lr_search.best_params_.items():
        f.write(f"  - {param}: {value}\n")

    if XGBOOST_AVAILABLE:
        f.write("\nXGBoost:\n")
        for param, value in xgb_search.best_params_.items():
            f.write(f"  - {param}: {value}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("RESULTADOS NO CONJUNTO DE TESTE\n")
    f.write("="*80 + "\n\n")
    f.write(comparison_df.to_string(index=False))

    f.write("\n\n" + "="*80 + "\n")
    f.write("MELHOR MODELO POR MÉTRICA\n")
    f.write("="*80 + "\n\n")
    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        best = comparison_df.loc[comparison_df[metric].idxmax()]
        f.write(f"{metric:20s}: {best['Modelo']:30s} ({best[metric]:.4f})\n")

    f.write("\n" + "="*80 + "\n")
    f.write("TEMPO DE TREINAMENTO\n")
    f.write("="*80 + "\n\n")
    f.write(f"Random Forest: {rf_time/60:.2f} minutos\n")
    f.write(f"Logistic Regression: {lr_time/60:.2f} minutos\n")
    if XGBOOST_AVAILABLE:
        f.write(f"XGBoost: {xgb_time/60:.2f} minutos\n")

print(f"  Relatório salvo: {OUTPUT_DIR}/SUMMARY.txt")

print("\n" + "="*80)
print("EXECUÇÃO CONCLUÍDA")
print("="*80)
print(f"\nTodos os resultados foram salvos em: {OUTPUT_DIR}/")
print("\nArquivos gerados:")
print("  - optimized_models_comparison.csv")
print("  - best_hyperparameters.csv")
print("  - 01_optimized_comparison.png")
print("  - 02_roc_curves.png")
print("  - 03_pr_curves.png")
print("  - 04_confusion_matrices.png")
print("  - 05_hyperparameter_analysis.png")
print("  - SUMMARY.txt")
print("\n" + "="*80)
