#!/usr/bin/env python3
"""
Script Completo: Todos os Modelos + Visualizações Completas
Inclui: Isolation Forest, Random Forest, XGBoost, GB, LR + Matrizes de Confusão
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import time

# Sklearn
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

# XGBoost
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = Path(f"results/complete_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ANÁLISE COMPLETA - TODOS OS MODELOS COM VISUALIZAÇÕES")
print("="*80)
print(f"\nDiretório de saída: {OUTPUT_DIR}\n")

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
print("[1/6] Carregando dados...")

try:
    import kagglehub
    path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
    csv_file = os.path.join(path, 'creditcard.csv')
    df = pd.read_csv(csv_file)
except Exception as e:
    df = pd.read_csv('creditcard.csv')

print(f"  Dataset: {df.shape}")
print(f"  Fraudes: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)")
print(f"  Legítimas: {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.3f}%)")
print(f"  Ratio: {(df['Class']==0).sum() / df['Class'].sum():.1f}:1")

# ============================================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n[2/6] Pré-processamento...")

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Treino: {len(X_train):,} | Teste: {len(X_test):,}")

# ============================================================================
# 3. DEFINIR E TREINAR MODELOS
# ============================================================================
print("\n[3/6] Treinando modelos...")

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

models = {
    'Isolation Forest': IsolationForest(
        n_estimators=100,
        contamination=0.00173,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='aucpr',
        verbosity=0
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        verbose=0
    ),
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
}

results = []
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"\n  {name}:")
    print(f"    Treinando...", end=' ')
    start_time = time.time()

    if name == 'Gradient Boosting':
        sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time

    # Predição
    if name == 'Isolation Forest':
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)
        y_score = -model.decision_function(X_test)  # Score invertido
    else:
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_score

    # Métricas
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall_curve, precision_curve)
    except:
        roc_auc = 0
        pr_auc = 0

    print(f"F1={f1:.3f} ({train_time:.1f}s)")

    results.append({
        'Modelo': name,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Tempo (s)': train_time
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("RESULTADOS:")
print("="*80)
print(results_df.to_string(index=False))

# ============================================================================
# 4. MATRIZES DE CONFUSÃO
# ============================================================================
print("\n[4/6] Gerando matrizes de confusão...")

n_models = len(models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legítima', 'Fraude'],
                yticklabels=['Legítima', 'Fraude'],
                cbar_kws={'label': 'Contagem'})

    ax.set_title(f'{name}\nF1={results_df[results_df["Modelo"]==name]["F1-Score"].values[0]:.3f}',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Classe Predita', fontweight='bold')
    ax.set_ylabel('Classe Real', fontweight='bold')

# Remover eixos extras
for idx in range(n_models, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Matrizes de Confusão - Todos os Modelos', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Matrizes de confusão salvas")

# ============================================================================
# 5. CURVAS ROC
# ============================================================================
print("\n[5/6] Gerando curvas ROC e PR...")

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, y_score in probabilities.items():
    if name == 'Isolation Forest':
        continue  # Skip Isolation Forest para ROC

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = results_df[results_df['Modelo'] == name]['ROC-AUC'].values[0]

    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Comparação de Modelos', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# PR Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, y_score in probabilities.items():
    if name == 'Isolation Forest':
        continue

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = results_df[results_df['Modelo'] == name]['PR-AUC'].values[0]

    ax.plot(recall_curve, precision_curve, linewidth=2, label=f'{name} (AUC={pr_auc:.3f})')

baseline = (y_test == 1).sum() / len(y_test)
ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=2, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curves - Comparação de Modelos', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Curvas ROC e PR salvas")

# ============================================================================
# 6. COMPARAÇÃO DE MÉTRICAS
# ============================================================================
print("\n[6/6] Gerando gráficos de comparação...")

# Gráfico de barras - Todas as métricas
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparação de Métricas - Todos os Modelos', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC', 'Tempo (s)']
axes_flat = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes_flat[idx]

    data = results_df.sort_values(metric, ascending=False)
    bars = ax.barh(data['Modelo'], data[metric], alpha=0.8)

    # Colorir a melhor barra
    bars[0].set_color('#2ecc71')

    ax.set_xlabel(metric, fontweight='bold')
    ax.set_title(f'{metric}', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # Adicionar valores
    for i, (model, value) in enumerate(zip(data['Modelo'], data[metric])):
        ax.text(value, i, f' {value:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Ranking F1-Score
fig, ax = plt.subplots(figsize=(10, 6))

data = results_df.sort_values('F1-Score', ascending=True)
colors = ['#2ecc71' if i == len(data)-1 else '#3498db' for i in range(len(data))]
bars = ax.barh(data['Modelo'], data['F1-Score'], color=colors, alpha=0.8)

ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Ranking de Modelos por F1-Score', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (model, value) in enumerate(zip(data['Modelo'], data['F1-Score'])):
    ax.text(value, i, f' {value:.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_f1_ranking.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Gráficos de comparação salvos")

# ============================================================================
# 7. SALVAR RESULTADOS
# ============================================================================

results_df.to_csv(OUTPUT_DIR / 'results_all_models.csv', index=False)

# Relatório
with open(OUTPUT_DIR / 'RELATORIO.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO: ANÁLISE COMPLETA DE MODELOS\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET:\n")
    f.write(f"  Total: {len(df):,} transações\n")
    f.write(f"  Fraudes: {(df['Class']==1).sum():,} ({(df['Class']==1).sum()/len(df)*100:.3f}%)\n")
    f.write(f"  Legítimas: {(df['Class']==0).sum():,} ({(df['Class']==0).sum()/len(df)*100:.3f}%)\n")
    f.write(f"  Desbalanceamento: {(df['Class']==0).sum() / (df['Class']==1).sum():.1f}:1\n\n")

    f.write("="*80 + "\n")
    f.write("RESULTADOS\n")
    f.write("="*80 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("="*80 + "\n")
    f.write("RANKING POR MÉTRICA\n")
    f.write("="*80 + "\n\n")

    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        best = results_df.sort_values(metric, ascending=False).iloc[0]
        f.write(f"{metric:15s}: {best['Modelo']:25s} ({best[metric]:.3f})\n")

print("\n" + "="*80)
print("ANÁLISE COMPLETA FINALIZADA")
print("="*80)
print(f"\nResultados salvos em: {OUTPUT_DIR}")
print("\nArquivos gerados:")
print("  ✓ 01_confusion_matrices.png")
print("  ✓ 02_roc_curves.png")
print("  ✓ 03_pr_curves.png")
print("  ✓ 04_metrics_comparison.png")
print("  ✓ 05_f1_ranking.png")
print("  ✓ results_all_models.csv")
print("  ✓ RELATORIO.txt")
print("\n" + "="*80)
