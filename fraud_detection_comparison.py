#!/usr/bin/env python3
"""
Comparação de Algoritmos de Detecção de Anomalias para Fraude
Dataset: Credit Card Fraud Detection
Algoritmos: Isolation Forest, HBOS, COPOD, LOF, Random Forest, Logistic Regression
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, precision_score, recall_score
)

# PyOD - Algoritmos de detecção de anomalias
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.lof import LOF

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
OUTPUT_DIR = Path(f"results/{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPARAÇÃO DE ALGORITMOS DE DETECÇÃO DE FRAUDE")
print("="*80)
print(f"\n Diretório de saída: {OUTPUT_DIR}\n")

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
    print(f"   Dataset carregado: {df.shape}")

except Exception as e:
    print(f"   Erro ao baixar via kagglehub: {e}")
    print("  Tentando carregar arquivo local 'creditcard.csv'...")
    df = pd.read_csv('creditcard.csv')
    print(f"   Dataset carregado: {df.shape}")

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# ============================================================================
print("\n[2/8] Análise exploratória...")

print(f"\n  Valores ausentes: {df.isnull().sum().sum()}")

class_dist = df['Class'].value_counts()
print(f"\n  Legítimas: {class_dist[0]:,} ({class_dist[0]/len(df)*100:.3f}%)")
print(f"  Fraudes: {class_dist[1]:,} ({class_dist[1]/len(df)*100:.3f}%)")
print(f"  Desbalanceamento: {class_dist[0]/class_dist[1]:.1f}:1")

# Visualização da distribuição
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

class_dist.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribuição de Classes', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Classe (0=Legítima, 1=Fraude)')
axes[0].set_ylabel('Contagem')
axes[0].set_xticklabels(['Legítima', 'Fraude'], rotation=0)

axes[1].pie(class_dist, labels=['Legítima', 'Fraude'],
           autopct='%1.3f%%', startangle=90,
           colors=['#2ecc71', '#e74c3c'], explode=(0, 0.1))
axes[1].set_title('Proporção de Classes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_distribuicao_classes.png', dpi=150, bbox_inches='tight')
print(f"   Gráfico salvo: {OUTPUT_DIR}/01_distribuicao_classes.png")
plt.close()

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
print(f"  Fraudes no treino: {y_train.sum()/len(y_train)*100:.3f}%")
print(f"  Fraudes no teste: {y_test.sum()/len(y_test)*100:.3f}%")

# Normalização
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(
    X_train[['Time', 'Amount']]
)
X_test_scaled[['Time', 'Amount']] = scaler.transform(
    X_test[['Time', 'Amount']]
)

print("   Dados normalizados")

contamination_rate = y_train.sum() / len(y_train)
print(f"  Taxa de contaminação: {contamination_rate:.5f}")

# ============================================================================
# 4. TREINAMENTO DOS MODELOS DE DETECÇÃO DE ANOMALIAS
# ============================================================================
print("\n[4/8] Treinando modelos de detecção de anomalias...")

models_anomaly = {}
predictions_anomaly = {}
scores_anomaly = {}

# 4.1 Isolation Forest
print("\n  [4.1] Isolation Forest...")
with tqdm(total=100, desc="      Treinando", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination_rate,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    pbar.update(20)
    iso_forest.fit(X_train_scaled)
    pbar.update(50)
    y_pred_iso = iso_forest.predict(X_test_scaled)
    y_scores_iso = iso_forest.decision_function(X_test_scaled)
    pbar.update(20)
    # Converter: -1 (anomalia) -> 1, 1 (normal) -> 0
    y_pred_iso_binary = (y_pred_iso == -1).astype(int)
    pbar.update(10)

models_anomaly['Isolation Forest'] = iso_forest
predictions_anomaly['Isolation Forest'] = y_pred_iso_binary
scores_anomaly['Isolation Forest'] = -y_scores_iso  # Inverter para maior = mais anômalo

print(f"       Treinado | Anomalias: {y_pred_iso_binary.sum():,}")

# 4.2 HBOS (Histogram-based Outlier Score)
print("\n  [4.2] HBOS (Histogram-based Outlier Score)...")
with tqdm(total=100, desc="      Treinando", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
    hbos = HBOS(contamination=contamination_rate, n_bins=20)
    pbar.update(30)
    hbos.fit(X_train_scaled)
    pbar.update(50)
    y_pred_hbos = hbos.predict(X_test_scaled)  # 0: inlier, 1: outlier
    y_scores_hbos = hbos.decision_function(X_test_scaled)
    pbar.update(20)

models_anomaly['HBOS'] = hbos
predictions_anomaly['HBOS'] = y_pred_hbos
scores_anomaly['HBOS'] = y_scores_hbos

print(f"       Treinado | Anomalias: {y_pred_hbos.sum():,}")

# 4.3 COPOD (Copula-based Outlier Detection)
print("\n  [4.3] COPOD (Copula-based Outlier Detection)...")
with tqdm(total=100, desc="      Treinando", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
    copod = COPOD(contamination=contamination_rate)
    pbar.update(30)
    copod.fit(X_train_scaled)
    pbar.update(50)
    y_pred_copod = copod.predict(X_test_scaled)
    y_scores_copod = copod.decision_function(X_test_scaled)
    pbar.update(20)

models_anomaly['COPOD'] = copod
predictions_anomaly['COPOD'] = y_pred_copod
scores_anomaly['COPOD'] = y_scores_copod

print(f"       Treinado | Anomalias: {y_pred_copod.sum():,}")

# 4.4 LOF (Local Outlier Factor)
print("\n  [4.4] LOF (Local Outlier Factor) com novelty=True...")
print("       LOF pode ser lento com datasets grandes...")
with tqdm(total=100, desc="      Treinando", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
    lof = LOF(
        n_neighbors=35,
        contamination=contamination_rate,
        novelty=True,
        n_jobs=-1  # Usar todos os cores disponíveis
    )
    pbar.update(10)
    lof.fit(X_train_scaled)
    pbar.update(70)
    y_pred_lof = lof.predict(X_test_scaled)
    pbar.update(10)
    y_scores_lof = lof.decision_function(X_test_scaled)
    pbar.update(10)

models_anomaly['LOF'] = lof
predictions_anomaly['LOF'] = y_pred_lof
scores_anomaly['LOF'] = y_scores_lof

print(f"       Treinado | Anomalias: {y_pred_lof.sum():,}")

# ============================================================================
# 5. TREINAMENTO DOS MODELOS BASELINE (SUPERVISIONADOS)
# ============================================================================
print("\n[5/8] Treinando modelos baseline supervisionados...")

models_supervised = {}
predictions_supervised = {}
scores_supervised = {}

# 5.1 Random Forest
print("\n  [5.1] Random Forest...")
with tqdm(total=100, desc="      Treinando", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    pbar.update(20)
    rf.fit(X_train_scaled, y_train)
    pbar.update(60)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    pbar.update(20)

models_supervised['Random Forest'] = rf
predictions_supervised['Random Forest'] = y_pred_rf
scores_supervised['Random Forest'] = y_proba_rf

print(f"       Treinado | Fraudes preditas: {y_pred_rf.sum():,}")

# 5.2 Logistic Regression
print("\n  [5.2] Logistic Regression...")
with tqdm(total=100, desc="      Treinando", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
    lr = LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
        verbose=0
    )
    pbar.update(20)
    lr.fit(X_train_scaled, y_train)
    pbar.update(60)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    pbar.update(20)

models_supervised['Logistic Regression'] = lr
predictions_supervised['Logistic Regression'] = y_pred_lr
scores_supervised['Logistic Regression'] = y_proba_lr

print(f"       Treinado | Fraudes preditas: {y_pred_lr.sum():,}")

# ============================================================================
# 6. AVALIAÇÃO E COMPARAÇÃO
# ============================================================================
print("\n[6/8] Avaliando e comparando modelos...")

# Combinar todos os modelos
all_models = {**models_anomaly, **models_supervised}
all_predictions = {**predictions_anomaly, **predictions_supervised}
all_scores = {**scores_anomaly, **scores_supervised}

# Calcular métricas para todos
results = []

for model_name in all_models.keys():
    y_pred = all_predictions[model_name]
    y_score = all_scores[model_name]

    # Métricas básicas
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results.append({
        'Modelo': model_name,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Tipo': 'Não Supervisionado' if model_name in models_anomaly else 'Supervisionado'
    })

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("COMPARAÇÃO DE TODOS OS MODELOS")
print("="*80)
print(comparison_df[['Modelo', 'Tipo', 'F1-Score', 'Precision', 'Recall',
                     'ROC-AUC', 'PR-AUC']].to_string(index=False))

# Salvar resultados completos
comparison_df.to_csv(OUTPUT_DIR / 'comparison_all_models.csv', index=False)
print(f"\n   Resultados salvos: {OUTPUT_DIR}/comparison_all_models.csv")

# ============================================================================
# 7. VISUALIZAÇÕES
# ============================================================================
print("\n[7/8] Gerando visualizações...")

viz_steps = ['Métricas', 'ROC', 'Precision-Recall', 'Matrizes de Confusão', 'Heatmap']
viz_pbar = tqdm(total=len(viz_steps), desc="  Progresso", bar_format='{desc}: {n}/{total} |{bar}| [{elapsed}]')

# 7.1 Comparação de métricas (barras agrupadas)
print("\n  [7.1] Gráfico de comparação de métricas...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparação Completa de Modelos', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']
colors_unsup = plt.cm.Blues(np.linspace(0.4, 0.8, len(models_anomaly)))
colors_sup = plt.cm.Greens(np.linspace(0.4, 0.8, len(models_supervised)))

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    # Separar por tipo
    unsup_data = comparison_df[comparison_df['Tipo'] == 'Não Supervisionado']
    sup_data = comparison_df[comparison_df['Tipo'] == 'Supervisionado']

    x_pos = np.arange(len(comparison_df))

    bars = ax.bar(x_pos, comparison_df[metric], alpha=0.8)

    # Colorir por tipo
    for i, (idx_orig, row) in enumerate(comparison_df.iterrows()):
        if row['Tipo'] == 'Não Supervisionado':
            bars[i].set_color('#3498db')
        else:
            bars[i].set_color('#2ecc71')

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_df['Modelo'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    # Adicionar valores no topo das barras
    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# Legenda no último subplot
ax = axes[1, 2]
ax.bar([0, 1], [0, 0], color=['#3498db', '#2ecc71'], alpha=0.8)
ax.legend(['Não Supervisionado', 'Supervisionado'], loc='center')
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_comparison_metrics.png', dpi=150, bbox_inches='tight')
print(f"       Salvo: {OUTPUT_DIR}/02_comparison_metrics.png")
plt.close()
viz_pbar.update(1)

# 7.2 Curvas ROC
print("\n  [7.2] Curvas ROC...")
fig, ax = plt.subplots(figsize=(12, 8))

for model_name, color in zip(
    list(models_anomaly.keys()) + list(models_supervised.keys()),
    list(plt.cm.tab10.colors[:len(all_models)])
):
    y_score = all_scores[model_name]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    linestyle = '--' if model_name in models_anomaly else '-'
    linewidth = 2 if model_name in models_supervised else 1.5

    ax.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
            label=f'{model_name} (AUC={roc_auc:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5000)', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Curvas ROC - Comparação de Todos os Modelos', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_roc_curves_all.png', dpi=150, bbox_inches='tight')
print(f"       Salvo: {OUTPUT_DIR}/03_roc_curves_all.png")
plt.close()
viz_pbar.update(1)

# 7.3 Curvas Precision-Recall
print("\n  [7.3] Curvas Precision-Recall...")
fig, ax = plt.subplots(figsize=(12, 8))

baseline = y_test.sum() / len(y_test)

for model_name, color in zip(
    list(models_anomaly.keys()) + list(models_supervised.keys()),
    list(plt.cm.tab10.colors[:len(all_models)])
):
    y_score = all_scores[model_name]
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    linestyle = '--' if model_name in models_anomaly else '-'
    linewidth = 2 if model_name in models_supervised else 1.5

    ax.plot(recall_curve, precision_curve, linestyle=linestyle, linewidth=linewidth,
            label=f'{model_name} (AP={pr_auc:.4f})', color=color)

ax.axhline(y=baseline, color='k', linestyle='--', lw=2,
          label=f'Baseline (AP={baseline:.4f})', alpha=0.5)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Curvas Precision-Recall - Comparação de Todos os Modelos',
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_pr_curves_all.png', dpi=150, bbox_inches='tight')
print(f"       Salvo: {OUTPUT_DIR}/04_pr_curves_all.png")
plt.close()
viz_pbar.update(1)

# 7.4 Matriz de confusão para cada modelo
print("\n  [7.4] Matrizes de confusão...")
n_models = len(all_models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, model_name in enumerate(all_models.keys()):
    y_pred = all_predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Legítima', 'Fraude'],
               yticklabels=['Legítima', 'Fraude'],
               ax=axes[idx], cbar=True)

    tipo = 'Não Supervisionado' if model_name in models_anomaly else 'Supervisionado'
    axes[idx].set_title(f'{model_name}\n({tipo})', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Classe Real', fontsize=10)
    axes[idx].set_xlabel('Classe Predita', fontsize=10)

# Esconder subplots extras
for idx in range(len(all_models), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_confusion_matrices.png', dpi=150, bbox_inches='tight')
print(f"       Salvo: {OUTPUT_DIR}/05_confusion_matrices.png")
plt.close()
viz_pbar.update(1)

# 7.5 Heatmap de comparação
print("\n  [7.5] Heatmap de métricas...")
fig, ax = plt.subplots(figsize=(10, 8))

heatmap_data = comparison_df.set_index('Modelo')[['F1-Score', 'Precision', 'Recall',
                                                    'ROC-AUC', 'PR-AUC']]

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
           center=0.5, vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})

ax.set_title('Heatmap de Métricas - Todos os Modelos', fontsize=14, fontweight='bold')
ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
ax.set_ylabel('Modelo', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_metrics_heatmap.png', dpi=150, bbox_inches='tight')
print(f"       Salvo: {OUTPUT_DIR}/06_metrics_heatmap.png")
plt.close()
viz_pbar.update(1)
viz_pbar.close()

# ============================================================================
# 8. RELATÓRIO FINAL
# ============================================================================
print("\n[8/8] Gerando relatório final...")

# Salvar relatórios individuais
for model_name in tqdm(all_models.keys(), desc="  Salvando relatórios",
                       bar_format='{desc}: {n}/{total} |{bar}| [{elapsed}]'):
    y_pred = all_predictions[model_name]
    report = classification_report(
        y_test, y_pred,
        target_names=['Legítima', 'Fraude'],
        digits=4
    )

    report_path = OUTPUT_DIR / f'report_{model_name.replace(" ", "_").lower()}.txt'
    with open(report_path, 'w') as f:
        f.write(f"RELATÓRIO DE CLASSIFICAÇÃO - {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n\nMATRIZ DE CONFUSÃO\n")
        f.write("="*80 + "\n")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        f.write(f"True Negatives:  {tn:,}\n")
        f.write(f"False Positives: {fp:,}\n")
        f.write(f"False Negatives: {fn:,}\n")
        f.write(f"True Positives:  {tp:,}\n")

print(f"   Relatórios individuais salvos em: {OUTPUT_DIR}/report_*.txt")

# Sumário executivo
summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SUMÁRIO EXECUTIVO - COMPARAÇÃO DE MODELOS DE DETECÇÃO DE FRAUDE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data de execução: {timestamp}\n")
    f.write(f"Dataset: Credit Card Fraud Detection (Kaggle)\n")
    f.write(f"Total de transações: {len(df):,}\n")
    f.write(f"  - Legítimas: {class_dist[0]:,} ({class_dist[0]/len(df)*100:.3f}%)\n")
    f.write(f"  - Fraudes: {class_dist[1]:,} ({class_dist[1]/len(df)*100:.3f}%)\n")
    f.write(f"Conjunto de teste: {len(y_test):,} transações\n\n")

    f.write("MODELOS AVALIADOS\n")
    f.write("-"*80 + "\n")
    f.write("Não Supervisionados (Detecção de Anomalias):\n")
    for model in models_anomaly.keys():
        f.write(f"  - {model}\n")
    f.write("\nSupervisionados (Classificação):\n")
    for model in models_supervised.keys():
        f.write(f"  - {model}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("RANKING POR F1-SCORE\n")
    f.write("="*80 + "\n\n")
    f.write(comparison_df[['Modelo', 'Tipo', 'F1-Score', 'Precision', 'Recall',
                          'ROC-AUC', 'PR-AUC']].to_string(index=False))

    f.write("\n\n" + "="*80 + "\n")
    f.write("MELHOR MODELO POR MÉTRICA\n")
    f.write("="*80 + "\n\n")
    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        best = comparison_df.loc[comparison_df[metric].idxmax()]
        f.write(f"{metric:20s}: {best['Modelo']:25s} ({best[metric]:.4f})\n")

    f.write("\n" + "="*80 + "\n")
    f.write("ANÁLISE DE TRADE-OFFS\n")
    f.write("="*80 + "\n\n")
    f.write("Alto Recall (detectar mais fraudes):\n")
    high_recall = comparison_df.nlargest(3, 'Recall')
    for _, row in high_recall.iterrows():
        f.write(f"  - {row['Modelo']:25s} | Recall: {row['Recall']:.4f} | "
               f"Precision: {row['Precision']:.4f} | F1: {row['F1-Score']:.4f}\n")

    f.write("\nAlta Precision (poucos falsos positivos):\n")
    high_precision = comparison_df.nlargest(3, 'Precision')
    for _, row in high_precision.iterrows():
        f.write(f"  - {row['Modelo']:25s} | Precision: {row['Precision']:.4f} | "
               f"Recall: {row['Recall']:.4f} | F1: {row['F1-Score']:.4f}\n")

    f.write("\nBalanceado (F1-Score):\n")
    high_f1 = comparison_df.nlargest(3, 'F1-Score')
    for _, row in high_f1.iterrows():
        f.write(f"  - {row['Modelo']:25s} | F1: {row['F1-Score']:.4f} | "
               f"Precision: {row['Precision']:.4f} | Recall: {row['Recall']:.4f}\n")

print(f"   Sumário executivo salvo: {OUTPUT_DIR}/SUMMARY.txt")

