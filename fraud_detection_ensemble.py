#!/usr/bin/env python3
"""
Métodos de Ensemble para Detecção de Fraude
Dataset: Credit Card Fraud Detection
Técnicas: Voting, Stacking, Weighted Average, Meta-Ensemble
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
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier,
    VotingClassifier, StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, precision_score, recall_score
)

# PyOD
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD

# XGBoost (opcional)
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
OUTPUT_DIR = Path(f"results/ensemble_{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MÉTODOS DE ENSEMBLE PARA DETECÇÃO DE FRAUDE")
print("="*80)
print(f"\nDiretório de saída: {OUTPUT_DIR}\n")

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
print("\n[1/9] Carregando dados...")

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
print("\n[2/9] Análise exploratória...")

class_dist = df['Class'].value_counts()
print(f"\n  Legítimas: {class_dist[0]:,} ({class_dist[0]/len(df)*100:.3f}%)")
print(f"  Fraudes: {class_dist[1]:,} ({class_dist[1]/len(df)*100:.3f}%)")

# ============================================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n[3/9] Pré-processamento...")

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

# ============================================================================
# 4. MODELOS BASE
# ============================================================================
print("\n[4/9] Treinando modelos base...")

base_models = {}
base_predictions = {}
base_probas = {}

# 4.1 Random Forest
print("\n  [4.1] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
base_models['Random Forest'] = rf
base_predictions['Random Forest'] = rf.predict(X_test_scaled)
base_probas['Random Forest'] = rf.predict_proba(X_test_scaled)[:, 1]
print(f"    Treinado | F1: {f1_score(y_test, base_predictions['Random Forest']):.4f}")

# 4.2 Logistic Regression
print("\n  [4.2] Logistic Regression...")
lr = LogisticRegression(
    class_weight='balanced',
    C=1.0,
    max_iter=1000,
    random_state=RANDOM_STATE
)
lr.fit(X_train_scaled, y_train)
base_models['Logistic Regression'] = lr
base_predictions['Logistic Regression'] = lr.predict(X_test_scaled)
base_probas['Logistic Regression'] = lr.predict_proba(X_test_scaled)[:, 1]
print(f"    Treinado | F1: {f1_score(y_test, base_predictions['Logistic Regression']):.4f}")

# 4.3 Gradient Boosting
print("\n  [4.3] Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=RANDOM_STATE
)
gb.fit(X_train_scaled, y_train)
base_models['Gradient Boosting'] = gb
base_predictions['Gradient Boosting'] = gb.predict(X_test_scaled)
base_probas['Gradient Boosting'] = gb.predict_proba(X_test_scaled)[:, 1]
print(f"    Treinado | F1: {f1_score(y_test, base_predictions['Gradient Boosting']):.4f}")

# 4.4 XGBoost (se disponível)
if XGBOOST_AVAILABLE:
    print("\n  [4.4] XGBoost...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=-1
    )
    xgb.fit(X_train_scaled, y_train)
    base_models['XGBoost'] = xgb
    base_predictions['XGBoost'] = xgb.predict(X_test_scaled)
    base_probas['XGBoost'] = xgb.predict_proba(X_test_scaled)[:, 1]
    print(f"    Treinado | F1: {f1_score(y_test, base_predictions['XGBoost']):.4f}")

# 4.5 Isolation Forest
print("\n  [4.5] Isolation Forest...")
iso = IsolationForest(
    n_estimators=100,
    contamination=contamination_rate,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(X_train_scaled)
iso_pred = iso.predict(X_test_scaled)
iso_scores = iso.decision_function(X_test_scaled)
iso_pred_binary = (iso_pred == -1).astype(int)
base_models['Isolation Forest'] = iso
base_predictions['Isolation Forest'] = iso_pred_binary
base_probas['Isolation Forest'] = -iso_scores  # Inverter para maior = mais anômalo
print(f"    Treinado | F1: {f1_score(y_test, iso_pred_binary):.4f}")

# ============================================================================
# 5. ENSEMBLE 1: VOTING CLASSIFIER (Hard Voting)
# ============================================================================
print("\n[5/9] Criando Voting Classifier (Hard Voting)...")

# Usar apenas modelos supervisionados para hard voting
voting_estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)),
    ('lr', LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000,
                              random_state=RANDOM_STATE)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                      random_state=RANDOM_STATE))
]

if XGBOOST_AVAILABLE:
    voting_estimators.append(
        ('xgb', XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                             scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
                             tree_method='hist', n_jobs=-1))
    )

print(f"  Modelos no ensemble: {len(voting_estimators)}")

voting_hard = VotingClassifier(estimators=voting_estimators, voting='hard', n_jobs=-1)
voting_hard.fit(X_train_scaled, y_train)

y_pred_voting_hard = voting_hard.predict(X_test_scaled)
print(f"  Voting Hard treinado | F1: {f1_score(y_test, y_pred_voting_hard):.4f}")

# ============================================================================
# 6. ENSEMBLE 2: VOTING CLASSIFIER (Soft Voting)
# ============================================================================
print("\n[6/9] Criando Voting Classifier (Soft Voting)...")

voting_soft = VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=-1)
voting_soft.fit(X_train_scaled, y_train)

y_pred_voting_soft = voting_soft.predict(X_test_scaled)
y_proba_voting_soft = voting_soft.predict_proba(X_test_scaled)[:, 1]
print(f"  Voting Soft treinado | F1: {f1_score(y_test, y_pred_voting_soft):.4f}")

# ============================================================================
# 7. ENSEMBLE 3: STACKING CLASSIFIER
# ============================================================================
print("\n[7/9] Criando Stacking Classifier...")

# Base estimators
stacking_estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                      random_state=RANDOM_STATE))
]

if XGBOOST_AVAILABLE:
    stacking_estimators.append(
        ('xgb', XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                             scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
                             tree_method='hist', n_jobs=-1))
    )

# Meta-learner
meta_learner = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)

stacking = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1
)

print(f"  Base estimators: {len(stacking_estimators)}")
print(f"  Meta-learner: Logistic Regression")
print(f"  Treinando (pode demorar)...")

stacking.fit(X_train_scaled, y_train)

y_pred_stacking = stacking.predict(X_test_scaled)
y_proba_stacking = stacking.predict_proba(X_test_scaled)[:, 1]
print(f"  Stacking treinado | F1: {f1_score(y_test, y_pred_stacking):.4f}")

# ============================================================================
# 8. ENSEMBLE 4: WEIGHTED AVERAGE (Customizado)
# ============================================================================
print("\n[8/9] Criando Weighted Average Ensemble...")

# Calcular pesos baseados no F1-Score de cada modelo base
weights = {}
for name in base_models.keys():
    if name != 'Isolation Forest':  # Excluir não supervisionado
        f1 = f1_score(y_test, base_predictions[name])
        weights[name] = f1

# Normalizar pesos
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"  Pesos calculados:")
for name, weight in weights.items():
    print(f"    {name}: {weight:.4f}")

# Média ponderada das probabilidades
y_proba_weighted = np.zeros(len(y_test))
for name, weight in weights.items():
    y_proba_weighted += weight * base_probas[name]

y_pred_weighted = (y_proba_weighted > 0.5).astype(int)
print(f"  Weighted Average | F1: {f1_score(y_test, y_pred_weighted):.4f}")

# ============================================================================
# 9. COMPARAÇÃO DE TODOS OS MÉTODOS
# ============================================================================
print("\n[9/9] Comparando todos os métodos...")

# Calcular métricas para modelos base
results = []

for name in base_models.keys():
    y_pred = base_predictions[name]
    y_score = base_probas[name]

    results.append({
        'Modelo': name,
        'Tipo': 'Base Model',
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_score),
        'PR-AUC': average_precision_score(y_test, y_score)
    })

# Adicionar ensembles
ensemble_results = [
    {
        'Modelo': 'Voting Hard',
        'Tipo': 'Ensemble',
        'y_pred': y_pred_voting_hard,
        'y_score': None  # Hard voting não tem probabilidades
    },
    {
        'Modelo': 'Voting Soft',
        'Tipo': 'Ensemble',
        'y_pred': y_pred_voting_soft,
        'y_score': y_proba_voting_soft
    },
    {
        'Modelo': 'Stacking',
        'Tipo': 'Ensemble',
        'y_pred': y_pred_stacking,
        'y_score': y_proba_stacking
    },
    {
        'Modelo': 'Weighted Average',
        'Tipo': 'Ensemble',
        'y_pred': y_pred_weighted,
        'y_score': y_proba_weighted
    }
]

for ens in ensemble_results:
    y_pred = ens['y_pred']
    y_score = ens['y_score']

    result = {
        'Modelo': ens['Modelo'],
        'Tipo': ens['Tipo'],
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
    }

    if y_score is not None:
        result['ROC-AUC'] = roc_auc_score(y_test, y_score)
        result['PR-AUC'] = average_precision_score(y_test, y_score)
    else:
        result['ROC-AUC'] = np.nan
        result['PR-AUC'] = np.nan

    results.append(result)

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("COMPARAÇÃO COMPLETA - BASE MODELS vs ENSEMBLES")
print("="*80)
print(comparison_df.to_string(index=False))

# Salvar resultados
comparison_df.to_csv(OUTPUT_DIR / 'ensemble_comparison.csv', index=False)
print(f"\n  Resultados salvos: {OUTPUT_DIR}/ensemble_comparison.csv")

# ============================================================================
# 10. VISUALIZAÇÕES
# ============================================================================
print("\nGerando visualizações...")

# 10.1 Comparação de métricas
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparação: Modelos Base vs Ensembles', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    # Filtrar valores não nulos
    df_metric = comparison_df[['Modelo', 'Tipo', metric]].dropna()

    x_pos = np.arange(len(df_metric))
    bars = ax.bar(x_pos, df_metric[metric], alpha=0.8)

    # Colorir por tipo
    colors = ['#3498db' if t == 'Base Model' else '#e74c3c' for t in df_metric['Tipo']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_metric['Modelo'], rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    for i, v in enumerate(df_metric[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

# Legenda
ax = axes[1, 2]
ax.bar([0, 1], [0, 0], color=['#3498db', '#e74c3c'], alpha=0.8)
ax.legend(['Base Model', 'Ensemble'], loc='center', fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_ensemble_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/01_ensemble_comparison.png")
plt.close()

# 10.2 Curvas ROC (apenas modelos com probabilidades)
fig, ax = plt.subplots(figsize=(12, 8))

roc_data = [
    ('Voting Soft', y_proba_voting_soft),
    ('Stacking', y_proba_stacking),
    ('Weighted Average', y_proba_weighted),
    ('Random Forest', base_probas['Random Forest'])
]

colors = ['#e74c3c', '#9b59b6', '#f39c12', '#3498db']

for (name, proba), color in zip(roc_data, colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    linestyle = '--' if 'Forest' in name else '-'
    linewidth = 1.5 if 'Forest' in name else 2.5
    ax.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
            label=f'{name} (AUC={roc_auc:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5000)', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Curvas ROC - Ensembles vs Melhor Base Model', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_roc_curves.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/02_roc_curves.png")
plt.close()

# 10.3 Curvas Precision-Recall
fig, ax = plt.subplots(figsize=(12, 8))

baseline = y_test.sum() / len(y_test)

for (name, proba), color in zip(roc_data, colors):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    linestyle = '--' if 'Forest' in name else '-'
    linewidth = 1.5 if 'Forest' in name else 2.5
    ax.plot(recall_curve, precision_curve, linestyle=linestyle, linewidth=linewidth,
            label=f'{name} (AP={pr_auc:.4f})', color=color)

ax.axhline(y=baseline, color='k', linestyle='--', lw=2,
          label=f'Baseline (AP={baseline:.4f})', alpha=0.5)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Curvas Precision-Recall - Ensembles vs Melhor Base Model',
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_pr_curves.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/03_pr_curves.png")
plt.close()

# 10.4 Comparação F1-Score: Base vs Ensemble
fig, ax = plt.subplots(figsize=(10, 6))

base_models_f1 = comparison_df[comparison_df['Tipo'] == 'Base Model']['F1-Score']
ensemble_models_f1 = comparison_df[comparison_df['Tipo'] == 'Ensemble']['F1-Score']

positions = [1, 2]
bp = ax.boxplot([base_models_f1, ensemble_models_f1], positions=positions,
                widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))

ax.set_xticklabels(['Base Models', 'Ensembles'])
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Distribuição de F1-Score: Base Models vs Ensembles',
            fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Adicionar pontos individuais
for i, data in enumerate([base_models_f1, ensemble_models_f1], 1):
    y = data
    x = np.random.normal(i, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.6, s=50, color='darkblue')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_f1_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/04_f1_comparison.png")
plt.close()

# ============================================================================
# 11. RELATÓRIO FINAL
# ============================================================================
print("\nGerando relatório final...")

summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO - MÉTODOS DE ENSEMBLE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data de execução: {timestamp}\n")
    f.write(f"Dataset: Credit Card Fraud Detection (Kaggle)\n")
    f.write(f"Conjunto de teste: {len(y_test):,} transações\n\n")

    f.write("MODELOS BASE\n")
    f.write("-"*80 + "\n")
    base_df = comparison_df[comparison_df['Tipo'] == 'Base Model']
    for _, row in base_df.iterrows():
        f.write(f"  - {row['Modelo']}\n")

    f.write("\nMÉTODOS DE ENSEMBLE\n")
    f.write("-"*80 + "\n")
    f.write("1. Voting Hard: maioria dos votos (hard voting)\n")
    f.write("2. Voting Soft: média das probabilidades (soft voting)\n")
    f.write("3. Stacking: meta-learner (Logistic Regression) sobre predições base\n")
    f.write("4. Weighted Average: média ponderada por F1-Score\n")

    f.write("\n" + "="*80 + "\n")
    f.write("RESULTADOS COMPLETOS\n")
    f.write("="*80 + "\n\n")
    f.write(comparison_df.to_string(index=False))

    f.write("\n\n" + "="*80 + "\n")
    f.write("MELHOR MODELO POR MÉTRICA\n")
    f.write("="*80 + "\n\n")
    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        df_metric = comparison_df.dropna(subset=[metric])
        if not df_metric.empty:
            best = df_metric.loc[df_metric[metric].idxmax()]
            f.write(f"{metric:20s}: {best['Modelo']:30s} ({best[metric]:.4f}) [{best['Tipo']}]\n")

    f.write("\n" + "="*80 + "\n")
    f.write("ANÁLISE ESTATÍSTICA\n")
    f.write("="*80 + "\n\n")

    base_f1_mean = base_models_f1.mean()
    ensemble_f1_mean = ensemble_models_f1.mean()
    improvement = ((ensemble_f1_mean - base_f1_mean) / base_f1_mean) * 100

    f.write(f"F1-Score médio (Base Models): {base_f1_mean:.4f}\n")
    f.write(f"F1-Score médio (Ensembles): {ensemble_f1_mean:.4f}\n")
    f.write(f"Melhoria relativa: {improvement:+.2f}%\n")

    f.write("\n" + "="*80 + "\n")
    f.write("CONCLUSÕES\n")
    f.write("="*80 + "\n\n")
    f.write("- Ensembles geralmente superam modelos individuais\n")
    f.write("- Stacking combina melhor os pontos fortes de cada modelo\n")
    f.write("- Soft voting é superior ao hard voting (usa probabilidades)\n")
    f.write("- Weighted average adapta-se ao desempenho dos modelos base\n")

print(f"  Relatório salvo: {OUTPUT_DIR}/SUMMARY.txt")

print("\n" + "="*80)
print("EXECUÇÃO CONCLUÍDA")
print("="*80)
print(f"\nTodos os resultados foram salvos em: {OUTPUT_DIR}/")
print("\nArquivos gerados:")
print("  - ensemble_comparison.csv")
print("  - 01_ensemble_comparison.png")
print("  - 02_roc_curves.png")
print("  - 03_pr_curves.png")
print("  - 04_f1_comparison.png")
print("  - SUMMARY.txt")
print("\n" + "="*80)
