"""
Script de Comparação: Dataset Original vs SMOTE Balanceado
Aplica SMOTE para balancear classes e compara performance dos modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import time

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, roc_curve
)

# Balanceamento
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# Dataset
import kagglehub

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("COMPARAÇÃO: DATASET ORIGINAL vs SMOTE BALANCEADO")
print("=" * 80)
print()

# ==============================================================================
# 1. CARREGAR DATASET
# ==============================================================================

print("[1/7] Carregando dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = Path(path) / 'creditcard.csv'
df = pd.read_csv(csv_file)

print(f"✓ Dataset carregado: {len(df):,} transações")
print(f"  - Fraudes: {(df['Class'] == 1).sum():,} ({(df['Class'] == 1).sum() / len(df) * 100:.3f}%)")
print(f"  - Legítimas: {(df['Class'] == 0).sum():,} ({(df['Class'] == 0).sum() / len(df) * 100:.3f}%)")
print(f"  - Desbalanceamento: {(df['Class'] == 0).sum() / (df['Class'] == 1).sum():.1f}:1")
print()

# Separar features e target
X = df.drop('Class', axis=1)
y = df['Class']

# ==============================================================================
# 2. DIVIDIR TREINO/TESTE
# ==============================================================================

print("[2/7] Dividindo dados (treino/teste)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"✓ Treino: {len(X_train):,} amostras")
print(f"  - Fraudes: {(y_train == 1).sum():,}")
print(f"  - Legítimas: {(y_train == 0).sum():,}")
print(f"✓ Teste: {len(X_test):,} amostras")
print(f"  - Fraudes: {(y_test == 1).sum():,}")
print(f"  - Legítimas: {(y_test == 0).sum():,}")
print()

# ==============================================================================
# 3. APLICAR SMOTE
# ==============================================================================

print("[3/7] Aplicando SMOTE para balanceamento...")
print("  Estratégia: 0.5 (50% de fraudes, ratio 2:1)")

smote = SMOTE(
    sampling_strategy=0.5,  # 50% de fraudes (2:1 ao invés de 577.9:1)
    random_state=RANDOM_STATE,
    k_neighbors=5           # Número de vizinhos para interpolação
)

start_time = time.time()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
smote_time = time.time() - start_time

print(f"✓ SMOTE aplicado em {smote_time:.2f}s")
print(f"  Dataset original (treino):")
print(f"    - Total: {len(X_train):,}")
print(f"    - Fraudes: {(y_train == 1).sum():,}")
print(f"    - Legítimas: {(y_train == 0).sum():,}")
print(f"    - Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1")
print(f"  Dataset SMOTE (treino):")
print(f"    - Total: {len(X_train_smote):,}")
print(f"    - Fraudes: {(y_train_smote == 1).sum():,}")
print(f"    - Legítimas: {(y_train_smote == 0).sum():,}")
print(f"    - Ratio: {(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.1f}:1")
print()

# ==============================================================================
# 4. DEFINIR MODELOS
# ==============================================================================

print("[4/7] Definindo modelos...")

# Pesos para dataset original
scale_pos_weight_original = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Pesos para dataset SMOTE (não precisa, já está balanceado)
scale_pos_weight_smote = len(y_train_smote[y_train_smote == 0]) / len(y_train_smote[y_train_smote == 1])

models = {
    'Random Forest': {
        'original': RandomForestClassifier(
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
        'smote': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight=None,  # Não precisa, já balanceado
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
    },
    'XGBoost': {
        'original': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight_original,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='aucpr',
            verbosity=0
        ),
        'smote': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # Já balanceado
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='aucpr',
            verbosity=0
        )
    },
    'Logistic Regression': {
        'original': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        ),
        'smote': LogisticRegression(
            class_weight=None,  # Não precisa
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
    },
    'Gradient Boosting': {
        'original': GradientBoostingClassifier(
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
        'smote': GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            verbose=0
        )
    }
}

print(f"✓ {len(models)} modelos definidos")
print()

# ==============================================================================
# 5. TREINAR E AVALIAR
# ==============================================================================

print("[5/7] Treinando e avaliando modelos...")
print()

results = []

for model_name, model_dict in models.items():
    print(f"  {model_name}:")

    # ========== ORIGINAL ==========
    print(f"    [Original] Treinando...", end=' ')
    start_time = time.time()

    if model_name == 'Gradient Boosting':
        # Balanceamento manual via sample_weight
        sample_weight = np.where(y_train == 1, scale_pos_weight_original, 1.0)
        model_dict['original'].fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model_dict['original'].fit(X_train, y_train)

    train_time_original = time.time() - start_time

    # Predição
    y_pred_original = model_dict['original'].predict(X_test)
    y_pred_proba_original = model_dict['original'].predict_proba(X_test)[:, 1]

    # Métricas
    f1_original = f1_score(y_test, y_pred_original)
    precision_original = precision_score(y_test, y_pred_original)
    recall_original = recall_score(y_test, y_pred_original)
    roc_auc_original = roc_auc_score(y_test, y_pred_proba_original)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_original)
    pr_auc_original = auc(recall_curve, precision_curve)

    print(f"F1={f1_original:.3f} ({train_time_original:.1f}s)")

    # ========== SMOTE ==========
    print(f"    [SMOTE]    Treinando...", end=' ')
    start_time = time.time()

    if model_name == 'Gradient Boosting':
        # Sem sample_weight, SMOTE já balanceou
        model_dict['smote'].fit(X_train_smote, y_train_smote)
    else:
        model_dict['smote'].fit(X_train_smote, y_train_smote)

    train_time_smote = time.time() - start_time

    # Predição
    y_pred_smote = model_dict['smote'].predict(X_test)
    y_pred_proba_smote = model_dict['smote'].predict_proba(X_test)[:, 1]

    # Métricas
    f1_smote = f1_score(y_test, y_pred_smote)
    precision_smote = precision_score(y_test, y_pred_smote)
    recall_smote = recall_score(y_test, y_pred_smote)
    roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba_smote)
    pr_auc_smote = auc(recall_curve, precision_curve)

    print(f"F1={f1_smote:.3f} ({train_time_smote:.1f}s)")

    # Calcular melhoria
    f1_improvement = ((f1_smote - f1_original) / f1_original * 100) if f1_original > 0 else 0

    # Armazenar resultados
    results.append({
        'Modelo': model_name,
        'Dataset': 'Original',
        'F1-Score': f1_original,
        'Precision': precision_original,
        'Recall': recall_original,
        'ROC-AUC': roc_auc_original,
        'PR-AUC': pr_auc_original,
        'Tempo (s)': train_time_original
    })

    results.append({
        'Modelo': model_name,
        'Dataset': 'SMOTE',
        'F1-Score': f1_smote,
        'Precision': precision_smote,
        'Recall': recall_smote,
        'ROC-AUC': roc_auc_smote,
        'PR-AUC': pr_auc_smote,
        'Tempo (s)': train_time_smote
    })

    print(f"    → Melhoria: {f1_improvement:+.1f}%")
    print()

# ==============================================================================
# 6. CRIAR DATAFRAME DE RESULTADOS
# ==============================================================================

print("[6/7] Consolidando resultados...")
results_df = pd.DataFrame(results)

# Pivot para comparação lado a lado
comparison_df = results_df.pivot_table(
    index='Modelo',
    columns='Dataset',
    values=['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC', 'Tempo (s)']
)

# Calcular melhorias
improvement_df = pd.DataFrame(index=comparison_df.index)
for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
    improvement_df[f'{metric} Δ%'] = (
        (comparison_df[metric]['SMOTE'] - comparison_df[metric]['Original']) /
        comparison_df[metric]['Original'] * 100
    )

print("✓ Resultados consolidados")
print()

# ==============================================================================
# 7. VISUALIZAÇÕES
# ==============================================================================

print("[7/7] Gerando visualizações...")

# Criar diretório
results_dir = Path('results') / f'smote_comparison_{datetime.now().strftime("%Y-%m-%d")}'
results_dir.mkdir(parents=True, exist_ok=True)

# ========== Gráfico 1: Comparação F1-Score ==========
fig, ax = plt.subplots(figsize=(12, 6))

models_list = results_df['Modelo'].unique()
x = np.arange(len(models_list))
width = 0.35

original_f1 = [results_df[(results_df['Modelo'] == m) & (results_df['Dataset'] == 'Original')]['F1-Score'].values[0] for m in models_list]
smote_f1 = [results_df[(results_df['Modelo'] == m) & (results_df['Dataset'] == 'SMOTE')]['F1-Score'].values[0] for m in models_list]

bars1 = ax.bar(x - width/2, original_f1, width, label='Original', alpha=0.8)
bars2 = ax.bar(x + width/2, smote_f1, width, label='SMOTE', alpha=0.8)

ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Comparação F1-Score: Dataset Original vs SMOTE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(results_dir / '01_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== Gráfico 2: Todas as Métricas ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparação Completa: Original vs SMOTE', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC', 'Tempo (s)']
axes_flat = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes_flat[idx]

    original_values = [results_df[(results_df['Modelo'] == m) & (results_df['Dataset'] == 'Original')][metric].values[0] for m in models_list]
    smote_values = [results_df[(results_df['Modelo'] == m) & (results_df['Dataset'] == 'SMOTE')][metric].values[0] for m in models_list]

    bars1 = ax.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
    bars2 = ax.bar(x + width/2, smote_values, width, label='SMOTE', alpha=0.8)

    ax.set_title(metric, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / '02_all_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== Gráfico 3: Heatmap de Melhorias ==========
fig, ax = plt.subplots(figsize=(10, 6))

improvement_matrix = improvement_df[['F1-Score Δ%', 'Precision Δ%', 'Recall Δ%', 'ROC-AUC Δ%', 'PR-AUC Δ%']]
improvement_matrix.columns = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']

sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Melhoria (%)'}, ax=ax, linewidths=0.5)

ax.set_title('Melhoria Percentual com SMOTE (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
ax.set_ylabel('Modelo', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / '03_improvement_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== Gráfico 4: Distribuição de Classes ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original
axes[0].bar(['Legítimas', 'Fraudes'],
            [(y_train == 0).sum(), (y_train == 1).sum()],
            color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[0].set_title('Dataset Original (Treino)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Quantidade', fontsize=11)
axes[0].text(0, (y_train == 0).sum() * 0.5, f"{(y_train == 0).sum():,}",
             ha='center', va='center', fontsize=14, fontweight='bold')
axes[0].text(1, (y_train == 1).sum() * 0.5, f"{(y_train == 1).sum():,}",
             ha='center', va='center', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, max((y_train == 0).sum(), (y_train_smote == 0).sum()) * 1.1)

# SMOTE
axes[1].bar(['Legítimas', 'Fraudes'],
            [(y_train_smote == 0).sum(), (y_train_smote == 1).sum()],
            color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[1].set_title('Dataset SMOTE (Treino)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Quantidade', fontsize=11)
axes[1].text(0, (y_train_smote == 0).sum() * 0.5, f"{(y_train_smote == 0).sum():,}",
             ha='center', va='center', fontsize=14, fontweight='bold')
axes[1].text(1, (y_train_smote == 1).sum() * 0.5, f"{(y_train_smote == 1).sum():,}",
             ha='center', va='center', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, max((y_train == 0).sum(), (y_train_smote == 0).sum()) * 1.1)

plt.suptitle('Balanceamento de Classes: Original vs SMOTE', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(results_dir / '04_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Visualizações salvas em: {results_dir}")
print()

# ==============================================================================
# 8. SALVAR RESULTADOS
# ==============================================================================

# CSV completo
results_df.to_csv(results_dir / 'results_comparison.csv', index=False)

# CSV de melhorias
improvement_df.to_csv(results_dir / 'improvements.csv')

# Relatório texto
with open(results_dir / 'RELATORIO.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RELATÓRIO: COMPARAÇÃO DATASET ORIGINAL vs SMOTE\n")
    f.write("=" * 80 + "\n\n")

    f.write("CONFIGURAÇÃO SMOTE:\n")
    f.write(f"  - Estratégia: 0.5 (50% de fraudes)\n")
    f.write(f"  - K-neighbors: 5\n")
    f.write(f"  - Tempo de execução: {smote_time:.2f}s\n\n")

    f.write("DATASET ORIGINAL (TREINO):\n")
    f.write(f"  - Total: {len(X_train):,}\n")
    f.write(f"  - Fraudes: {(y_train == 1).sum():,}\n")
    f.write(f"  - Legítimas: {(y_train == 0).sum():,}\n")
    f.write(f"  - Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.1f}:1\n\n")

    f.write("DATASET SMOTE (TREINO):\n")
    f.write(f"  - Total: {len(X_train_smote):,}\n")
    f.write(f"  - Fraudes: {(y_train_smote == 1).sum():,}\n")
    f.write(f"  - Legítimas: {(y_train_smote == 0).sum():,}\n")
    f.write(f"  - Ratio: {(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.1f}:1\n\n")

    f.write("=" * 80 + "\n")
    f.write("RESULTADOS COMPLETOS\n")
    f.write("=" * 80 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("MELHORIAS COM SMOTE (%)\n")
    f.write("=" * 80 + "\n\n")
    f.write(improvement_df.to_string())
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("RESUMO\n")
    f.write("=" * 80 + "\n\n")

    for model_name in models_list:
        original_f1 = results_df[(results_df['Modelo'] == model_name) & (results_df['Dataset'] == 'Original')]['F1-Score'].values[0]
        smote_f1 = results_df[(results_df['Modelo'] == model_name) & (results_df['Dataset'] == 'SMOTE')]['F1-Score'].values[0]
        improvement = ((smote_f1 - original_f1) / original_f1 * 100) if original_f1 > 0 else 0

        f.write(f"{model_name}:\n")
        f.write(f"  Original: F1={original_f1:.3f}\n")
        f.write(f"  SMOTE:    F1={smote_f1:.3f}\n")
        f.write(f"  Melhoria: {improvement:+.1f}%\n\n")

print("✓ Resultados salvos:")
print(f"  - {results_dir / 'results_comparison.csv'}")
print(f"  - {results_dir / 'improvements.csv'}")
print(f"  - {results_dir / 'RELATORIO.txt'}")
print()

# ==============================================================================
# 9. RESUMO FINAL
# ==============================================================================

print("=" * 80)
print("RESUMO FINAL")
print("=" * 80)
print()

print("MELHORIAS COM SMOTE:")
print()
for model_name in models_list:
    original_f1 = results_df[(results_df['Modelo'] == model_name) & (results_df['Dataset'] == 'Original')]['F1-Score'].values[0]
    smote_f1 = results_df[(results_df['Modelo'] == model_name) & (results_df['Dataset'] == 'SMOTE')]['F1-Score'].values[0]
    improvement = ((smote_f1 - original_f1) / original_f1 * 100) if original_f1 > 0 else 0

    status = "✓" if improvement > 0 else "✗"
    print(f"  {status} {model_name:25s} {original_f1:.3f} → {smote_f1:.3f} ({improvement:+.1f}%)")

print()
print(f"Resultados salvos em: {results_dir}")
print()
print("=" * 80)
