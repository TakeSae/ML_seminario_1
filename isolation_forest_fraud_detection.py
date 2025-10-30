#!/usr/bin/env python3
"""
Detecção de Fraude com Isolation Forest
Credit Card Fraud Detection Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("DETECÇÃO DE FRAUDE COM ISOLATION FOREST")
print("="*70)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
print("\n[1/9] Carregando dados...")

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    print("  Baixando dataset do Kaggle...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        'mlg-ulb/creditcardfraud',
        '',
    )
    print(f"  ✓ Dataset carregado: {df.shape}")
    
except Exception as e:
    print(f"  ⚠ Erro ao baixar via kagglehub: {e}")
    print("  Tentando carregar arquivo local 'creditcard.csv'...")
    df = pd.read_csv('creditcard.csv')
    print(f"  ✓ Dataset carregado: {df.shape}")

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# ============================================================================
print("\n[2/9] Análise exploratória...")

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
plt.savefig('01_distribuicao_classes.png', dpi=150, bbox_inches='tight')
print("  ✓ Gráfico salvo: 01_distribuicao_classes.png")
plt.close()

# ============================================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n[3/9] Pré-processamento...")

# Separar features e target
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

print("  ✓ Dados normalizados")

# ============================================================================
# 4. TREINAMENTO - ISOLATION FOREST
# ============================================================================
print("\n[4/9] Treinando Isolation Forest...")

contamination_rate = y_train.sum() / len(y_train)
print(f"  Taxa de contaminação: {contamination_rate:.5f}")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=contamination_rate,
    max_samples='auto',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

iso_forest.fit(X_train_scaled)
print("  ✓ Modelo treinado")

# Predições
y_train_pred = iso_forest.predict(X_train_scaled)
y_test_pred = iso_forest.predict(X_test_scaled)
y_train_scores = iso_forest.decision_function(X_train_scaled)
y_test_scores = iso_forest.decision_function(X_test_scaled)

# Converter: -1 (anomalia) -> 1, 1 (normal) -> 0
y_train_pred_binary = (y_train_pred == -1).astype(int)
y_test_pred_binary = (y_test_pred == -1).astype(int)

print(f"  Anomalias detectadas (teste): {y_test_pred_binary.sum():,}")

# ============================================================================
# 5. AVALIAÇÃO
# ============================================================================
print("\n[5/9] Avaliando modelo...")

print("\n" + "="*70)
print("RELATÓRIO DE CLASSIFICAÇÃO - ISOLATION FOREST")
print("="*70)
print(classification_report(
    y_test, y_test_pred_binary,
    target_names=['Legítima', 'Fraude'], digits=4
))

# Matriz de confusão
cm = confusion_matrix(y_test, y_test_pred_binary)
tn, fp, fn, tp = cm.ravel()

print("MATRIZ DE CONFUSÃO")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,} ⚠️")
print(f"  True Positives:  {tp:,} ✓")

# Métricas
roc_auc = roc_auc_score(y_test, -y_test_scores)
ap_score = average_precision_score(y_test, -y_test_scores)
f1 = f1_score(y_test, y_test_pred_binary)

print(f"\n  ROC-AUC: {roc_auc:.4f}")
print(f"  PR-AUC (Average Precision): {ap_score:.4f}")
print(f"  F1-Score: {f1:.4f}")

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print(f"  Recall: {tpr:.4f} ({tpr*100:.2f}% fraudes detectadas)")
print(f"  FPR: {fpr:.4f} ({fpr*100:.2f}% falsos alarmes)")

# Visualização da matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Legítima', 'Fraude'],
           yticklabels=['Legítima', 'Fraude'], ax=ax)
ax.set_title('Matriz de Confusão - Isolation Forest', fontsize=14, fontweight='bold')
ax.set_ylabel('Classe Real', fontsize=12)
ax.set_xlabel('Classe Predita', fontsize=12)
plt.tight_layout()
plt.savefig('02_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("  ✓ Gráfico salvo: 02_confusion_matrix.png")
plt.close()

# ============================================================================
# 6. CURVAS ROC E PRECISION-RECALL
# ============================================================================
print("\n[6/9] Gerando curvas de avaliação...")

fpr_curve, tpr_curve, _ = roc_curve(y_test, -y_test_scores)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, -y_test_scores)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curve
axes[0].plot(fpr_curve, tpr_curve, lw=2, label=f'ROC (AUC={roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
axes[0].fill_between(fpr_curve, tpr_curve, alpha=0.2)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve - Isolation Forest', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
baseline = y_test.sum() / len(y_test)
axes[1].plot(recall_curve, precision_curve, lw=2, label=f'PR (AP={ap_score:.4f})')
axes[1].axhline(y=baseline, color='k', lw=2, linestyle='--',
               label=f'Baseline={baseline:.4f}')
axes[1].fill_between(recall_curve, precision_curve, alpha=0.2)
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('03_roc_pr_curves.png', dpi=150, bbox_inches='tight')
print("  ✓ Gráfico salvo: 03_roc_pr_curves.png")
plt.close()

print(f"  Ganho sobre baseline: {(ap_score/baseline - 1)*100:.1f}%")

# ============================================================================
# 7. COMPARAÇÃO COM BASELINES
# ============================================================================
print("\n[7/9] Treinando modelos baseline...")

# Random Forest
print("  Treinando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Logistic Regression
print("  Treinando Logistic Regression...")
lr = LogisticRegression(
    class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000, verbose=0
)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Comparação
comparison = pd.DataFrame({
    'Modelo': ['Isolation Forest', 'Random Forest', 'Logistic Regression'],
    'F1': [
        f1_score(y_test, y_test_pred_binary),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_lr)
    ],
    'Precision': [
        precision_score(y_test, y_test_pred_binary),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_lr)
    ],
    'Recall': [
        recall_score(y_test, y_test_pred_binary),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_lr)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, -y_test_scores),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_lr)
    ],
    'PR-AUC': [
        average_precision_score(y_test, -y_test_scores),
        average_precision_score(y_test, y_proba_rf),
        average_precision_score(y_test, y_proba_lr)
    ]
})

print("\n" + "="*70)
print("COMPARAÇÃO DE MODELOS")
print("="*70)
print(comparison.to_string(index=False))

# Visualização comparativa
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(5)
width = 0.25

ax.bar(x - width, comparison.iloc[0, 1:].values, width, label='Isolation Forest', alpha=0.8)
ax.bar(x, comparison.iloc[1, 1:].values, width, label='Random Forest', alpha=0.8)
ax.bar(x + width, comparison.iloc[2, 1:].values, width, label='Logistic Regression', alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparação de Performance entre Modelos', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['F1', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC'])
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('04_model_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Gráfico salvo: 04_model_comparison.png")
plt.close()

# ============================================================================
# 8. ANÁLISE DE ERROS
# ============================================================================
print("\n[8/9] Análise de erros...")

results_df = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Label': y_test_pred_binary,
    'Anomaly_Score': y_test_scores,
    'Amount': X_test['Amount'].values
})

results_df['Error_Type'] = 'Correct'
results_df.loc[(results_df['True_Label']==0) & (results_df['Predicted_Label']==1), 'Error_Type'] = 'False Positive'
results_df.loc[(results_df['True_Label']==1) & (results_df['Predicted_Label']==0), 'Error_Type'] = 'False Negative'

fp_count = len(results_df[results_df['Error_Type']=='False Positive'])
fn_count = len(results_df[results_df['Error_Type']=='False Negative'])

print(f"  False Positives: {fp_count:,} (legítimas classificadas como fraude)")
print(f"  False Negatives: {fn_count:,} (fraudes não detectadas)")

# ============================================================================
# 9. SALVAR RESULTADOS
# ============================================================================
print("\n[9/9] Salvando resultados...")

# Salvar comparação em CSV
comparison.to_csv('comparison_results.csv', index=False)
print("  ✓ Resultados salvos: comparison_results.csv")

# Salvar métricas principais
metrics_summary = {
    'Modelo': 'Isolation Forest',
    'ROC-AUC': roc_auc,
    'PR-AUC': ap_score,
    'F1-Score': f1,
    'Precision': precision_score(y_test, y_test_pred_binary),
    'Recall': recall_score(y_test, y_test_pred_binary),
    'True Positives': tp,
    'False Positives': fp,
    'False Negatives': fn,
    'True Negatives': tn
}

metrics_df = pd.DataFrame([metrics_summary])
metrics_df.to_csv('isolation_forest_metrics.csv', index=False)
print("  ✓ Métricas salvas: isolation_forest_metrics.csv")

