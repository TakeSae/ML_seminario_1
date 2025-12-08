#!/usr/bin/env python3
"""
Comparação Completa de Todos os Métodos de Detecção de Fraude
Dataset: Credit Card Fraud Detection
Executa todos os modelos e gera análise comparativa final
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path(f"results/comparison_all/run_{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPARAÇÃO COMPLETA - TODOS OS MÉTODOS DE DETECÇÃO DE FRAUDE")
print("="*80)
print(f"\nDiretório de saída: {OUTPUT_DIR}\n")

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================
print("[1/5] Carregando dados...")

try:
    import kagglehub
    path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
    csv_file = os.path.join(path, 'creditcard.csv')
    df = pd.read_csv(csv_file)
except Exception as e:
    df = pd.read_csv('creditcard.csv')

print(f"  Dataset: {df.shape}")
print(f"  Fraudes: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)")

# ============================================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================================
print("\n[2/5] Pré-processamento...")

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Para deep learning: normalizar todas as features
scaler_full = StandardScaler()
X_train_dl = scaler_full.fit_transform(X_train)
X_test_dl = scaler_full.transform(X_test)

contamination_rate = y_train.sum() / len(y_train)

print(f"  Treino: {X_train.shape}")
print(f"  Teste: {X_test.shape}")

# ============================================================================
# 3. TREINAMENTO DE TODOS OS MODELOS
# ============================================================================
print("\n[3/5] Treinando todos os modelos...")

results = []
training_times = {}

# -------------------------------------------------------------------------
# 3.1 ISOLATION FOREST
# -------------------------------------------------------------------------
print("\n  [3.1] Isolation Forest...")
start = time.time()
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
training_times['Isolation Forest'] = time.time() - start

results.append({
    'Modelo': 'Isolation Forest',
    'Categoria': 'Não Supervisionado',
    'F1-Score': f1_score(y_test, iso_pred_binary),
    'Precision': precision_score(y_test, iso_pred_binary, zero_division=0),
    'Recall': recall_score(y_test, iso_pred_binary),
    'ROC-AUC': roc_auc_score(y_test, -iso_scores),
    'PR-AUC': average_precision_score(y_test, -iso_scores),
    'Tempo (s)': training_times['Isolation Forest']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.2 RANDOM FOREST
# -------------------------------------------------------------------------
print("\n  [3.2] Random Forest...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
training_times['Random Forest'] = time.time() - start

results.append({
    'Modelo': 'Random Forest',
    'Categoria': 'Tradicional',
    'F1-Score': f1_score(y_test, rf_pred),
    'Precision': precision_score(y_test, rf_pred),
    'Recall': recall_score(y_test, rf_pred),
    'ROC-AUC': roc_auc_score(y_test, rf_proba),
    'PR-AUC': average_precision_score(y_test, rf_proba),
    'Tempo (s)': training_times['Random Forest']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.3 LOGISTIC REGRESSION
# -------------------------------------------------------------------------
print("\n  [3.3] Logistic Regression...")
start = time.time()
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=RANDOM_STATE
)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
training_times['Logistic Regression'] = time.time() - start

results.append({
    'Modelo': 'Logistic Regression',
    'Categoria': 'Tradicional',
    'F1-Score': f1_score(y_test, lr_pred),
    'Precision': precision_score(y_test, lr_pred),
    'Recall': recall_score(y_test, lr_pred),
    'ROC-AUC': roc_auc_score(y_test, lr_proba),
    'PR-AUC': average_precision_score(y_test, lr_proba),
    'Tempo (s)': training_times['Logistic Regression']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.4 GRADIENT BOOSTING
# -------------------------------------------------------------------------
print("\n  [3.4] Gradient Boosting...")
start = time.time()
gb = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=RANDOM_STATE
)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
training_times['Gradient Boosting'] = time.time() - start

results.append({
    'Modelo': 'Gradient Boosting',
    'Categoria': 'Tradicional',
    'F1-Score': f1_score(y_test, gb_pred),
    'Precision': precision_score(y_test, gb_pred),
    'Recall': recall_score(y_test, gb_pred),
    'ROC-AUC': roc_auc_score(y_test, gb_proba),
    'PR-AUC': average_precision_score(y_test, gb_proba),
    'Tempo (s)': training_times['Gradient Boosting']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.5 XGBOOST
# -------------------------------------------------------------------------
if XGBOOST_AVAILABLE:
    print("\n  [3.5] XGBoost...")
    start = time.time()
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    training_times['XGBoost'] = time.time() - start

    results.append({
        'Modelo': 'XGBoost',
        'Categoria': 'Tradicional',
        'F1-Score': f1_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'ROC-AUC': roc_auc_score(y_test, xgb_proba),
        'PR-AUC': average_precision_score(y_test, xgb_proba),
        'Tempo (s)': training_times['XGBoost']
    })
    print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.6 AUTOENCODER
# -------------------------------------------------------------------------
print("\n  [3.6] Autoencoder...")
start = time.time()

X_train_normal = X_train_dl[y_train == 0]
input_dim = X_train_dl.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(20, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(14, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(7, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(14, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(20, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=50, batch_size=256, validation_split=0.2,
    verbose=0
)

X_test_reconstructed = autoencoder.predict(X_test_dl, verbose=0)
reconstruction_errors = np.mean(np.square(X_test_dl - X_test_reconstructed), axis=1)

X_train_reconstructed = autoencoder.predict(X_train_normal, verbose=0)
train_errors = np.mean(np.square(X_train_normal - X_train_reconstructed), axis=1)
threshold = np.percentile(train_errors, 95)

ae_pred = (reconstruction_errors > threshold).astype(int)
training_times['Autoencoder'] = time.time() - start

results.append({
    'Modelo': 'Autoencoder',
    'Categoria': 'Deep Learning',
    'F1-Score': f1_score(y_test, ae_pred),
    'Precision': precision_score(y_test, ae_pred, zero_division=0),
    'Recall': recall_score(y_test, ae_pred),
    'ROC-AUC': roc_auc_score(y_test, reconstruction_errors),
    'PR-AUC': average_precision_score(y_test, reconstruction_errors),
    'Tempo (s)': training_times['Autoencoder']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.7 MLP DEEP
# -------------------------------------------------------------------------
print("\n  [3.7] MLP Deep...")
start = time.time()

class_weight = {0: 1.0, 1: (len(y_train) - y_train.sum()) / y_train.sum()}

mlp_deep = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

mlp_deep.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
mlp_deep.fit(
    X_train_dl, y_train,
    epochs=30, batch_size=256, validation_split=0.2,
    class_weight=class_weight, verbose=0
)

mlp_deep_proba = mlp_deep.predict(X_test_dl, verbose=0).flatten()
mlp_deep_pred = (mlp_deep_proba > 0.5).astype(int)
training_times['MLP Deep'] = time.time() - start

results.append({
    'Modelo': 'MLP Deep',
    'Categoria': 'Deep Learning',
    'F1-Score': f1_score(y_test, mlp_deep_pred),
    'Precision': precision_score(y_test, mlp_deep_pred),
    'Recall': recall_score(y_test, mlp_deep_pred),
    'ROC-AUC': roc_auc_score(y_test, mlp_deep_proba),
    'PR-AUC': average_precision_score(y_test, mlp_deep_proba),
    'Tempo (s)': training_times['MLP Deep']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# -------------------------------------------------------------------------
# 3.8 MLP WIDE
# -------------------------------------------------------------------------
print("\n  [3.8] MLP Wide...")
start = time.time()

mlp_wide = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

mlp_wide.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
mlp_wide.fit(
    X_train_dl, y_train,
    epochs=30, batch_size=256, validation_split=0.2,
    class_weight=class_weight, verbose=0
)

mlp_wide_proba = mlp_wide.predict(X_test_dl, verbose=0).flatten()
mlp_wide_pred = (mlp_wide_proba > 0.5).astype(int)
training_times['MLP Wide'] = time.time() - start

results.append({
    'Modelo': 'MLP Wide',
    'Categoria': 'Deep Learning',
    'F1-Score': f1_score(y_test, mlp_wide_pred),
    'Precision': precision_score(y_test, mlp_wide_pred),
    'Recall': recall_score(y_test, mlp_wide_pred),
    'ROC-AUC': roc_auc_score(y_test, mlp_wide_proba),
    'PR-AUC': average_precision_score(y_test, mlp_wide_proba),
    'Tempo (s)': training_times['MLP Wide']
})
print(f"    F1: {results[-1]['F1-Score']:.4f} | Tempo: {results[-1]['Tempo (s)']:.2f}s")

# ============================================================================
# 4. ANÁLISE COMPARATIVA
# ============================================================================
print("\n[4/5] Gerando análise comparativa...")

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("RESULTADOS COMPARATIVOS - TODOS OS MODELOS")
print("="*80)
print(comparison_df.to_string(index=False))

# Salvar resultados
comparison_df.to_csv(OUTPUT_DIR / 'all_models_comparison.csv', index=False)
print(f"\n  Resultados salvos: {OUTPUT_DIR}/all_models_comparison.csv")

# Estatísticas por categoria
print("\n" + "="*80)
print("ESTATÍSTICAS POR CATEGORIA")
print("="*80)
for categoria in comparison_df['Categoria'].unique():
    cat_df = comparison_df[comparison_df['Categoria'] == categoria]
    print(f"\n{categoria}:")
    print(f"  F1-Score médio: {cat_df['F1-Score'].mean():.4f}")
    print(f"  Melhor modelo: {cat_df.iloc[0]['Modelo']} (F1={cat_df.iloc[0]['F1-Score']:.4f})")
    print(f"  Tempo médio: {cat_df['Tempo (s)'].mean():.2f}s")

# Melhor por métrica
print("\n" + "="*80)
print("MELHOR MODELO POR MÉTRICA")
print("="*80)
for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
    best = comparison_df.loc[comparison_df[metric].idxmax()]
    print(f"{metric:15s}: {best['Modelo']:20s} ({best[metric]:.4f}) [{best['Categoria']}]")

# ============================================================================
# 5. RELATÓRIO FINAL
# ============================================================================
print("\n[5/5] Gerando relatório final...")

summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPARAÇÃO COMPLETA - TODOS OS MÉTODOS DE DETECÇÃO DE FRAUDE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data de execução: {timestamp}\n")
    f.write(f"Dataset: Credit Card Fraud Detection (Kaggle)\n")
    f.write(f"Total de transações: {len(df):,}\n")
    f.write(f"Fraudes: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)\n")
    f.write(f"Conjunto de teste: {len(y_test):,} transações\n\n")

    f.write("MODELOS TESTADOS\n")
    f.write("-"*80 + "\n")
    f.write("Não Supervisionado:\n  - Isolation Forest\n")
    f.write("Tradicional:\n  - Random Forest\n  - Logistic Regression\n  - Gradient Boosting\n")
    if XGBOOST_AVAILABLE:
        f.write("  - XGBoost\n")
    f.write("Deep Learning:\n  - Autoencoder\n  - MLP Deep\n  - MLP Wide\n\n")

    f.write("="*80 + "\n")
    f.write("RESULTADOS COMPLETOS\n")
    f.write("="*80 + "\n\n")
    f.write(comparison_df.to_string(index=False))

    f.write("\n\n" + "="*80 + "\n")
    f.write("ESTATÍSTICAS POR CATEGORIA\n")
    f.write("="*80 + "\n\n")
    for categoria in comparison_df['Categoria'].unique():
        cat_df = comparison_df[comparison_df['Categoria'] == categoria]
        f.write(f"{categoria}:\n")
        f.write(f"  F1-Score médio: {cat_df['F1-Score'].mean():.4f}\n")
        f.write(f"  ROC-AUC médio: {cat_df['ROC-AUC'].mean():.4f}\n")
        f.write(f"  PR-AUC médio: {cat_df['PR-AUC'].mean():.4f}\n")
        f.write(f"  Melhor modelo: {cat_df.iloc[0]['Modelo']} (F1={cat_df.iloc[0]['F1-Score']:.4f})\n")
        f.write(f"  Tempo médio: {cat_df['Tempo (s)'].mean():.2f}s\n\n")

    f.write("="*80 + "\n")
    f.write("MELHOR MODELO POR MÉTRICA\n")
    f.write("="*80 + "\n\n")
    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        best = comparison_df.loc[comparison_df[metric].idxmax()]
        f.write(f"{metric:15s}: {best['Modelo']:20s} ({best[metric]:.4f}) [{best['Categoria']}]\n")

    f.write("\n" + "="*80 + "\n")
    f.write("CONCLUSÕES\n")
    f.write("="*80 + "\n\n")
    f.write("1. Melhor modelo geral (F1-Score): " +
            f"{comparison_df.iloc[0]['Modelo']} ({comparison_df.iloc[0]['F1-Score']:.4f})\n")
    f.write("2. Melhor ROC-AUC: " +
            f"{comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]['Modelo']} " +
            f"({comparison_df['ROC-AUC'].max():.4f})\n")
    f.write("3. Melhor Precision: " +
            f"{comparison_df.loc[comparison_df['Precision'].idxmax()]['Modelo']} " +
            f"({comparison_df['Precision'].max():.4f})\n")
    f.write("4. Melhor Recall: " +
            f"{comparison_df.loc[comparison_df['Recall'].idxmax()]['Modelo']} " +
            f"({comparison_df['Recall'].max():.4f})\n")
    f.write(f"5. Modelo mais rápido: " +
            f"{comparison_df.loc[comparison_df['Tempo (s)'].idxmin()]['Modelo']} " +
            f"({comparison_df['Tempo (s)'].min():.2f}s)\n")

print(f"  Relatório salvo: {OUTPUT_DIR}/SUMMARY.txt")

print("\n" + "="*80)
print("EXECUÇÃO CONCLUÍDA")
print("="*80)
print(f"\nTodos os resultados foram salvos em: {OUTPUT_DIR}/")
print("\nArquivos gerados:")
print("  - all_models_comparison.csv")
print("  - SUMMARY.txt")
print("\n" + "="*80)
