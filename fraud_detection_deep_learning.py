#!/usr/bin/env python3
"""
Detecção de Fraude com Deep Learning
Dataset: Credit Card Fraud Detection
Modelos: Autoencoder, MLP (Multi-Layer Perceptron)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, precision_score, recall_score
)

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Criar diretório de saída com timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path(f"results/deep_learning_{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DETECÇÃO DE FRAUDE COM DEEP LEARNING")
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
print(f"  Gráfico salvo: {OUTPUT_DIR}/01_distribuicao_classes.png")
plt.close()

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
print(f"  Fraudes no treino: {y_train.sum()/len(y_train)*100:.3f}%")
print(f"  Fraudes no teste: {y_test.sum()/len(y_test)*100:.3f}%")

# Normalização completa (StandardScaler em todas as features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  Dados normalizados (todas as features)")

# Para o Autoencoder, vamos treinar apenas com dados normais (não-fraude)
X_train_normal = X_train_scaled[y_train == 0]
X_test_normal = X_test_scaled[y_test == 0]

print(f"  Dados normais para Autoencoder - Treino: {X_train_normal.shape}, Teste: {X_test_normal.shape}")

# ============================================================================
# 4. MODELO 1: AUTOENCODER (Detecção Não-Supervisionada)
# ============================================================================
print("\n[4/9] Treinando Autoencoder...")

input_dim = X_train_scaled.shape[1]

# Arquitetura do Autoencoder
def build_autoencoder(input_dim, encoding_dims=[20, 14, 10, 7]):
    """
    Autoencoder com arquitetura simétrica
    encoding_dims: dimensões das camadas do encoder
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = input_layer

    for dim in encoding_dims:
        encoded = layers.Dense(dim, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)

    # Decoder (invertido)
    decoded = encoded
    for dim in reversed(encoding_dims[:-1]):
        decoded = layers.Dense(dim, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)

    # Camada de saída (reconstrução)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)

    # Modelo completo
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    return autoencoder

print("  Construindo Autoencoder...")
autoencoder = build_autoencoder(input_dim)

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"  Arquitetura:")
autoencoder.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("\n  Treinando (apenas com dados normais)...")
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("  Autoencoder treinado!")

# Predições: erro de reconstrução como anomaly score
print("  Calculando erros de reconstrução...")
X_test_reconstructed = autoencoder.predict(X_test_scaled, verbose=0)
reconstruction_errors = np.mean(np.square(X_test_scaled - X_test_reconstructed), axis=1)

# Threshold baseado no percentil dos erros normais
X_train_reconstructed = autoencoder.predict(X_train_normal, verbose=0)
train_reconstruction_errors = np.mean(np.square(X_train_normal - X_train_reconstructed), axis=1)
threshold = np.percentile(train_reconstruction_errors, 95)

print(f"  Threshold (95º percentil dos dados normais): {threshold:.6f}")

# Predições binárias
y_pred_autoencoder = (reconstruction_errors > threshold).astype(int)

print(f"  Anomalias detectadas: {y_pred_autoencoder.sum():,}")

# Métricas Autoencoder
ae_f1 = f1_score(y_test, y_pred_autoencoder)
ae_precision = precision_score(y_test, y_pred_autoencoder, zero_division=0)
ae_recall = recall_score(y_test, y_pred_autoencoder)
ae_roc_auc = roc_auc_score(y_test, reconstruction_errors)
ae_pr_auc = average_precision_score(y_test, reconstruction_errors)

print(f"\n  Métricas Autoencoder:")
print(f"    F1-Score: {ae_f1:.4f}")
print(f"    Precision: {ae_precision:.4f}")
print(f"    Recall: {ae_recall:.4f}")
print(f"    ROC-AUC: {ae_roc_auc:.4f}")
print(f"    PR-AUC: {ae_pr_auc:.4f}")

# Visualizar histórico de treinamento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Autoencoder - Training History (Loss)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['mae'], label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('Autoencoder - Training History (MAE)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_autoencoder_training.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/02_autoencoder_training.png")
plt.close()

# ============================================================================
# 5. MODELO 2: MLP - Arquitetura 1 (Profunda)
# ============================================================================
print("\n[5/9] Treinando MLP - Arquitetura 1 (Deep)...")

def build_mlp_deep(input_dim):
    """MLP com arquitetura profunda"""
    model = keras.Sequential([
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
    return model

print("  Construindo MLP Deep...")
mlp_deep = build_mlp_deep(input_dim)

# Class weights para balancear
class_weight = {
    0: 1.0,
    1: (len(y_train) - y_train.sum()) / y_train.sum()
}
print(f"  Class weights: {class_weight}")

mlp_deep.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

print(f"\n  Arquitetura MLP Deep:")
mlp_deep.summary()

print("\n  Treinando MLP Deep...")
history_mlp_deep = mlp_deep.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ],
    verbose=1
)

print("  MLP Deep treinado!")

# Predições
y_proba_mlp_deep = mlp_deep.predict(X_test_scaled, verbose=0).flatten()
y_pred_mlp_deep = (y_proba_mlp_deep > 0.5).astype(int)

# Métricas
mlp_deep_f1 = f1_score(y_test, y_pred_mlp_deep)
mlp_deep_precision = precision_score(y_test, y_pred_mlp_deep)
mlp_deep_recall = recall_score(y_test, y_pred_mlp_deep)
mlp_deep_roc_auc = roc_auc_score(y_test, y_proba_mlp_deep)
mlp_deep_pr_auc = average_precision_score(y_test, y_proba_mlp_deep)

print(f"\n  Métricas MLP Deep:")
print(f"    F1-Score: {mlp_deep_f1:.4f}")
print(f"    Precision: {mlp_deep_precision:.4f}")
print(f"    Recall: {mlp_deep_recall:.4f}")
print(f"    ROC-AUC: {mlp_deep_roc_auc:.4f}")
print(f"    PR-AUC: {mlp_deep_pr_auc:.4f}")

# ============================================================================
# 6. MODELO 3: MLP - Arquitetura 2 (Wide)
# ============================================================================
print("\n[6/9] Treinando MLP - Arquitetura 2 (Wide)...")

def build_mlp_wide(input_dim):
    """MLP com arquitetura ampla"""
    model = keras.Sequential([
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
    return model

print("  Construindo MLP Wide...")
mlp_wide = build_mlp_wide(input_dim)

mlp_wide.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

print(f"\n  Arquitetura MLP Wide:")
mlp_wide.summary()

print("\n  Treinando MLP Wide...")
history_mlp_wide = mlp_wide.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ],
    verbose=1
)

print("  MLP Wide treinado!")

# Predições
y_proba_mlp_wide = mlp_wide.predict(X_test_scaled, verbose=0).flatten()
y_pred_mlp_wide = (y_proba_mlp_wide > 0.5).astype(int)

# Métricas
mlp_wide_f1 = f1_score(y_test, y_pred_mlp_wide)
mlp_wide_precision = precision_score(y_test, y_pred_mlp_wide)
mlp_wide_recall = recall_score(y_test, y_pred_mlp_wide)
mlp_wide_roc_auc = roc_auc_score(y_test, y_proba_mlp_wide)
mlp_wide_pr_auc = average_precision_score(y_test, y_proba_mlp_wide)

print(f"\n  Métricas MLP Wide:")
print(f"    F1-Score: {mlp_wide_f1:.4f}")
print(f"    Precision: {mlp_wide_precision:.4f}")
print(f"    Recall: {mlp_wide_recall:.4f}")
print(f"    ROC-AUC: {mlp_wide_roc_auc:.4f}")
print(f"    PR-AUC: {mlp_wide_pr_auc:.4f}")

# ============================================================================
# 7. COMPARAÇÃO DE TODOS OS MODELOS
# ============================================================================
print("\n[7/9] Comparando modelos...")

comparison_df = pd.DataFrame({
    'Modelo': ['Autoencoder', 'MLP Deep', 'MLP Wide'],
    'Tipo': ['Não Supervisionado', 'Supervisionado', 'Supervisionado'],
    'F1-Score': [ae_f1, mlp_deep_f1, mlp_wide_f1],
    'Precision': [ae_precision, mlp_deep_precision, mlp_wide_precision],
    'Recall': [ae_recall, mlp_deep_recall, mlp_wide_recall],
    'ROC-AUC': [ae_roc_auc, mlp_deep_roc_auc, mlp_wide_roc_auc],
    'PR-AUC': [ae_pr_auc, mlp_deep_pr_auc, mlp_wide_pr_auc]
})

comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("COMPARAÇÃO DE MODELOS DEEP LEARNING")
print("="*80)
print(comparison_df.to_string(index=False))

# Salvar resultados
comparison_df.to_csv(OUTPUT_DIR / 'deep_learning_comparison.csv', index=False)
print(f"\n  Resultados salvos: {OUTPUT_DIR}/deep_learning_comparison.csv")

# ============================================================================
# 8. VISUALIZAÇÕES
# ============================================================================
print("\n[8/9] Gerando visualizações...")

# 8.1 Comparação de métricas
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparação de Modelos Deep Learning', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    x_pos = np.arange(len(comparison_df))
    bars = ax.bar(x_pos, comparison_df[metric], alpha=0.8)

    # Colorir por tipo
    colors = ['#e67e22' if t == 'Não Supervisionado' else '#9b59b6' for t in comparison_df['Tipo']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_df['Modelo'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')

    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# Legenda
ax = axes[1, 2]
ax.bar([0, 1], [0, 0], color=['#e67e22', '#9b59b6'], alpha=0.8)
ax.legend(['Não Supervisionado', 'Supervisionado'], loc='center')
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_comparison_metrics.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/03_comparison_metrics.png")
plt.close()

# 8.2 Curvas ROC
fig, ax = plt.subplots(figsize=(12, 8))

all_predictions = [
    ('Autoencoder', reconstruction_errors),
    ('MLP Deep', y_proba_mlp_deep),
    ('MLP Wide', y_proba_mlp_wide)
]

colors = ['#e67e22', '#9b59b6', '#3498db']

for (name, scores), color in zip(all_predictions, colors):
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.4f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5000)', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Curvas ROC - Deep Learning Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_roc_curves.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/04_roc_curves.png")
plt.close()

# 8.3 Curvas Precision-Recall
fig, ax = plt.subplots(figsize=(12, 8))

baseline = y_test.sum() / len(y_test)

for (name, scores), color in zip(all_predictions, colors):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    ax.plot(recall_curve, precision_curve, linewidth=2,
            label=f'{name} (AP={pr_auc:.4f})', color=color)

ax.axhline(y=baseline, color='k', linestyle='--', lw=2,
          label=f'Baseline (AP={baseline:.4f})', alpha=0.5)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Curvas Precision-Recall - Deep Learning Models', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_pr_curves.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/05_pr_curves.png")
plt.close()

# 8.4 Matrizes de confusão
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

all_preds_binary = [
    ('Autoencoder', y_pred_autoencoder),
    ('MLP Deep', y_pred_mlp_deep),
    ('MLP Wide', y_pred_mlp_wide)
]

for idx, (name, preds) in enumerate(all_preds_binary):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Legítima', 'Fraude'],
               yticklabels=['Legítima', 'Fraude'],
               ax=axes[idx], cbar=True)
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Classe Real', fontsize=10)
    axes[idx].set_xlabel('Classe Predita', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_confusion_matrices.png', dpi=150, bbox_inches='tight')
print(f"  Gráfico salvo: {OUTPUT_DIR}/06_confusion_matrices.png")
plt.close()

# ============================================================================
# 9. RELATÓRIO FINAL
# ============================================================================
print("\n[9/9] Gerando relatório final...")

summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO - DETECÇÃO DE FRAUDE COM DEEP LEARNING\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data de execução: {timestamp}\n")
    f.write(f"Dataset: Credit Card Fraud Detection (Kaggle)\n")
    f.write(f"Total de transações: {len(df):,}\n")
    f.write(f"  - Legítimas: {class_dist[0]:,} ({class_dist[0]/len(df)*100:.3f}%)\n")
    f.write(f"  - Fraudes: {class_dist[1]:,} ({class_dist[1]/len(df)*100:.3f}%)\n")
    f.write(f"Conjunto de teste: {len(y_test):,} transações\n\n")

    f.write("MODELOS DEEP LEARNING AVALIADOS\n")
    f.write("-"*80 + "\n")
    f.write("1. Autoencoder (Não Supervisionado)\n")
    f.write("   - Arquitetura: [30, 20, 14, 10, 7] -> [10, 14, 20, 30]\n")
    f.write("   - Detecção baseada em erro de reconstrução\n\n")
    f.write("2. MLP Deep (Supervisionado)\n")
    f.write("   - Arquitetura: [128, 64, 32, 16, 1]\n")
    f.write("   - BatchNormalization + Dropout\n\n")
    f.write("3. MLP Wide (Supervisionado)\n")
    f.write("   - Arquitetura: [256, 128, 64, 1]\n")
    f.write("   - BatchNormalization + Dropout\n\n")

    f.write("="*80 + "\n")
    f.write("RESULTADOS COMPARATIVOS\n")
    f.write("="*80 + "\n\n")
    f.write(comparison_df.to_string(index=False))

    f.write("\n\n" + "="*80 + "\n")
    f.write("MELHOR MODELO POR MÉTRICA\n")
    f.write("="*80 + "\n\n")
    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        best = comparison_df.loc[comparison_df[metric].idxmax()]
        f.write(f"{metric:20s}: {best['Modelo']:25s} ({best[metric]:.4f})\n")

    f.write("\n" + "="*80 + "\n")
    f.write("OBSERVAÇÕES\n")
    f.write("="*80 + "\n\n")
    f.write("- Autoencoder: treinado apenas com transações normais\n")
    f.write("- MLP: treinados com class weights para balancear classes\n")
    f.write("- Early Stopping aplicado para evitar overfitting\n")
    f.write("- ReduceLROnPlateau para ajuste adaptativo da taxa de aprendizado\n")

print(f"  Relatório salvo: {OUTPUT_DIR}/SUMMARY.txt")

# Salvar modelos
autoencoder.save(OUTPUT_DIR / 'autoencoder_model.keras')
mlp_deep.save(OUTPUT_DIR / 'mlp_deep_model.keras')
mlp_wide.save(OUTPUT_DIR / 'mlp_wide_model.keras')
print(f"  Modelos salvos em: {OUTPUT_DIR}/")

print("\n" + "="*80)
print("EXECUÇÃO CONCLUÍDA")
print("="*80)
print(f"\nTodos os resultados foram salvos em: {OUTPUT_DIR}/")
print("\nArquivos gerados:")
print("  - deep_learning_comparison.csv")
print("  - 01_distribuicao_classes.png")
print("  - 02_autoencoder_training.png")
print("  - 03_comparison_metrics.png")
print("  - 04_roc_curves.png")
print("  - 05_pr_curves.png")
print("  - 06_confusion_matrices.png")
print("  - SUMMARY.txt")
print("  - autoencoder_model.keras")
print("  - mlp_deep_model.keras")
print("  - mlp_wide_model.keras")
print("\n" + "="*80)
