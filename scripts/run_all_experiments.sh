#!/bin/bash

# Script para executar todos os experimentos de detecção de fraude
# Autor: Auto-gerado
# Data: 08/12/2025

set -e  # Para em caso de erro

echo "================================================================================"
echo "EXECUÇÃO COMPLETA - DETECÇÃO DE FRAUDE EM CARTÕES DE CRÉDITO"
echo "================================================================================"
echo ""
echo "Este script irá executar todos os experimentos em sequência:"
echo "  1. Modelos Principais (RF, XGBoost, GB, LR, Isolation Forest)"
echo "  2. Deep Learning (Autoencoder, MLP Deep, MLP Wide)"
echo "  3. Ensemble (Voting, Stacking, Weighted Average)"
echo "  4. SMOTE Comparison (Original vs Balanceado)"
echo ""
echo "Tempo estimado total: ~10-15 minutos"
echo ""
read -p "Pressione ENTER para continuar ou CTRL+C para cancelar..."
echo ""

mkdir -p results
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Timestamp da execução: $TIMESTAMP"
echo ""

echo "================================================================================"
echo "[1/4] EXECUTANDO: Modelos Principais"
echo "================================================================================"
echo ""
python src/models/fraud_detection_complete_with_plots.py
echo "✓ Modelos principais executados com sucesso!"
echo ""

echo "================================================================================"
echo "[2/4] EXECUTANDO: Deep Learning"
echo "================================================================================"
echo ""
python src/deep_learning/fraud_detection_deep_learning.py
echo "✓ Deep Learning executado com sucesso!"
echo ""

echo "================================================================================"
echo "[3/4] EXECUTANDO: Ensemble Methods"
echo "================================================================================"
echo ""
python src/ensemble/fraud_detection_ensemble.py
echo "✓ Ensemble executado com sucesso!"
echo ""

echo "================================================================================"
echo "[4/4] EXECUTANDO: SMOTE Comparison"
echo "================================================================================"
echo ""
python src/models/fraud_detection_smote_comparison.py
echo "✓ SMOTE Comparison executado com sucesso!"
echo ""

echo "================================================================================"
echo "EXECUÇÃO COMPLETA FINALIZADA COM SUCESSO!"
echo "================================================================================"
echo ""
echo "Resultados salvos em: results/"
echo ""
echo "Arquivos de análise:"
echo "  - docs/ANALISE_COMPARATIVA_CONSOLIDADA.md"
echo "  - docs/APRESENTACAO_RESULTADOS.md"
echo "  - docs/DOCUMENTACAO_TECNICA_IMPLEMENTACAO.md"
echo ""
echo "================================================================================"
