#!/bin/bash

#SBATCH --job-name=brugada_model
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

source myenv/bin/activate
cd "$SLURM_SUBMIT_DIR/programs"

MODEL_NAME=$1
MODEL_PATH=$2
MODEL_TYPE=$3

SAVE_BASE_DIR="../model_results"

echo "Running model: $MODEL_NAME | Type: $MODEL_TYPE"

if [ "$MODEL_TYPE" == "transformer" ]; then
    python brugadaTransformer_CLUSTER-SBATCH.py \
        --save_model_base_path "${MODEL_PATH}"
else
    tokenizer_path="fine_tuned_${MODEL_NAME}"
    python BrugadaCNN_CLUSTER-SBATCH.py \
        --save_model_path "${MODEL_PATH}" \
        --bert_model "${tokenizer_path}"
fi

# Limpieza explícita al final
python -c "import torch, gc; torch.cuda.empty_cache(); gc.collect()"
