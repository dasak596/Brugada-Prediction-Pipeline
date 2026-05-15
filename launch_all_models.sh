#!/bin/bash

declare -A MODEL_CONFIGS=(
    [1]="BioClinicalBERT,classificationLayer_based_model_BioClinicalBERT_512,transformer"
    [2]="ClinicalDistilBERT,classificationLayer_based_model_ClinicalDistilBERT_512,transformer"
    [3]="SpanishBERT,classificationLayer_based_model_SpanishBERT_512,transformer"
    [4]="SpanishClinicalROBERT,classificationLayer_based_model_SpanishClinicalROBERT_512,transformer"
    [5]="BioClinicalBERT,cnn_based_model_finetunedBioClinicalBERT_512,cnn"
    [6]="ClinicalDistilBERT,cnn_based_model_finetunedClinicalDistilBERT_512,cnn"
    [7]="SpanishBERT,cnn_based_model_finetunedSpanishBERT_512,cnn"
    [8]="SpanishClinicalROBERT,cnn_based_model_finetunedSpanishClinicalROBERT_512,cnn"
)

for run_id in {1..8}; do
    IFS=',' read -r model_name model_path model_type <<< "${MODEL_CONFIGS[$run_id]}"
    
    echo "Submitting job for $model_name ($model_type)"
    sbatch run_model.sh "$model_name" "$model_path" "$model_type"
    
    echo "Sleeping 10 seconds before next submission..."
    sleep 10  # para evitar sobrecargar SLURM
done