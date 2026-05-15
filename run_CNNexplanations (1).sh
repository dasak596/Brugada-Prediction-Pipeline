#!/bin/bash
#SBATCH --job-name=model_explanations
#SBATCH --output=explanation_%j.out
#SBATCH --error=explanation_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# List of model paths to process sequentially
declare -A MODEL_CONFIGS=(
    [1]="cnn_based_model_finetunedBioClinicalBERT_512,fine_tuned_BioClinicalBERT,bert"
    [2]="cnn_based_model_finetunedClinicalDistilBERT_512,fine_tuned_ClinicalDistilBERT,distilbert"
    [3]="cnn_based_model_finetunedSpanishBERT_512,fine_tuned_SpanishBERT,bert"
    [4]="cnn_based_model_finetunedSpanishClinicalROBERT_512,fine_tuned_SpanishClinicalROBERT,roberta"
)

# Get working directory (where sbatch was run from)
WORK_DIR=$SLURM_SUBMIT_DIR

# Activate virtual environment
echo "Activating virtual environment from $WORK_DIR/myenv/bin/activate"
source $WORK_DIR/myenv/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Verify Python path
echo "Using Python from: $(which python)"

# Move to programs subdirectory
cd $WORK_DIR/programs

# Run sequentially for each model path
for run_id in {1..4}; do
    IFS=',' read -r model_path tokenizer model_type <<< "${MODEL_CONFIGS[$run_id]}"
    echo "========================================"
    echo "Processing model: $model_path"
    echo "Working directory: $(pwd)"
    echo "Python version: $(python --version)"
    echo "========================================"
    date
    
    python brugadaCNN_GradientDescent_SBATCH.py \
        --model_path "$model_path" \
        --tokenizer "$tokenizer" \
        --bert_type "$model_type"
    
    echo "Finished processing $model_path"
    date
    echo "----------------------------------------"
done

echo "All model explanations completed"