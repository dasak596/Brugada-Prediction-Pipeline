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
MODEL_PATHS=(
    "./classificationLayer_based_model_ClinicalDistilBERT_512"
    "./classificationLayer_based_model_SpanishBERT_512"
    "./classificationLayer_based_model_SpanishClinicalROBERT_512"
    "./classificationLayer_based_model_BioClinicalBERT_512"  
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
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "========================================"
    echo "Processing model: $MODEL_PATH"
    echo "Working directory: $(pwd)"
    echo "Python version: $(python --version)"
    echo "========================================"
    date
    
    python TransformerSHAP_Full-withScores-BestPhrases-Definitive-ForSbatch.py --model_path "$MODEL_PATH"
    
    echo "Finished processing $MODEL_PATH"
    date
    echo "----------------------------------------"
done

echo "All model explanations completed"
