#!/bin/bash
#SBATCH --job-name=mamba_training
#SBATCH --partition=gpu                   # GPU partition (NVIDIA L40 nodes)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10                # 10 cores (GPU nodes have 32 cores total)
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1         # Request 1 NVIDIA L40 GPU (48GB VRAM)
#SBATCH --time=08:00:00                   # Adjust based on expected training time
#SBATCH --output=logs/slurm-%j.out        # %j = job ID
#SBATCH --error=logs/slurm-%j.err
# #SBATCH --account=YOUR_ACCOUNT          # TODO: Add if required

# =============================================================================
# SLURM Job Script for Mamba History Model Training on Blythe HPC
# =============================================================================
#
# This script runs the Mamba History Model training from main.py.
#
# Usage:
#   sbatch submit_training.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/slurm-<jobid>.out
#
# =============================================================================

mkdir -p logs

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

# Load required modules
# Note: Adjust module versions based on what's available on Blythe
module purge
module load python/3.11  # or python/3.13 if available
module load cuda/12.1    # NVIDIA L40 supports CUDA 12.x
module load cudnn/8.9    # cuDNN for PyTorch

# Activate virtual environment

source .venv/bin/activate

# Set WandB API key (required for experiment tracking)
export WANDB_API_KEY="00e7f4c5450fcf5969642612ae0f2a5776fecc67"

# =============================================================================
# Training Configuration
# =============================================================================

PROJECT_NAME="mamba-history"
RUN_NAME="blythe_${SLURM_JOB_ID}"

# Model hyperparameters
D_MODEL=512
N_LAYERS=6
BATCH_SIZE=32
SEQ_LEN=256
LR_PEAK=0.0001
STEPS=5000
WARMUP_STEPS=400

# =============================================================================
# Run Training
# =============================================================================

echo "Starting training..."
echo "Project: $PROJECT_NAME"
echo "Run Name: $RUN_NAME"
echo ""

python main.py train \
    --project "$PROJECT_NAME" \
    --name "$RUN_NAME" \
    --d-model $D_MODEL \
    --n-layers $N_LAYERS \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --lr-peak $LR_PEAK \
    --steps $STEPS \
    --warmup-steps $WARMUP_STEPS \
    --device cuda \
    --patience 10 \
    --seed 1337

TRAIN_EXIT_CODE=$?

# =============================================================================
# Cleanup and Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED"
fi
echo "=========================================="

exit $TRAIN_EXIT_CODE
