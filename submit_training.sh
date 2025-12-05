#!/bin/bash
#SBATCH --job-name=mamba_training
#SBATCH --partition=gpu                    # GPU partition (NVIDIA L40 nodes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                 # 8 cores (GPU nodes have 32 cores total)
#SBATCH --gres=gpu:1                      # Request 1 NVIDIA L40 GPU (48GB VRAM)
#SBATCH --mem=64G                         # GPU nodes have 192GB RAM total
#SBATCH --time=48:00:00                   # Adjust based on expected training time
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
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
# Choose the method that matches your setup:

# Option 1: If using conda/mamba environment
# conda activate lomba

# Option 2: If using the local .venv
source .venv/bin/activate

# Option 3: If using uv (project uses uv for dependencies)
# Note: You may need to install uv first or use pip within venv

# Set WandB API key (required for experiment tracking)
# Replace with your actual API key or set it in your ~/.bashrc
export WANDB_API_KEY="your_wandb_api_key_here"

# Optional: Set WandB mode
# export WANDB_MODE=offline  # Use this if no internet connection on compute nodes

# =============================================================================
# Training Configuration
# =============================================================================

PROJECT_NAME="mamba-history"
RUN_NAME="blythe_${SLURM_JOB_ID}"

# Model hyperparameters (adjust as needed)
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
