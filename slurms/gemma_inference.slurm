#!/bin/bash
#SBATCH --job-name=gemma_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/sfs/gpfs/tardis/home/xxn2pb/6051HackathonBestTeam/slurms/logs/gemma_inference_%j.out
#SBATCH --error=/sfs/gpfs/tardis/home/xxn2pb/6051HackathonBestTeam/slurms/logs/gemma_inference_%j.err

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"

mkdir -p ../logs

# Reset modules
module purge

# Load a specific Python version
# Change this version based on what `module spider python` shows
module load python/3.11

echo "Loaded modules:"
module list

echo "Python path:"
which python
python --version

echo "Installing requirements..."
python -m pip install --user -r requirements.txt

echo "CUDA check:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo "Starting Gemma inference..."

python gemma_inference.py

echo "End time: $(date)"
echo "Job finished."