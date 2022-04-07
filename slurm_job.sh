#!/bin/sh
#SBATCH --job-name=name-of-job # Name of the job, can be useful for identifying your job later
#SBATCH --cpus-per-task=4      # Maximum amount of CPU cores (per MPI process)
#SBATCH --mem=16G              # Maximum amount of memory (RAM)
#SBATCH --time=0-10:00         # Time limit (DD-HH:MM)
#SBATCH --nice=100             # Allow other priority jobs to go first
#SBATCH --gres=gpu:v100:1      # Number of (V100) GPUs
#SBATCH -o out.out
#SBATCH -e err.err
# Initialize conda functions and activate your environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ppinn

# Step 1 of job
srun python main.py configs/config.json

# Step 2 of job
# srun python create_images.py --in ~/scratch/project/processed &

# Step 3 of job that runs simultaneously with step 2, becasue of the &
# srun python statistics.py --in ~/scratch/project/processed &

# Wait is needed to make sure slurm doesn't quit the job when the lines with '&' immediately return
#wait