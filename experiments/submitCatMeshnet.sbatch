#!/bin/bash
#SBATCH --job-name=catMNFL         # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --mem=64G                      # Memory requirement (10 gigs)
#SBATCH --cpus-per-task=8              # Number of CPUs
#SBATCH --gres=gpu:A40:1               # Request 1 GPU of type RTX
#SBATCH --time=4-00:00:00              # Maximum runtime (D-H:MM:SS, here it's set to 2 hours for example)
#SBATCH -e jobs/%A_%a.err            # error logs (remember to create jobs folder)
#SBATCH -o jobs/%A_%a.out              # output logs (remember to create jobs folder)
## You can fill in account and partition here
#SBATCH --account=psy53c17
#SBATCH --partition=qTRDGPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

#SBATCH --exclude=arctrdgn001
sleep 5s

# Activate conda environment using the specific path

. /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate neuro

BASE_SEED=12345

# Modify the base seed using the SLURM_ARRAY_ID
SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

python catMeshnetWorkshop.py --large \
--train_subvolumes 128

sleep 5
