#!/bin/bash

#SBATCH --job-name=testSeg         # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --mem=20G                      # Memory requirement (10 gigs)
#SBATCH --cpus-per-task=8              # Number of CPUs
#SBATCH --gres=gpu:V100:1               # Request 1 GPU of type RTX
#SBATCH --time=1-00:00:00              # Maximum runtime (D-H:MM:SS, here it's set to 2 hours for example)
#SBATCH -e jobs/%A.err            # error logs (remember to create jobs folder)
#SBATCH -o jobs/%A.out              # output logs (remember to create jobs folder)
## You can fill in account and partition here
#SBATCH --account=psy53c17
#SBATCH --partition=qTRDGPUH
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdgn002,arctrddgx001
sleep 5s

# Activate conda environment using the specific path

. /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate neurips

# Your commands go here:
# e.g., python my_script.py
python mongodemo3dVit.py

sleep 5s
