#!/bin/bash
#SBATCH --job-name=cat3DWS         
#SBATCH --nodes=1                      
#SBATCH --mem=64G                      
#SBATCH --cpus-per-task=8              
#SBATCH --gres=gpu:RTX:1               
#SBATCH --time=4-00:00:00              
#SBATCH -e jobs/%A_%a.err            
#SBATCH -o jobs/%A_%a.out              
#SBATCH --account=psy53c17
#SBATCH --partition=qTRDGPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

. /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate transunet


# Read values from the CSV file using SLURM_ARRAY_TASK_ID directly

python catTransUnet.py \
--patch_size 8 \
--vit_choice $SLURM_ARRAY_TASK_ID 
sleep 5
