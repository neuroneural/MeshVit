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
#SBATCH --exclude=arctrdagn005,arctrdagn014,arctrdagn015

sleep 5s

. /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate neuro

BASE_SEED=12345
SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

# Read values from the CSV file using SLURM_ARRAY_TASK_ID directly
PARAMS=$(awk -v line=$SLURM_ARRAY_TASK_ID 'NR==line' top20_128.csv)

SUBVOLUME_SIZE=$(echo $PARAMS | cut -d',' -f1)
PATCH_SIZE=$(echo $PARAMS | cut -d',' -f2)
N_LAYERS=$(echo $PARAMS | cut -d',' -f3)
D_MODEL=$(echo $PARAMS | cut -d',' -f4)
D_FF=$(echo $PARAMS | cut -d',' -f5)
N_HEADS=$(echo $PARAMS | cut -d',' -f6)
D_ENCODER=$(echo $PARAMS | cut -d',' -f7)

echo 
echo "SEED SLURM_ARRAY_TASK_ID SUBVOLUME_SIZE PATCH_SIZE N_LAYERS D_MODEL D_FF N_HEADS D_ENCODER lr"
echo $SEED $SLURM_ARRAY_TASK_ID $SUBVOLUME_SIZE $PATCH_SIZE $N_LAYERS $D_MODEL $D_FF $N_HEADS $D_ENCODER

python cat3DVitWorkshopTop20_bs1.py \
--train_subvolumes $SUBVOLUME_SIZE \
--patch_size $PATCH_SIZE \
--n_layers $N_LAYERS \
--d_model $D_MODEL \
--d_ff $D_FF \
--n_heads $N_HEADS \
--d_encoder $D_ENCODER 

sleep 5
