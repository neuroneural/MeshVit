#!/bin/bash
#SBATCH --job-name=cat3DWS         # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --mem=64G                      # Memory requirement (10 gigs)
#SBATCH --cpus-per-task=4              # Number of CPUs
#SBATCH --gres=gpu:RTX:1               # Request 1 GPU of type RTX
#SBATCH --time=0-00:10:00              # Maximum runtime (D-H:MM:SS, here it's set to 2 hours for example)
#SBATCH -e jobs2/%A_%a.err            # error logs (remember to create jobs folder)
#SBATCH -o jobs2/%A_%a.out              # output logs (remember to create jobs folder)
## You can fill in account and partition here
#SBATCH --account=psy53c17
#SBATCH --partition=qTRDGPU
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn005

sleep 5s

# Activate conda environment using the specific path

. /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate catmonai

BASE_SEED=12345

# Modify the base seed using the SLURM_ARRAY_ID
SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

# Now use the Python script to generate reproducible random numbers
SUBVOLUME_SIZE=128
# $(python random_seeded.py $SEED 32 64 128)
PATCH_SIZE=$(python random_seeded.py $SEED 8 16 32 64)
N_LAYERS=$(python random_seeded.py $SEED 4 8 16 32 64 128 256)
D_MODEL=$(python random_seeded.py $SEED 8 16 32 64 128 256)
D_FF=$(python random_seeded.py $SEED 8 16 32 64 128 256)
N_HEADS=$(python random_seeded.py $SEED 4 8 16 32 64 128 256)
D_ENCODER=$(python random_seeded.py $SEED 4 8 16 32 64 128 256)
# PATCH_SIZE=16
# N_LAYERS=8
# D_MODEL=8
# D_FF=8
# N_HEADS=8
# D_ENCODER=8

echo 
echo "SEED SLURM_ARRAY_TASK_ID SUBVOLUME_SIZE PATCH_SIZE N_LAYERS D_MODEL D_FF N_HEADS D_ENCODER lr"
echo $SEED $SLURM_ARRAY_TASK_ID $SUBVOLUME_SIZE $PATCH_SIZE $N_LAYERS $D_MODEL $D_FF $N_HEADS $D_ENCODER
# Call the Python script with the randomly selected hyperparameters
python cat3DVitWorkshopRTX.py \
--train_subvolumes $SUBVOLUME_SIZE \
--patch_size $PATCH_SIZE \
--n_layers $N_LAYERS \
--d_model $D_MODEL \
--d_ff $D_FF \
--n_heads $N_HEADS \
--d_encoder $D_ENCODER 

sleep 5