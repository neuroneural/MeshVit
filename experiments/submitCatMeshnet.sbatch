#!/bin/bash
#SBATCH --job-name=catMNWS         # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --mem=64G                      # Memory requirement (10 gigs)
#SBATCH --cpus-per-task=8              # Number of CPUs
#SBATCH --gres=gpu:V100:1               # Request 1 GPU of type RTX
#SBATCH --time=4-00:00:00              # Maximum runtime (D-H:MM:SS, here it's set to 2 hours for example)
#SBATCH -e jobs/%A_%a.err            # error logs (remember to create jobs folder)
#SBATCH -o jobs/%A_%a.out              # output logs (remember to create jobs folder)
## You can fill in account and partition here
#SBATCH --account=psy53c17
#SBATCH --partition=qTRDGPUH
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

# Now use the Python script to generate reproducible random numbers
SUBVOLUME_SIZE=128
#$(python random_seeded.py $SEED 32 64 128)
PATCH_SIZE=16
#$(($SUBVOLUME_SIZE / $(python random_seeded.py $SEED 2 4 8 16 32 )))
N_LAYERS=8
#$(python random_seeded.py $SEED 8 16 32 64 128 256)
D_MODEL=64
#$(python random_seeded.py $SEED 8 16 32 64 128 256)
D_FF=128
#$(python random_seeded.py $SEED 8 16 32 64 128 256)
N_HEADS=16
#$(python random_seeded.py $SEED 8 16 32 64)
D_ENCODER=128
#$(python random_seeded.py $SEED 8 16 32 64 128 256)
lr=.001
#$(python random_seeded.py $SEED .001 .0001 .00005 .00001)

echo 
echo "SEED SLURM_ARRAY_TASK_ID SUBVOLUME_SIZE PATCH_SIZE N_LAYERS D_MODEL D_FF N_HEADS D_ENCODER lr"
echo $SEED $SLURM_ARRAY_TASK_ID $SUBVOLUME_SIZE $PATCH_SIZE $N_LAYERS $D_MODEL $D_FF $N_HEADS $D_ENCODER $lr
# Call the Python script with the randomly selected hyperparameters
python catMeshnetWorkshop.py \
  --subvolume_size $SUBVOLUME_SIZE \
  --patch_size $PATCH_SIZE \
  --n_layers $N_LAYERS \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --n_heads $N_HEADS \
  --d_encoder $D_ENCODER \
  --lr $lr

sleep 5