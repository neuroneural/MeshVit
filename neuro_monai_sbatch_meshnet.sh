#!/bin/bash
#SBATCH -J meshnet
#SBATCH -o /data/users2/bbaker/projects/MeshVit/slurm//%j.out
#SBATCH -e /data/users2/bbaker/projects/MeshVit/slurm//%j.err
#SBATCH --nodes=1
#SBATCH -c 10
#SBATCH --mem 124G
#SBATCH --gres=gpu:v100:2
#SBATCH -p qTRDGPUH,qTRDGPUM,qTRDGPUL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kwang26@gsu.edu
#SBATCH --oversubscribe
#SBATCH -t 7200
eval "$(conda shell.bash hook)"
conda activate neuro
cd /data/users2/bbaker/projects/MeshVit/neuro2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/bin/miniconda3/lib
size=128
tsv=128
isv=32
PYTHONPATH=.. python training/minimal_monai_torchio_example.py --parallel True --model meshnet --sv_w ${size} --sv_h ${size} --sv_d ${size} --train_subvolumes ${tsv} --infer_subvolumes ${isv} --n_epochs 200 --train_path ./data/afedorov_T1_c_atlas_data/dataset_train_limited.csv --validation_path ./data/afedorov_T1_c_atlas_data/dataset_valid_limited.csv --inference_path ./data/afedorov_T1_c_atlas_data/dataset_infer_limited.csv --n_classes 104 --logdir ../vit3d_results/
