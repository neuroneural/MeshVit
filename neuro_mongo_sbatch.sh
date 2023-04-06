#!/bin/bash
#SBATCH -J vit3d
#SBATCH -o /data/users2/bbaker/projects/MeshVit/slurm//%j.out
#SBATCH -e /data/users2/bbaker/projects/MeshVit/slurm//%j.err
#SBATCH --nodes=1
#SBATCH -c 10
#SBATCH --mem 124G
#SBATCH --gres=gpu:v100:2
#SBATCH -p qTRDGPUH,qTRDGPUM,qTRDGPUL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bbaker43@gsu.edu
#SBATCH --oversubscribe
#SBATCH -t 7200
eval "$(conda shell.bash hook)"
conda activate neuro
cd /data/users2/bbaker/projects/MeshVit/neuro2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/bin/miniconda3/lib
tsv=64
sv=38
ps=19
PYTHONPATH=.. python training/minimal_mongo.py --parallel True --model segmenter --n_epochs 10 --n_classes 3 --logdir ../vit3d_results/ --train_subvolumes ${tsv} --patch_size ${ps} --sv_h ${sv}  --sv_w ${sv} --sv_d ${sv}