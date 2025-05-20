#!/usr/bin/env bash
#SBATCH --account=cvgl
#SBATCH --partition=svl --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:titanrtx:1

USERNAME=chengshu
TASK=$1
ABLATION=$2
DR=$3
WORKER_ID=$4

cd /cvgl2/u/chengshu/anaconda3/bin
source activate ../envs/omnigibson-test
cd /cvgl2/u/chengshu/mimicgen
OMNIGIBSON_HEADLESS=1 python mimicgen/scripts/videogen.py \
    --config_hdf5_path /cvgl2/u/chengshu/mimicgen/datasets/source_og/r1_$TASK.hdf5 \
    --data_hdf5_path /mnt/$USERNAME/momagen/$TASK\_$ABLATION\_vis/r1_$TASK\_worker_$WORKER_ID/demo_src_r1_$TASK\_task_D$DR/demo.hdf5 \
    --video_folder_path /mnt/$USERNAME/figure_images \
    --task $TASK \
    --dr D$DR
