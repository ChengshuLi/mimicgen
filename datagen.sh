#!/usr/bin/env bash
#SBATCH --account=cvgl
#SBATCH --partition=svl --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:2080ti:1

NUM_DEMOS=50
DR=0
WORKER_ID=0

cd /cvgl2/u/chengshu/anaconda3/bin
source activate ../envs/omnigibson-test
cd /cvgl2/u/chengshu/mimicgen
OMNIGIBSON_HEADLESS=1 python mimicgen/scripts/generate_dataset.py \
    --config datasets/generated_data_mimicgen_format/core_configs_og/demo_src_r1_pick_cup_task_D$DR.json \
    --num_demos $NUM_DEMOS \
    --bimanual \
    --auto-remove-exp \
    --folder datasets/generated_data_mimicgen_format/core_datasets_og/r1_pick_cup_worker_$WORKER_ID \
    --seed $WORKER_ID