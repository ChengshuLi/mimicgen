# 1. copy dataset to the home dataset folder

cd /tmp/core_datasets_og/test_tiago_cup/demo_src_test_tiago_cup_task_D0
cp demo.hdf5 /home/mengdi/dataset/demo.hdf5

# 2. copy mg config to the mimicgen folder

cp /tmp/core_configs_og/demo_src_test_tiago_cup_task_D1.json /home/mengdi/b1k_datagen/mimicgen/mimicgen/train_scripts/mg_configs/

# 3. change to robomimic dataset

#  --vis_sign
# D0
python mimicgen/train_scripts/train_prep_data.py --file_path /home/mengdi/dataset/test_tiago_cup/demo.hdf5 --output_path /home/mengdi/dataset/test_tiago_cup/robomimic_dataset.hdf5 --split_ratio 0.5 --num_pcd_samples 1024 --fps

# D1
python mimicgen/train_scripts/train_prep_data.py --file_path /home/mengdi/dataset/test_tiago_cup/demo_D1.hdf5 --output_path /home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup/robomimic_dataset_D1.hdf5 --split_ratio 0.1 --num_pcd_samples 2048 --fps --vis_sign


# 4 [not needed now]. need to copy the dataset to the mimicgen folders

# # D0
# cp /home/mengdi/dataset/test_tiago_cup/robomimic_dataset.hdf5 /home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup

# # D1
# cp /home/mengdi/dataset/test_tiago_cup/robomimic_dataset_D1_fps_2048.hdf5 /home/mengdi/b1k_datagen/mimicgen/datasets/generated_data/test_tiago_cup

# 5. start training

# D0
# training with low dim state
python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_dexcap.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json

# training with diffusion policy
python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_dexcap.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json


# D1
python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json

# some dependencies are missing, need to install then
# open3d
# pygpg
# fpsample
# diffusers

# 6. Evaluation of the trained model with rollout 
# --debug

python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250119014309/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json --load_checkpoint_folder logs/test_tiago_cup/20250119014309 --start_epoch 2000

python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250120135620/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_cup/20250120135620 --start_epoch 2000


python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250120140017/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_cup/20250120140017 --start_epoch 2000