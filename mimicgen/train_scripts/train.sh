# change to robomimic dataset
python mimicgen/train_scripts/train_prep_data.py --file_path /home/mengdi/dataset/demo.hdf5 --output_path /home/mengdi/dataset/robomimic_dataset.hdf5 --split_ratio 0.5

python mimicgen/train_scripts/train_prep_data.py --file_path /home/mengdi/dataset/demo.hdf5 --output_path /home/mengdi/dataset/robomimic_dataset.hdf5 --split_ratio 0.5 --debug

# start training
python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_config_cup.json --mg_config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D0.json

# training with diffusion policy
python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_config_cup_diffusion.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json --debug


# some dependencies are missing, need to install then
# open3d
# pygpg
# fpsample
# diffusers