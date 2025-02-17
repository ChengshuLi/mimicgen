# D1
# 10
sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_prep_data.py --file_path datasets/source_og/test_tiago_cup/demo_D1_10.hdf5 --output_path datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_10.hdf5 --split_ratio 0.1 --num_pcd_samples 2048 --fps --with_color'

# 64
# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_prep_data.py --file_path datasets/source_og/test_tiago_cup/demo_D1_64.hdf5 --output_path datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_64.hdf5 --split_ratio 0.1 --num_pcd_samples 2048 --fps --with_color'

# 500
# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_prep_data.py --file_path datasets/source_og/test_tiago_cup/demo_D1_500.hdf5 --output_path datasets/generated_data/test_tiago_cup/robomimic_dataset_D1_500.hdf5 --split_ratio 0.1 --num_pcd_samples 2048 --fps --with_color'