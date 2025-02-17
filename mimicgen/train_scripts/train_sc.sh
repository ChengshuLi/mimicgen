# D1

sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos_color.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_ema.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos_ema.json'

# interactive batch
# srun --account viscam --partition=viscam-interactive --gres=gpu:1 --pty bash

