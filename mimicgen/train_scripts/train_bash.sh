# D1

sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_ema.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos_ema.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json'
