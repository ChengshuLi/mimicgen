# for r1

sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_colorPCD_lr0001_b128_transformer.json'

sleep 20

sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_rgb_lr0001_b128.json'

sleep 20

sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_rgb_lr0001_b128_transformer.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_eef_colorPCD_lr0001_b128.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_eef_colorPCD_lr0001_b256.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_prop_colorPCD_lr0001_b256.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_colorPCD_lr0001_b128.json'

sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_colorPCD_lr0001_b256.json'


# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_prop_colorPCD_lr0001_b64.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_colorPCD_lr0001_b64.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_eef_colorPCD_lr0001_b64.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_prop_PCD_lr0001_b64.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_PCD_lr0001_b64.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_eef_PCD_lr0001_b64.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_prop_colorPCD_lr0001_b32.json'

# sleep 20

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_prop_colorPCD_lr0001_b128.json'


# for tiago
# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_ema.json'

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/train_config_cup_diffusion_D1_no_jpos_ema.json'

# interactive batch
# srun --account viscam --partition=viscam-interactive --gres=gpu:1 --pty bash

