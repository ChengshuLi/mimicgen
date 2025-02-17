# D1

sbatch start_sbatch.sh 'python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250122132200/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_cup/20250122132200 --start_epoch 2000 --eval_on_train_init_states --headless'


# other hyperparameters
# --headless
# --eval_on_train_init_states 

