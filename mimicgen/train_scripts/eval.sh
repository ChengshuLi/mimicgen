cd /home/mengdi/Dropbox/Research/00-BEHAVIOR/b1k-mimicgen/mimicgen/

LOG_INDEX=20250122132200
LOG_INDEX=20250122101119
LOG_INDEX=20250201214318
LOG_INDEX=20250201213350

python mimicgen/train_scripts/eval_mimicgen.py --config "logs/test_tiago_cup/$LOG_INDEX/config.json" --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --load_checkpoint_folder "logs/test_tiago_cup/$LOG_INDEX" --eval_start_epoch 1900 --eval_on_train_init_states

# other hyperparameters
# --headless
# --eval_on_train_init_states 

