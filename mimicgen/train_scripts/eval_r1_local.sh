cd /home/mengdi/Dropbox/Research/00-BEHAVIOR/b1k-mimicgen/mimicgen/

### for r1 cup
LOG_INDEX=20250219235928
# LOG_INDEX=20250220001850
# LOG_INDEX=20250228224354
# LOG_INDEX=20250302141729
# LOG_INDEX=20250302151828

# for batch 256, epoch 800
LOG_INDEX=20250303001540
LOG_INDEX=20250303004126
# LOG_INDEX=20250303004117

# for batch 128, epoch 
# LOG_INDEX=20250303003626
LOG_INDEX=20250303002402
# LOG_INDEX=20250302235524

# for D0
python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_r1_cup/$LOG_INDEX/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --load_checkpoint_folder logs/test_r1_cup/$LOG_INDEX --eval_start_epoch 50 --single_epoch 2400

# other hyperparameters
# --headless
# --eval_on_train_init_states 

