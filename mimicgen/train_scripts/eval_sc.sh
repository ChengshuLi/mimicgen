# write a for loop in bash
for LOG_INDEX in 20250303001540 20250303004126 20250303004117;
do
    for epoch in 700;
    do
        echo "Evaluate $LOG_INDEX with start epoch $epoch"
        echo sbatch start_sbatch.sh "python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_r1_cup/$LOG_INDEX/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --load_checkpoint_folder logs/test_r1_cup/$LOG_INDEX --eval_start_epoch $epoch --headless"
    done
done


# # D0

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250122132200/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_cup/20250122132200 --eval_start_epoch 2000 --eval_on_train_init_states --headless'


# other hyperparameters
# --headless
# --eval_on_train_init_states 



# policies to evaluate
############################  03/03/2025
######## the effect of batch size
#### 256
# epoch, 600, 650, 700, 750
# prop: 20250303001540
# qpos: 20250303004126
# eef:  20250303004117

#### 128
# epoch, 1100, 1150, 1200, 1250, 1300
# prop: 20250302235524
# qpos: 20250303002402
# eef:  20250303003626

#### 64
# epoch, 1600, 1650
# prop: 20250302235435
# qpos: 20250302235458
# eef:  20250302235527

######## the effect of color

# D1

# sbatch start_sbatch.sh 'python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250122132200/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_cup/20250122132200 --eval_start_epoch 2000 --eval_on_train_init_states --headless'


# other hyperparameters
# --headless
# --eval_on_train_init_states 