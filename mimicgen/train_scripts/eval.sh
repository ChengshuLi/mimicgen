cd /home/mengdi/Dropbox/Research/00-BEHAVIOR/b1k-mimicgen/mimicgen/

python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250120135620/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json --load_checkpoint_folder logs/test_tiago_cup/20250120135620 --start_epoch 2000

sleep 3

cd /home/mengdi/Dropbox/Research/00-BEHAVIOR/b1k-mimicgen/mimicgen/

python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250120140016/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json --load_checkpoint_folder logs/test_tiago_cup/20250120140016 --start_epoch 2000

sleep 3

cd /home/mengdi/Dropbox/Research/00-BEHAVIOR/b1k-mimicgen/mimicgen/

python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_cup/20250120140017/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D0.json --load_checkpoint_folder logs/test_tiago_cup/20250120140017 --start_epoch 2000





