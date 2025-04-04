# MimicGen

### commands
1. python mimicgen/scripts/generate_dataset.py --config datasets/generated_data_mimicgen_format/core_configs_og/demo_src_temp_r1_task_D2.json --auto-remove-exp --num_demos 200 --bimanual --seed 2 --video_path temp

remove the split ratio in case you don't want anything in the validation set
2. python mimicgen/train_scripts/train_prep_data.py --file_path /tmp/core_datasets_og/test_tiago_single_arm_cup/demo_src_test_tiago_single_arm_cup_task_D1/demo_failed.hdf5 --output_path datasets/generated_data/test_tiago_single_arm_cup/robomimic_dataset_D1_ds.hdf5 --split_ratio 0.0 --num_pcd_samples 4096 --fps --vis_sign --with_color

3. python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/tiago_D1_jpos_colorPCD_lr0001_b128_ds.json

4. python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_single_arm_cup_pick/20250322184759/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_single_arm_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_single_arm_cup_pick/20250322184759 --eval_start_epoch 50 --single_epoch 1650 --eval_on_train_init_states

### Steps:
1. To change the folder where OG generated data is saved, change "dataset_name" and "generation_path" in generate_core_configs_og.py
2. To change task name, change "tasks"
3. Also remember to use the correct json file in the list BASE_CONFIGS in generate_core_configs_og.py 
4. run: python mimicgen/scripts/generate_core_configs_og.py
5. For Tiago ensure that self.single_arm = True in env_omnigibson.py script


### For installation:
1. mimicgen installation on its homepage
2. robomimic on the b1k-mimicgen branch
3. robosuite on its latest branch
4. omnigibson on its b1k-mimicgen branch  
5. curobo installation: TODO


### Parameters to consider for curobo:
1. Enable_graph: https://github.com/StanfordVL/OmniGibson/blob/b66a042f106636cebda67c9b13bc8e872a08a837/omnigibson/action_primitives/curobo.py#L623 
2. Max_distance: https://github.com/StanfordVL/OmniGibson/blob/b66a042f106636cebda67c9b13bc8e872a08a837/omnigibson/action_primitives/curobo.py#L122
3. Batch_size: https://github.com/StanfordVL/OmniGibson/blob/b66a042f106636cebda67c9b13bc8e872a08a837/omnigibson/action_primitives/starter_semantic_action_primitives.py#L146
4. collision_activation_distance (in curobo.py and ..action_primitives.py)


### Mimicgen relevant config files:
1. mimicgen/mimicgen/env_interfaces/omnigibson.py
2. mimicgen/mimicgen/configs/omnigibson.py


before clip torso_joint1 type RevoluteJoint limit: tensor(-1.134, device='cuda:0') tensor(1.833, device='cuda:0')
after clip torso_joint1 type RevoluteJoint limit: tensor(0.424, device='cuda:0') tensor(0.724, device='cuda:0')

before clip torso_joint2 type RevoluteJoint limit: tensor(-2.793, device='cuda:0') tensor(2.531, device='cuda:0')
after clip torso_joint2 type RevoluteJoint limit: tensor(-2.847, device='cuda:0') tensor(-0.947, device='cuda:0')

before clip torso_joint3 type RevoluteJoint limit: tensor(-2.094, device='cuda:0') tensor(1.833, device='cuda:0')
after clip torso_joint3 type RevoluteJoint limit: tensor(-1.524, device='cuda:0') tensor(-0.424, device='cuda:0')

before clip torso_joint4 type RevoluteJoint limit: tensor(-3.054, device='cuda:0') tensor(3.054, device='cuda:0')
after clip torso_joint4 type RevoluteJoint limit: tensor(-0.540, device='cuda:0') tensor(0.540, device='cuda:0')


R1 without joint limits
**************************************************
trial 21 success: True
have 16 successes out of 21 trials so far
have 0 failures out of 21 trials so far
have 5 Base MP failures, 0 Arm MP failures, 0 Base sampling failures
**************************************************


R1 with joint limits. 4 arm MP failures are because during nav MP, robot collided with table and moved it a lot
**************************************************
trial 20 success: True
have 13 successes out of 20 trials so far
have 1 failures out of 20 trials so far
have 2 Base MP failures, 4 Arm MP failures, 0 Base sampling failures
**************************************************

Observations:
Start-of-manip visibility is decent (but that's just because this is a table-top setting and our start joint positions are conducive for this. So, we will need an
explicit visibility constraint). But, during manip the visibility goes really bad. Options:
a) use arm-no-torso -> This might be limiting when we have more diverse envs (e.g when objects are lower or much higher and when there are more obstacles)
b) Keep data gen pipeline as is and somehow force the policy to pay more attention to eef camera
c) Modify curobo motion planner with visibility cost -> This would be ideal!!


R1 with joint limits and ARM_NO_TORSO mode
**************************************************
trial 20 success: True
have 12 successes out of 20 trials so far
have 1 failures out of 20 trials so far
have 3 Base MP failures, 4 Arm MP failures, 0 Base sampling failures
**************************************************


Analyzing visibilty:

R1 with joint limits and ARM_NO_TORSO mode, table height varied, w/o visibility constraint
**************************************************
trial 20 success: True
have 10 successes out of 20 trials so far
have 1 failures out of 20 trials so far
have 1 Base MP failures, 8 Arm MP failures, 0 Base sampling failures
have 13 trials with obj visible at start of manip
**************************************************



Some useful curobo pointers:
1. Difference between TrajOpt and MotionGen: https://github.com/NVlabs/curobo/discussions/227
- MotionGen is kind of a wraper over TrajOptSolver. It calls TrajOptSolver's solve_from_solve_state
2. Details on trajectory optimization of curobo: https://curobo.org/_api/curobo.wrap.reacher.trajopt.html#module-curobo.wrap.reacher.trajopt
- first running a particle-based solver (MPPI) and then refining with a gradient-based solver (L-BFGS)
