# MoMaGen

### Pipeline:
0. Obtain the OG dataset (should have the following keys: ['action', 'state', 'state_size', 'reward', 'terminated', 'truncated', 'init_metadata']) and save it in
/home/arpit/test_projects/OmniGibson/teleop_collected_data/

1. Create a json file: 'mimicgen/mimicgen/exps/templates/omnigibson/{task_name}.json'
Note:
a. specify filter key if using specific demos from human collected demo
b. use the following script to obtain the MP and subtask end steps 
c. specify the ref objects
d. specify the attached objects

2. If this is a new task, update the following files with this new task (similar to other tasks present in these files)
a. mimicgen/mimicgen/env_interfaces/omnigibson.py
b. mimicgen/mimicgen/configs/omnigibson.py

3. Run playback to visualize the collected data: (Use this to annotate the subtasks and MP end steps)

Run: python mimicgen/scripts/prepare_src_dataset.py --dataset /home/arpit/test_projects/OmniGibson/teleop_collected_data/tidy_table_0.hdf5 --env_interface MG_R1TidyTable --env_interface_type omnigibson_bimanual --filter_key use

Add --filter_key use in case want to use selected demos for data gen

4. Add datagen key to this hdf5 file: 

Run: python mimicgen/scripts/prepare_src_dataset.py --dataset /home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_clean_pan.hdf5 --env_interface MG_R1CleanPan --env_interface_type omnigibson_bimanual --save --output /home/arpit/test_projects/mimicgen/datasets/source_og/r1_clean_pan.hdf5 --filter_key use

Add --filter_key use in case want to use selected demos for data gen


5. generating the configs for datagen:
a. To change the folder where MoMaGen generated data is saved, change "dataset_name" and "generation_path" in generate_core_configs_og.py
b. To change task name, change "tasks"
c. Also remember to use the correct json file in the list BASE_CONFIGS in generate_core_configs_og.py 
d. run: python mimicgen/scripts/generate_core_configs_og.py
e. For Tiago ensure that self.single_arm = True in env_omnigibson.py script

Run: python mimicgen/scripts/generate_core_configs_og.py


6. save the relevant scene json file here: /home/arpit/test_projects/OmniGibson/omnigibson/data/og_dataset/scenes/house_single_floor/json/

7. Run data gen
python mimicgen/scripts/generate_dataset.py --config datasets/generated_data_mimicgen_format/core_configs_og/demo_src_r1_pick_cup_task_D0.json --auto-remove-exp --num_demos 500 --seed 1 --bimanual --video_path r1_pick_cup


### additional commands
Generating data:
1. python mimicgen/scripts/generate_dataset.py --config datasets/generated_data_mimicgen_format/core_configs_og/demo_src_temp_r1_task_D2.json --auto-remove-exp --num_demos 200 --bimanual --seed 2 --video_path temp

Training:
remove the split ratio in case you don't want anything in the validation set
2. python mimicgen/train_scripts/train_prep_data.py --file_path /tmp/core_datasets_og/test_tiago_single_arm_cup/demo_src_test_tiago_single_arm_cup_task_D1/demo_failed.hdf5 --output_path datasets/generated_data/test_tiago_single_arm_cup/robomimic_dataset_D1_ds.hdf5 --split_ratio 0.0 --num_pcd_samples 4096 --fps --vis_sign --with_color

3. python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_cup_task_D1.json --config mimicgen/train_scripts/train_configs/tiago_D1_jpos_colorPCD_lr0001_b128_ds.json

4. python mimicgen/train_scripts/eval_mimicgen.py --config logs/test_tiago_single_arm_cup_pick/20250322184759/config.json --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_tiago_single_arm_cup_task_D1.json --load_checkpoint_folder logs/test_tiago_single_arm_cup_pick/20250322184759 --eval_start_epoch 50 --single_epoch 1650 --eval_on_train_init_states

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



Some useful curobo pointers:
1. Difference between TrajOpt and MotionGen: https://github.com/NVlabs/curobo/discussions/227
- MotionGen is kind of a wraper over TrajOptSolver. It calls TrajOptSolver's solve_from_solve_state
2. Details on trajectory optimization of curobo: https://curobo.org/_api/curobo.wrap.reacher.trajopt.html#module-curobo.wrap.reacher.trajopt
- first running a particle-based solver (MPPI) and then refining with a gradient-based solver (L-BFGS)
3. finetuning cost weights: https://github.com/NVlabs/curobo/discussions/116


Base placement procedure:
1. Given first eef pose in the sequence, sample base2d poses near that eef pose
 

