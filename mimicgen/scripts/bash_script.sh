###
# example scripts adopted for the tiagacup task 
###

cp OmniGibson/omnigibson/examples/data_collection/test_tiago_cup.hdf5 mimicgen/datasets/source_og/test_tiago_cup.hdf5

cd mimicgen

# TODO: need to build an environment interface in/b1k-mimicgen/mimicgen/mimicgen/env_interfaces/omnigibson.py 

python mimicgen/scripts/prepare_src_dataset.py --dataset datasets/source_og/test_tiago_cup.hdf5 --env_interface MG_TestTiagoCup --env_interface_type omnigibson_bimanual

python mimicgen/scripts/get_source_info.py --dataset datasets/source_og/test_tiago_cup.hdf5
# example output:

# Environment Interface: MG_TestTiagoCup
# Environment Interface Type: omnigibson_bimanual

# Structure of datagen_info in episode demo_0:
#   eef_pose: shape (992, 8, 4)
#   object_poses:
#     coffee_cup: shape (992, 4, 4)
#     paper_cup: shape (992, 4, 4)
#     breakfast_table: shape (992, 4, 4)
#   subtask_term_signals:
#     grasp_right: shape (992,)
#     ungrasp_right: shape (992,)
#     grasp_left: shape (992,)
#     ungrasp_left: shape (992,)
#   target_pose: shape (992, 8, 4)
#   gripper_action: shape (992, 2)


# TODO: Need to first create a json file: 'mimicgen/mimicgen/exps/templates/omnigibson/{task_name}.json'. The json file specifies the reference objects at each subtask
# TODO: Need to add the corresponding json information to the mimicgen/scripts/generate_core_configs_og.py script
# TODO: need to make sure that the offset range makes sense, other wise there could be the following errors when sanity check with D0
# assert subtask_term_signals[-1] is None, "end of final subtask does not need to be detected" AssertionError: end of final subtask does not need to be detected

python mimicgen/scripts/generate_core_configs_og.py

python mimicgen/scripts/visualize_subtasks.py --dataset datasets/source_og/test_tiago_cup.hdf5 --config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D0.json --render --bimanual

# TODO: in /mimicgen/mimicgen/configs/omnigibson.py, need to create the config file for the corresponding new task

# TODO: in /robomimic/robomimic/envs/env_omnigibson.py line 130, need to specify the corresponding reference object
# add more objects in this function, can it be changed to the config, another config different from the datagen config, or maybe the same one

# sanity check with D0 to see whether the drift problem exist
# TODO: D0 quickly raise cuda error, need to check why 
python mimicgen/scripts/generate_dataset.py --config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D0.json --auto-remove-exp --num_demos 4 --bimanual


# even with really small initial posiiton randomization range, the 
python mimicgen/scripts/generate_dataset.py --config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D1.json --auto-remove-exp --num_demos 100 --bimanual

# even with really small initial posiiton randomization range, the 
python mimicgen/scripts/generate_dataset.py --config /tmp/core_configs_og/demo_src_test_tiago_cup_task_D2.json --auto-remove-exp --num_demos 10 --bimanual


# saved data file folder and file name
# /tmp/core_datasets_og/test_tiago_cup/demo_src_test_tiago_cup_task_D0
# /tmp/core_datasets_og/test_tiago_cup/demo_src_test_tiago_cup_task_D0/demo_failed.hdf5

# save data format
# results['init_state'], dict {'states': } size 284
# results['states'], list size = episode length, each element is of size 284
# results['actions'], array of (episode length, 22)
# results['mp_end_steps'], len = number of subtasks, each element {'left': , 'right': }
# results['subtask_lengths'], len = number of subtasks, each element is the number of steps in each subtask
# results['observations'], a list with size equal to episode length, each element is a dictionary with 5 keys
# dict_keys([
# 'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::rgb', # 128, 128, 4
# 'robot_fjtzyj::robot_fjtzyj:eyes:Camera:0::depth', # 128, 128
# 'task::low_dim', # 58
# 'external::external_sensor0::rgb',  # 128, 128, 4
# 'external::external_sensor0::depth' # 128, 128
# ])

# copy the dataset to the right folder
cd /tmp/core_datasets_og/test_tiago_cup/demo_src_test_tiago_cup_task_D0
# copy the dataset to the right foler
cp demo.hdf5 /home/mengdi/dataset/test_tiago_cup/demo_D1.hdf5