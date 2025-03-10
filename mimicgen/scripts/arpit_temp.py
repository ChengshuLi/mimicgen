import h5py

import numpy as np
import torch as th

# f = h5py.File('datasets/source_og/test_r1_cup.hdf5', 'r')
# breakpoint()
# print(f['data/demo_0/datagen_info'].keys())
# print(f['data/demo_0/datagen_info/eef_pose'])
# print(np.array(f['data/demo_0/datagen_info/subtask_term_signals/grasp']))
# print(np.array(f['data/demo_0/datagen_info/gripper_action']))
# print(f['data/demo_0/action'])

# def update_element(mylist=None):
#     mylist[1] = 99
#     print("Inside function:", mylist)

# nums = [1, 2, 3]
# update_element(nums)
# print("Outside function:", nums)


successes = th.tensor([True, False, True, False])
success_idx = th.where(successes)[0].cpu()