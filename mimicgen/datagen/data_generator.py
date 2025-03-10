# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base class for data generator.
"""
import h5py
import sys
import numpy as np
import pdb

import mimicgen
import mimicgen.utils.pose_utils as PoseUtils
import mimicgen.utils.file_utils as MG_FileUtils

from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.datagen_info import DatagenInfo
from mimicgen.datagen.selection_strategy import make_selection_strategy
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory

import omnigibson as og
from omnigibson.object_states.contact_bodies import ContactBodies

class DataGenerator(object):
    """
    The main data generator object that loads a source dataset, parses it, and 
    generates new trajectories.
    """
    def __init__(
        self,
        task_spec,
        dataset_path,
        demo_keys=None,
        bimanual=False,
        D2_sign=False,
        ignore_failed_mp=True,
    ):
        """
        Args:
            task_spec (MG_TaskSpec instance): task specification that will be
                used to generate data
            dataset_path (str): path to hdf5 dataset to use for generation
            demo_keys (list of str): list of demonstration keys to use
                in file. If not provided, all demonstration keys will be
                used.
        """
        assert isinstance(task_spec, MG_TaskSpec)
        self.task_spec = task_spec
        self.dataset_path = dataset_path
        self.bimanual = bimanual
        self.D2_sign = D2_sign

        if self.bimanual:
            self.num_phases = len(self.task_spec)
            # sanity check on task spec offset ranges - final subtask should not have any offset randomization
            for phase_index in range(self.num_phases):
                phase_spec = self.task_spec[phase_index]
                # for left arm
                assert phase_spec[0][-1]["subtask_term_offset_range"][0] == 0
                assert phase_spec[0][-1]["subtask_term_offset_range"][1] == 0
                # for right arm
                assert phase_spec[1][-1]["subtask_term_offset_range"][0] == 0
                assert phase_spec[1][-1]["subtask_term_offset_range"][1] == 0

        else:
            # sanity check on task spec offset ranges - final subtask should not have any offset randomization
            assert self.task_spec[-1]["subtask_term_offset_range"][0] == 0
            assert self.task_spec[-1]["subtask_term_offset_range"][1] == 0

        # demonstration keys to use from hdf5 as source dataset
        if demo_keys is None:
            # get all demonstration keys from file
            demo_keys = MG_FileUtils.get_all_demos_from_dataset(dataset_path=self.dataset)
        self.demo_keys = demo_keys

        self.ignore_failed_mp = ignore_failed_mp

        # parse source dataset
        self._load_dataset(dataset_path=dataset_path, demo_keys=demo_keys)

    def _load_dataset(self, dataset_path, demo_keys):
        """
        Load important information from a dataset into internal memory.
        """
        print("\nDataGenerator: loading dataset at path {}...".format(dataset_path))
        if self.bimanual:
            self.src_dataset_infos, self.src_subtask_indices, self.subtask_names, _ = MG_FileUtils.parse_source_dataset_bimanual(
                dataset_path=dataset_path,
                demo_keys=demo_keys,
                task_spec=self.task_spec,
            )
        else:
            self.src_dataset_infos, self.src_subtask_indices, self.subtask_names, _ = MG_FileUtils.parse_source_dataset(
                dataset_path=dataset_path,
                demo_keys=demo_keys,
                task_spec=self.task_spec,
            )
        print("\nDataGenerator: done loading\n")

    def __repr__(self):
        """
        Pretty print this object.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tdataset_path={}\n\tdemo_keys={}\n)".format(
            self.dataset_path,
            self.demo_keys,
        )
        return msg

    def randomize_subtask_boundaries(self, src_subtask_indices, task_spec):
        """
        Apply random offsets to sample subtask boundaries according to the task spec.
        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index of each subtask can have a random offset.
        """
        # TODO: will need to sample the subtasks boundaries with the two arm coordination within consideration

        # initial subtask start and end indices - shape (N, S, 2)
        src_subtask_indices = np.array(src_subtask_indices)

        # for each subtask (except last one), sample all end offsets at once for each demonstration
        # add them to subtask end indices, and then set them as the start indices of next subtask too
        for i in range(src_subtask_indices.shape[1] - 1):
            end_offsets = np.random.randint(
                low=task_spec[i]["subtask_term_offset_range"][0],
                high=task_spec[i]["subtask_term_offset_range"][1] + 1,
                size=src_subtask_indices.shape[0]
            )
            src_subtask_indices[:, i, 1] = src_subtask_indices[:, i, 1] + end_offsets
            # don't forget to set these as start indices for next subtask too
            src_subtask_indices[:, i + 1, 0] = src_subtask_indices[:, i, 1]

        # ensure non-empty subtasks
        assert np.all((src_subtask_indices[:, :, 1] - src_subtask_indices[:, :, 0]) > 0), "got empty subtasks!"

        # ensure subtask indices increase (both starts and ends)
        assert np.all((src_subtask_indices[:, 1:, :] - src_subtask_indices[:, :-1, :]) > 0), "subtask indices do not strictly increase"

        # ensure subtasks are in order
        subtask_inds_flat = src_subtask_indices.reshape(src_subtask_indices.shape[0], -1)
        assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

        return src_subtask_indices

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        subtask_ind,
        src_subtask_inds,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
    ):
        """
        Helper method to run source subtask segment selection.

        Args:
            eef_pose (np.array): current end effector pose
            object_pose (np.array): current object pose for this subtask
            subtask_ind (int): index of subtask
            src_subtask_inds (np.array): start and end indices for subtask segment in source demonstrations of shape (N, 2)
            subtask_object_name (str): name of reference object for this subtask
            selection_strategy_name (str): name of selection strategy
            selection_strategy_kwargs (dict): extra kwargs for running selection strategy

        Returns:
            selected_src_demo_ind (int): selected source demo index
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random"

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source 
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.demo_keys)):
            # datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_dataset_infos[i]

            # time indices for subtask
            subtask_start_ind = src_subtask_inds[i][0]
            subtask_end_ind = src_subtask_inds[i][1]

            # get subtask segment using indices
            src_subtask_datagen_infos.append(DatagenInfo(
                eef_pose=src_ep_datagen_info.eef_pose[subtask_start_ind : subtask_end_ind],
                # only include object pose for relevant object in subtask
                object_poses={ subtask_object_name : src_ep_datagen_info.object_poses[subtask_object_name][subtask_start_ind : subtask_end_ind] } if (subtask_object_name is not None) else None,
                # subtask termination signal is unused
                subtask_term_signals=None,
                target_pose=src_ep_datagen_info.target_pose[subtask_start_ind : subtask_end_ind],
                gripper_action=src_ep_datagen_info.gripper_action[subtask_start_ind : subtask_end_ind],
            ))

        # make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def merge_trajs(self, traj_list_all):
        # merge the waypoints for each arm
        # print('#################### in merge trajectories ####################')
        
        waypoint_traj_list = []
        for i in range(2):
            traj_list = traj_list_all[i]
            waypoint_traj = WaypointTrajectory()
            for traj in traj_list:
                for seq in traj.waypoint_sequences:
                    if waypoint_traj.waypoint_sequences == []:
                        waypoint_traj.add_waypoint_sequence(seq)
                    else:
                        waypoint_traj.waypoint_sequences[-1].sequence += seq.sequence
                    # print('num waypoints:', len(waypoint_traj.waypoint_sequences[-1].sequence))
            waypoint_traj_list.append(waypoint_traj)
        
        
        # merge the left and right eef pose
        traj_left = waypoint_traj_list[0]
        traj_right = waypoint_traj_list[1]
        min_length = min(len(traj_left.waypoint_sequences[0].sequence), len(traj_right.waypoint_sequences[0].sequence))
        max_length = max(len(traj_left.waypoint_sequences[0].sequence), len(traj_right.waypoint_sequences[0].sequence))
        if max_length > min_length:
            if len(traj_left.waypoint_sequences[0].sequence) == min_length:
                for _ in range(max_length - min_length):
                    traj_left.waypoint_sequences[0].sequence.append(traj_left.waypoint_sequences[0].sequence[-1])
            else:
                for _ in range(max_length - min_length):
                    traj_right.waypoint_sequences[0].sequence.append(traj_right.waypoint_sequences[0].sequence[-1])
        for i in range(max_length):
            traj_left.waypoint_sequences[0].sequence[i].merge_wp(traj_right.waypoint_sequences[0].sequence[i])
        traj_to_execute = traj_left

        return traj_to_execute

    def change_arm_role_heuristic(self,
                                  env_interface,
                                  start_step,
                                  selected_src_demo_ind,
                                  cur_phase_task_spec
                                  ):
        change_role = False

        src_left_arm_start_pos = self.src_dataset_infos[selected_src_demo_ind].eef_pose[start_step:start_step+1][:,:4,:] # shape (1, 4, 4)
        src_right_arm_start_pose = self.src_dataset_infos[selected_src_demo_ind].eef_pose[start_step:start_step+1][:,4:,:] # shape (1, 4, 4)

        left_arm_object_name = cur_phase_task_spec[0][0]["object_ref"]
        right_arm_object_name = cur_phase_task_spec[1][0]["object_ref"]
        src_left_arm_object_pose = self.src_dataset_infos[selected_src_demo_ind].object_poses[left_arm_object_name][start_step]
        src_right_arm_object_pose = self.src_dataset_infos[selected_src_demo_ind].object_poses[right_arm_object_name][start_step]
        cur_left_arm_object_pose = env_interface.get_datagen_info().object_poses[left_arm_object_name] # shape (4, 4)
        cur_right_arm_object_pose = env_interface.get_datagen_info().object_poses[right_arm_object_name] # shape (4, 4)

        transformed_eef_poses_left_arm_object = PoseUtils.transform_source_data_segment_using_object_pose(
            obj_pose=cur_left_arm_object_pose, 
            src_eef_poses=src_left_arm_start_pos,
            src_obj_pose=src_left_arm_object_pose) # shape (1, 4, 4)
        transformed_eef_poses_right_arm_object = PoseUtils.transform_source_data_segment_using_object_pose(
            obj_pose=cur_right_arm_object_pose, 
            src_eef_poses=src_right_arm_start_pose,
            src_obj_pose=src_right_arm_object_pose) # shape (1, 4, 4)

        cur_left_arm_pose = env_interface.get_datagen_info().eef_pose[None][:,:4,:] # shape (1, 4, 4)
        cur_right_arm_pose = env_interface.get_datagen_info().eef_pose[None][:,4:,:] # shape (1, 4, 4)

        distance_left_arm_to_traj_left_arm_object = np.linalg.norm(cur_left_arm_pose[:,:,-1] - transformed_eef_poses_left_arm_object[:,:,-1])
        distance_right_arm_to_traj_left_arm_object = np.linalg.norm(cur_right_arm_pose[:,:,-1] - transformed_eef_poses_left_arm_object[:,:,-1])

        distance_left_arm_to_traj_right_arm_object = np.linalg.norm(cur_left_arm_pose[:,:,-1] - transformed_eef_poses_right_arm_object[:,:,-1])
        distance_right_arm_to_traj_right_arm_object = np.linalg.norm(cur_right_arm_pose[:,:,-1] - transformed_eef_poses_right_arm_object[:,:,-1])

        print('========================================== new phase ==========================================')
        print('distance_left_arm_to_traj_left_arm_object', distance_left_arm_to_traj_left_arm_object)
        print('distance_right_arm_to_traj_left_arm_object', distance_right_arm_to_traj_left_arm_object)
        print('distance_left_arm_to_traj_right_arm_object', distance_left_arm_to_traj_right_arm_object)
        print('distance_right_arm_to_traj_right_arm_object', distance_right_arm_to_traj_right_arm_object)

        # compare the distances 
        if distance_left_arm_to_traj_left_arm_object < distance_right_arm_to_traj_left_arm_object and distance_right_arm_to_traj_right_arm_object < distance_right_arm_to_traj_left_arm_object:
            change_role = False
            print('no change role')
        elif distance_left_arm_to_traj_left_arm_object > distance_right_arm_to_traj_left_arm_object and distance_right_arm_to_traj_right_arm_object > distance_right_arm_to_traj_left_arm_object:
            change_role = True
            print('change role')
            # breakpoint()
        else:
            # TODO: if the change arm role constaints are not satisfied, will keep the original arm role
            print('distance comparison heuristic is not applicable, check corner cases')
            change_role = False
            # breakpoint()
            # raise ValueError('The distance comparison heuristic is not applicable, check corner cases')

        return change_role

    def parse_MP_end_step_local(self):
        """
        parse the MP_end_step from the configuration file and get the local information
        """
        # example output
        # [
        #   [
        #       [160, -1], 
        #       [110, 0]
        #   ], 
        #   [
        #       [180], 
        #       [-1]
        #   ]
        # ]
        end_step_of_MP = []
        for phase_ind in range(self.num_phases):
            end_step_of_MP.append([])
            for arm_ind in range(2): # left and right arms
                num_subtasks_cur_phase = len(self.task_spec[phase_ind][arm_ind])
                end_step_of_MP[-1].append([])
                for i in range(num_subtasks_cur_phase):
                    if self.task_spec[phase_ind][arm_ind][i]["MP_end_step"] is not None:
                        end_step = self.task_spec[phase_ind][arm_ind][i]["MP_end_step"]
                    elif self.task_spec[phase_ind][arm_ind][i]['subtask_term_step'] is not None:
                        end_step = self.task_spec[phase_ind][arm_ind][i]['subtask_term_step']
                    else:
                        # We only have one demo right now, so we can use the length of the demo as the end step
                        end_step = self.src_dataset_infos[0].eef_pose.shape[0]

                    end_step_of_MP[-1][-1].append(end_step)
        print('end_step_of_MP', end_step_of_MP)
        return end_step_of_MP

    def parse_annotations(self, annotations):
        annotations = None
        return annotations

    def generate(
        self,
        env,
        env_interfaces,
        select_src_per_subtask=False,
        transform_first_robot_pose=False,
        interpolate_from_last_target_pose=True,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
        pause_subtask=False,
    ):
        """
        Attempt to generate a new demonstration.

        Args:
            env (robomimic EnvBase instance): environment to use for data collection
            
            env_interface (MG_EnvInterface instance): environment interface for some data generation operations

            select_src_per_subtask (bool): if True, select a different source demonstration for each subtask 
                during data generation, else keep the same one for the entire episode

            transform_first_robot_pose (bool): if True, each subtask segment will consist of the first
                robot pose and the target poses instead of just the target poses. Can sometimes help
                improve data generation quality as the interpolation segment will interpolate to where 
                the robot started in the source segment instead of the first target pose. Note that the
                first subtask segment of each episode will always include the first robot pose, regardless
                of this argument.
                TODO: not sure about the meaning of this property

            interpolate_from_last_target_pose (bool): if True, each interpolation segment will start from
                the last target pose in the previous subtask segment, instead of the current robot pose. Can
                sometimes improve data generation quality.

            render (bool): if True, render on-screen

            video_writer (imageio writer): video writer

            video_skip (int): determines rate at which environment frames are written to video

            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.

            pause_subtask (bool): if True, pause after every subtask during generation, for
                debugging.

        Returns:
            results (dict): dictionary with the following items:
                initial_state (dict): initial simulator state for the executed trajectory
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (np.array): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
                src_demo_inds (list): list of selected source demonstration indices for each subtask
                src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
        """

        # sample new task instance
        env.reset()
        new_initial_state = env.get_state()
        num_envs = len(env.env.envs)


        # # # check collisions between robot and all other objects
        # # print('breakpoint before collision check')
        # # breakpoint()
        # # robot = env.robots[0]
        # # collision_stats = robot.states[ContactBodies].get_state()

        # # print('breakpoint in run_rollout to check contacts')
        # # breakpoint()
        # def check_reset_requirement():
        #     need_to_reset = False
        #     robot = env.env.robots[0]
        #     contact_prim_set = robot.states[ContactBodies].get_value()
        #     for prim in contact_prim_set:
        #         print(prim.name, 'is in contact')
        #     contact_name_list = [prim.name for prim in contact_prim_set]
        #     for name in contact_name_list:
        #         if 'coffee_cup' in name:
        #             need_to_reset = True
        #         if 'paper_cup' in name:
        #             need_to_reset = True
        #     return need_to_reset
        
        # reset_max_times = 2

        # breakpoint()

        # reset_count = 1
        # while check_reset_requirement() and reset_count < reset_max_times:
        #     print('need to reset')
        #     reset_count += 1
        #     env.reset()
        #     breakpoint()

        # set camera postion
        import omnigibson as og
        import torch as th
        # og.sim.viewer_camera.set_position_orientation(
        #     position=th.tensor([ 1.7492, -0.0424,  1.5371]),
        #     orientation=th.tensor([0.3379, 0.3417, 0.6236, 0.6166]),
        # ) # viewer position

        for env_idx, e in enumerate(env.env.envs):
            # TODO: need to change the sensor resolution based on requirement
            sensor = e._external_sensors['external_sensor0']

            if env_idx == 0:
                sensor.set_position_orientation(position=th.tensor([ 3.22, -0.026,  2.476]),orientation=th.tensor([0.323, 0.329, 0.634, 0.621]),)
            elif env_idx == 1:
                sensor.set_position_orientation(position=th.tensor([ 19.18, -0.026,  2.476]),orientation=th.tensor([0.323, 0.329, 0.634, 0.621]),)
            elif env_idx == 2:
                sensor.set_position_orientation(position=th.tensor([ 35.14, -0.026,  2.476]),orientation=th.tensor([0.323, 0.329, 0.634, 0.621]),)

            # sensor config option 1: facing robot
            # sensor.set_position_orientation(
            #     position=th.tensor([ 1.7492, -0.0424,  1.5371]),
            #     orientation=th.tensor([0.3379, 0.3417, 0.6236, 0.6166]),
            #     )
            
            # sensor config option 2: camera zoomed in facing the robot
            # sensor.set_position_orientation(
            #     position=th.tensor([ 1.0693, -0.0211,  0.9937]),
            #     orientation=th.tensor([0.2479, 0.2451, 0.6590, 0.6665]),
                # )
            # sensor.set_position_orientation(
            #     position=th.tensor([ 1.0304, -0.0309,  1.0272]),
            #     orientation=th.tensor([0.2690, 0.2659, 0.6509, 0.6583]),
            # )

            # sensor config option 3: camera zoomed in
            # sensor.set_position_orientation(
            #     position=th.tensor([ 0.1300, -0.0262,  0.8532]),
            #     orientation=th.tensor([-0.3200,  0.3207,  0.6311, -0.6296]),
                # )

            sensor.image_height = 180
            sensor.image_width = 320

            # sensor.image_height = 1080
            # sensor.image_width = 1920
                    
            sensor._add_modality_to_backend(modality='depth_linear')
            sensor._add_modality_to_backend(modality='rgb')
            sensor._modalities = {"depth_linear", "rgb"}
        
        for _ in range(5): og.sim.render()

        # print(sensor.intrinsic_matrix)
        # print(sensor.get_position_orientation())

        # TODO: need to change the agent sensor correspondingly

        external_sensor_info = {
            "pose": sensor.get_position_orientation(),
            "intrinsic_matrix": sensor.intrinsic_matrix,
            "image_height": sensor.image_height,
            "image_width": sensor.image_width,
        }

        # parse MP_end_step from the configuration file
        end_step_of_MP_local = self.parse_MP_end_step_local()

        # after changing the phase structure, 
        # self.src_subtask_indices
        # [
        # [array([[[  0, 300],
        # [300, 650]]]), array([[[  0, 350],
        # [350, 650]]])], 
        # [array([[[650, 992]]]), array([[[650, 992]]])]
        # ]

        # sample new subtask boundaries
        all_subtask_inds_structure = []
        for phase_index in range(self.num_phases):
            all_subtask_inds_structure.append([])
            for arm_i in range(2): # arm_left, arm_right
                all_subtask_inds_arm = self.randomize_subtask_boundaries(self.src_subtask_indices[phase_index][arm_i], self.task_spec[phase_index][arm_i]) # shape (1,2,2)
                all_subtask_inds_structure[-1].append(all_subtask_inds_arm)

        # all_subtask_inds_structure is a list of length @num_phases
        # all_subtask_inds_structure[0] is a list of length 2, corresponding to left and right arms
        # all_subtask_inds_structure[0][0] is a numpy array of shape (@num_demos, @num_subtasks, 2)
        # where @num_demos is 1 right now, @num_subtasks can vary, 2 means start and end indices

        # (Pdb) p all_subtask_inds_structure
        # [[array([[[  0, 309],
        # [309, 650]]]), array([[[  0, 360],
        # [360, 650]]])], [array([[[650, 992]]]), array([[[650, 992]]])]]

        # some state variables used during generation
        selected_src_demo_ind = None
        prev_executed_traj = None

        # save generated data in these variables
        generated_states = [[] for _ in range(num_envs)]
        generated_obs = [[] for _ in range(num_envs)]
        generated_datagen_infos = [[] for _ in range(num_envs)]
        generated_actions = [[] for _ in range(num_envs)]
        generated_demo_mp_end_steps = [[] for _ in range(num_envs)]
        generated_demo_subtask_lengths = [[] for _ in range(num_envs)]
        generated_success = [False for _ in range(num_envs)]
        generated_src_demo_inds = [[] for _ in range(num_envs)] # store selected src demo ind for each subtask in each trajectory
        generated_src_demo_labels = [[] for _ in range(num_envs)] # like @generated_src_demo_inds, but padded to align with size of @generated_actions

        # Set all envs to be valid at the start of a new data gen episode
        env.valid_envs = [True] * num_envs
        env.primitive.valid_envs = env.valid_envs

        # remove later
        # env.env.envs[1].scene.object_registry("name", "breakfast_table").set_position_orientation(position=th.tensor([50.0, 0.0, 0.0]))
        # for _ in range(10): og.sim.step()

        # for left arms first
        for phase_ind in range(self.num_phases):
            # # remove later
            # if phase_ind < 1:
            #     continue
            cur_phase_task_spec = self.task_spec[phase_ind]
            selected_src_demo_ind = 0 # TODO: since we only have one demo, will need to modify if more demos are available

            # restructure subtasks indexes and reference objects
            all_subtask_inds = all_subtask_inds_structure[phase_ind]
            subtask_ind_vals = np.sort(np.unique(np.concatenate((np.unique(all_subtask_inds[0]), np.unique(all_subtask_inds[1])))))
            num_subtasks = len(subtask_ind_vals) - 1
            
            # TODO: Modify this for vecotrized envs [IMPORTANT!!]
            # # a distance based heuristic to change the role of the two arms
            # # calculate the start of the replay part
            # # currently assume that the start point is the first subtask of the current phase
            # # TODO: need to change this to other starting point when the motion planner is integrated
            # start_step = subtask_ind_vals[0]
            # change_role = self.change_arm_role_heuristic(
            #     env_interface,
            #     start_step,
            #     selected_src_demo_ind,
            #     cur_phase_task_spec
            #     )

            # if change_role:
            #     # change the information for two arms
            #     cur_phase_task_spec_new = []
            #     cur_phase_task_spec_new.append(cur_phase_task_spec[1])
            #     cur_phase_task_spec_new.append(cur_phase_task_spec[0])
            #     cur_phase_task_spec = cur_phase_task_spec_new
            #     all_subtask_inds_new = []
            #     all_subtask_inds_new.append(all_subtask_inds[1])
            #     all_subtask_inds_new.append(all_subtask_inds[0])
            #     all_subtask_inds = all_subtask_inds_new
            # remove later
            change_role = False

            for subtask_ind_reordered in range(num_subtasks):
                print("========== Phase {} Subtask {} ==========".format(phase_ind, subtask_ind_reordered))

                traj_to_execute_all_env = []
                MP_end_steps_per_env = []
                for env_idx, single_env in enumerate(env.env.envs):
                    selected_src_subtask_inds = subtask_ind_vals[subtask_ind_reordered : subtask_ind_reordered + 2] # [start_step, end_step]
                    traj_list_all = [[],[]]
                    attached_obj_dict = {}
                    object_ref = {}                
                    MP_end_steps = []
                    
                    for arm_i, arm_name in enumerate(['arm_left', 'arm_right']):

                        # need to recalculate the matched subtask_ind to retrieve the correct task spec
                        local_task_spec = cur_phase_task_spec[arm_i]
                        arm_spec_subtask_inds = all_subtask_inds[arm_i][0]
                        arm_unique_subtask_inds = np.sort(np.unique(arm_spec_subtask_inds))
                        subtask_ind = np.where(selected_src_subtask_inds[1] <= arm_unique_subtask_inds)[0][0] - 1

                        # print('arm_name:', arm_name, 'subtask_ind_reordered', subtask_ind_reordered, 'subtask_ind:', subtask_ind)
                        # print(f'env {env_idx} arm_name {arm_name} subtask start and end step', selected_src_subtask_inds)
                        # print('arm_spec_subtask_inds', arm_spec_subtask_inds)

                        is_first_subtask = (subtask_ind == 0) and (phase_ind == 0)
                        is_first_subtask_in_phase = (subtask_ind == 0)

                        cur_datagen_info = env_interfaces[env_idx].get_datagen_info()
                        subtask_object_name = cur_phase_task_spec[arm_i][subtask_ind]["object_ref"]
                        object_ref[arm_name] = subtask_object_name
                        cur_object_pose = cur_datagen_info.object_poses[subtask_object_name] if (subtask_object_name is not None) else None # 4x4
                        key_name = arm_name.replace('arm_', '')
                        attached_obj_dict[key_name] = cur_phase_task_spec[arm_i][subtask_ind]["attached_obj"]
                        MP_end_steps.append(end_step_of_MP_local[phase_ind][arm_i][subtask_ind])
                        
                        # get poses
                        src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]
                        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
                        # src_subtask_target_poses = src_ep_datagen_info.target_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
                        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 2

                        if (arm_name == 'arm_left' and not change_role) or (arm_name == 'arm_right' and change_role):
                            # print('select left arm demo pose')
                            src_subtask_eef_poses = src_subtask_eef_poses[:,:4,:]
                            # src_subtask_target_poses = src_subtask_target_poses[:,:4,:]
                            src_subtask_gripper_actions = src_subtask_gripper_actions[:,:1]
                        elif (arm_name == 'arm_right' and not change_role) or (arm_name == 'arm_left' and change_role):
                            # print('select right arm demo pose')
                            src_subtask_eef_poses = src_subtask_eef_poses[:,4:,:]
                            # src_subtask_target_poses = src_subtask_target_poses[:,4:,:]
                            src_subtask_gripper_actions = src_subtask_gripper_actions[:,1:]

                        # get reference object pose from source demo
                        src_subtask_object_pose = src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]] if (subtask_object_name is not None) else None # 4 x 4

                        # src_eef_poses = np.array(src_subtask_eef_poses)
                        # if is_first_subtask or transform_first_robot_pose:
                        #     # Source segment consists of first robot eef pose and the target poses. This ensures that
                        #     # we will interpolate to the first robot eef pose in this source segment, instead of the
                        #     # first robot target pose.
                        #     # TODO: not sure about the meaning of this; need to check the first dimension is 1 more
                        #     src_eef_poses = np.concatenate([src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0) # 107 x 8 x 4
                        # else:
                        #     # Source segment consists of just the target poses.
                        #     src_eef_poses = np.array(src_subtask_target_poses)

                        # account for extra timestep added to @src_eef_poses
                        # src_subtask_gripper_actions = np.concatenate([src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0) # 107 x2

                        src_eef_poses = src_subtask_eef_poses
                        # Transform source demonstration segment using relevant object pose.
                        if subtask_object_name is not None:
                            # print('cur_object_pose', cur_object_pose.shape)
                            # print('src_eef_poses', src_eef_poses.shape)
                            # print('src_subtask_object_pose', src_subtask_object_pose.shape)
                            transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                                obj_pose=cur_object_pose, 
                                src_eef_poses=src_eef_poses,
                                src_obj_pose=src_subtask_object_pose)
                            # transformed_eef_poses = np.concatenate([transformed_eef_poses_left, transformed_eef_poses_right], axis=1)
                        else:
                            # skip transformation if no reference object is provided
                            transformed_eef_poses = src_eef_poses

                        # We will construct a WaypointTrajectory instance to keep track of robot control targets 
                        # that will be executed and then execute it.
                        # traj_to_execute = WaypointTrajectory()

                        # TODO: change the interpolation to curobo motion planner

                        # if interpolate_from_last_target_pose and (not is_first_subtask_in_phase):
                        #     # Interpolation segment will start from last target pose (which may not have been achieved).

                        #     # TODO: since we did not execute the subtask within each phase, the assettion will fail -> remove the assertion
                        #     # assert prev_executed_traj is not None
                        #     # last_waypoint = prev_executed_traj.last_waypoint

                        #     # instead, we get the last waypoint from the last subtask
                        #     last_waypoint = traj_list_all[arm_i][-1].last_waypoint
                        #     init_sequence = WaypointSequence(sequence=[last_waypoint])
                        # else:
                        # if True:
                        # if arm_name == 'arm_left':
                        #     # Interpolation segment will start from current robot eef pose.
                        #     init_sequence = WaypointSequence.from_poses(
                        #         poses=cur_datagen_info.eef_pose[None][:,:4,:], # 1 x 8 x 4
                        #         gripper_actions=src_subtask_gripper_actions[0:1], # 1 x 1
                        #         action_noise=cur_phase_task_spec[0][subtask_ind]["action_noise"],
                        #     )
                        # elif arm_name == 'arm_right':
                        #     # Interpolation segment will start from current robot eef pose.
                        #     init_sequence = WaypointSequence.from_poses(
                        #         poses=cur_datagen_info.eef_pose[None][:,4:,:], # 1 x 4 x 4
                        #         gripper_actions=src_subtask_gripper_actions[0:1], # 1 x 1
                        #         action_noise=cur_phase_task_spec[1][subtask_ind]["action_noise"],
                        #     )

                        # print('init_sequence[0].pose.shape', init_sequence[0].pose.shape) # 4 x 4
                        # traj_to_execute.add_waypoint_sequence(init_sequence)

                        # Construct trajectory for the transformed segment.
                        transformed_seq = WaypointSequence.from_poses(
                            poses=transformed_eef_poses, # 107 x 4 x 4
                            gripper_actions=src_subtask_gripper_actions,
                            action_noise=local_task_spec[subtask_ind]["action_noise"],
                        )
                        transformed_traj = WaypointTrajectory()
                        transformed_traj.add_waypoint_sequence(transformed_seq, env_idx=env_idx)
                        # print('transformed_traj[10].pose.shape', transformed_traj[10].pose.shape) # 8 x 4

                        # Merge this trajectory into our trajectory using linear interpolation.
                        # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
                        # traj_to_execute.merge(
                        #     transformed_traj,
                        #     num_steps_interp=local_task_spec[subtask_ind]["num_interpolation_steps"],
                        #     num_steps_fixed=local_task_spec[subtask_ind]["num_fixed_steps"],
                        #     action_noise=(float(local_task_spec[subtask_ind]["apply_noise_during_interpolation"]) * local_task_spec[subtask_ind]["action_noise"]),
                        #     bimanual=self.bimanual
                        # )

                        # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
                        # for us. However, we can safely discard that first waypoint now, and just start by executing
                        # the rest of the trajectory (interpolation segment and transformed subtask segment).
                        # traj_to_execute.pop_first()

                        traj_to_execute = transformed_traj

                        # print('*****************************')
                        # print('finished processing one subtask for one arm')
                        # print('num sequences:', len(traj_to_execute.waypoint_sequences))
                        # for seq in traj_to_execute.waypoint_sequences:
                        #     print('num waypoints:', len(seq.sequence))
                    
                        traj_list_all[arm_i].append(traj_to_execute)
                
                    # breakpoint()
                    traj_to_execute = self.merge_trajs(traj_list_all)

                    # reformat the local info with the current subtask start and end steps
                    # for example, this converts MP_end_steps [2770, 2050] to [550, 330]
                    # TODO: the logic here can be problematic when other demonstration annotations, need to double check with other data demonstrations
                    for i in range(2):
                        # Clip between selected_src_subtask_inds[0] and selected_src_subtask_inds[1]
                        MP_end_steps[i] = min(max(MP_end_steps[i], selected_src_subtask_inds[0]), selected_src_subtask_inds[1])
                        MP_end_steps[i] -= selected_src_subtask_inds[0]

                    if change_role:
                        MP_end_steps = MP_end_steps[::-1]
                        # TODO: need to change the attached_obj_dict as well

                    MP_end_steps_per_env.append(MP_end_steps)
                    traj_to_execute_all_env.append(traj_to_execute)
                    # breakpoint()


                # print('MP_end_steps', MP_end_steps_per_env)
                # breakpoint()

                # Execute the trajectory and collect data.
                results_all_env = traj_to_execute.execute(
                    env=env,
                    env_interfaces=env_interfaces,
                    render=render,
                    video_writer=video_writer,
                    video_skip=video_skip,
                    camera_names=camera_names,
                    bimanual=self.bimanual,
                    cur_subtask_end_step_MP=MP_end_steps_per_env,
                    # attached_obj=attached_obj[phase_ind][subtask_ind_reordered],
                    attached_obj=attached_obj_dict,
                    phase_type=self.task_spec[phase_ind][0][0]["phase_type"],
                    object_ref=object_ref,
                    # TODO: Explain why I am doing this. Passing the trajectory to execute for all environments here instead of setting it in the waypoint class
                    traj_to_execute_all_env=traj_to_execute_all_env,
                    ignore_failed_mp=self.ignore_failed_mp
                )
                print(f"Phase {phase_ind} Subtask ind: {subtask_ind_reordered} env.valid_envs: ", env.valid_envs)
                # breakpoint()

                # TODO: Fix this
                # if exec_results is None:
                #     print('failed to execute the trajectory, breakpoint in data_generator.py')
                #     return None

                # check that trajectory is non-empty
                for env_idx, exec_results in enumerate(results_all_env):
                    if exec_results is None:
                        print(f'failed to execute the trajectory for env {env_idx}, breakpoint in data_generator.py')
                    
                    if len(exec_results["states"]) > 0:
                        generated_states[env_idx] = generated_states[env_idx] + exec_results["states"]
                        generated_obs[env_idx] += exec_results["observations"].tolist()   # FIXME: observations are empty dicts right now. 
                        generated_datagen_infos[env_idx] += exec_results["datagen_infos"].tolist()
                        generated_actions[env_idx] += exec_results["actions"].tolist()
                        generated_demo_mp_end_steps[env_idx].append(exec_results["mp_end_steps"])
                        generated_demo_subtask_lengths[env_idx].append(exec_results["subtask_lengths"])
                        # TODO: currently only checking if MP fails for determining success. Use task success as well
                        # generated_success[env_idx] = generated_success[env_idx] or exec_results["success"]
                        generated_success[env_idx] = env.valid_envs[env_idx]
                        generated_src_demo_inds[env_idx].append(selected_src_demo_ind)
                        val = selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int)
                        val = val.tolist()
                        generated_src_demo_labels[env_idx] += val

                # remember last trajectory
                prev_executed_traj = traj_to_execute

                if pause_subtask:
                    input("Pausing after subtask {} execution. Press any key to continue...".format(subtask_ind))

        # TODO: why need to merge the generated actions
        # merge numpy arrays
        # if len(generated_src_demo_labels) > 0:
            # generated_actions = np.concatenate(generated_actions, axis=0)
            # generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=1)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states, # NOTE: btw these states have data for all env. If er need per env, we need to change this!
            observations=generated_obs,
            datagen_infos=generated_datagen_infos,
            actions=np.array(generated_actions),
            success=generated_success,
            src_demo_inds=generated_src_demo_inds,
            src_demo_labels=generated_src_demo_labels,
            mp_end_steps=generated_demo_mp_end_steps,
            subtask_lengths=generated_demo_subtask_lengths,
            external_sensor_info=external_sensor_info,
        )
        print('before returning the results')
        return results