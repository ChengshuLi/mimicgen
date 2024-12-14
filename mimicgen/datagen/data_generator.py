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
        print('#################### in merge trajectories ####################')
        
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
                    print('num waypoints:', len(waypoint_traj.waypoint_sequences[-1].sequence))
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

    # for tiago cup
    def generate(
        self,
        env,
        env_interface,
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

        # set camera postion
        import omnigibson as og
        import torch as th
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 1.7492, -0.0424,  1.5371]),
            orientation=th.tensor([0.3379, 0.3417, 0.6236, 0.6166]),
        )
        for _ in range(5): og.sim.render()

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

        # (Pdb) p all_subtask_inds_structure
        # [[array([[[  0, 309],
        # [309, 650]]]), array([[[  0, 360],
        # [360, 650]]])], [array([[[650, 992]]]), array([[[650, 992]]])]]

        # some state variables used during generation
        selected_src_demo_ind = None
        prev_executed_traj = None

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_datagen_infos = []
        generated_actions = []
        generated_success = False
        generated_src_demo_inds = [] # store selected src demo ind for each subtask in each trajectory
        generated_src_demo_labels = [] # like @generated_src_demo_inds, but padded to align with size of @generated_actions

        # for left arms first
        for phase_ind in range(self.num_phases):
            cur_phase_task_spec = self.task_spec[phase_ind]
            selected_src_demo_ind = 0 # TODO: since we only have one demo, will need to modify if more demos are available

            # restructure subtasks indexes and reference objects
            all_subtask_inds = all_subtask_inds_structure[phase_ind]
            subtask_ind_vals = np.sort(np.unique(all_subtask_inds))
            num_subtasks = len(subtask_ind_vals) - 1


            for subtask_ind_reordered in range(num_subtasks):

                selected_src_subtask_inds = subtask_ind_vals[subtask_ind_reordered : subtask_ind_reordered + 2] # [start_step, end_step]
                traj_list_all = [[],[]]

                for arm_i, arm_name in enumerate(['arm_left', 'arm_right']):

                    # need to recalculate the matched subtask_ind to retrieve the correct task spec
                    local_task_spec = cur_phase_task_spec[arm_i]
                    arm_spec_subtask_inds = all_subtask_inds[arm_i][0]
                    arm_unique_subtask_inds = np.sort(np.unique(arm_spec_subtask_inds))
                    subtask_ind = np.where(selected_src_subtask_inds[1] <= arm_unique_subtask_inds)[0][0] - 1

                    print('==========================================')
                    print('arm_name:', arm_name, 'subtask_ind_reordered', subtask_ind_reordered, 'subtask_ind:', subtask_ind)
                    print('subtask start and end step', selected_src_subtask_inds)
                    print('arm_spec_subtask_inds', arm_spec_subtask_inds)

                    is_first_subtask = (subtask_ind == 0) and (phase_ind == 0)
                    is_first_subtask_in_phase = (subtask_ind == 0)

                    cur_datagen_info = env_interface.get_datagen_info()
                    subtask_object_name = cur_phase_task_spec[arm_i][subtask_ind]["object_ref"]
                    cur_object_pose = cur_datagen_info.object_poses[subtask_object_name] if (subtask_object_name is not None) else None # 4x4
                    print('subtask_object_name', subtask_object_name)
                    
                    # get poses
                    src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]
                    src_subtask_eef_poses = src_ep_datagen_info.eef_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
                    src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 2

                    if arm_name == 'arm_left':
                        src_subtask_eef_poses = src_subtask_eef_poses[:,:4,:]
                        src_subtask_gripper_actions = src_subtask_gripper_actions[:,:1]
                    elif arm_name == 'arm_right':
                        src_subtask_eef_poses = src_subtask_eef_poses[:,4:,:]
                        src_subtask_gripper_actions = src_subtask_gripper_actions[:,1:]

                    # get reference object pose from source demo
                    src_subtask_object_pose = src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]] if (subtask_object_name is not None) else None # 4 x 4

                    # account for extra timestep added to @src_eef_poses
                    src_subtask_gripper_actions = np.concatenate([src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0) # 107 x2

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
                    traj_to_execute = WaypointTrajectory()

                    if arm_name == 'arm_left':
                        # Interpolation segment will start from current robot eef pose.
                        init_sequence = WaypointSequence.from_poses(
                            poses=cur_datagen_info.eef_pose[None][:,:4,:], # 1 x 8 x 4
                            gripper_actions=src_subtask_gripper_actions[0:1],
                            action_noise=cur_phase_task_spec[0][subtask_ind]["action_noise"],
                        )
                    elif arm_name == 'arm_right':
                        # Interpolation segment will start from current robot eef pose.
                        init_sequence = WaypointSequence.from_poses(
                            poses=cur_datagen_info.eef_pose[None][:,4:,:], # 1 x 4 x 4
                            gripper_actions=src_subtask_gripper_actions[0:1],
                            action_noise=cur_phase_task_spec[1][subtask_ind]["action_noise"],
                        )

                    traj_to_execute.add_waypoint_sequence(init_sequence)

                    # Construct trajectory for the transformed segment.
                    transformed_seq = WaypointSequence.from_poses(
                        poses=transformed_eef_poses, # 107 x 4 x 4
                        gripper_actions=src_subtask_gripper_actions,
                        action_noise=local_task_spec[subtask_ind]["action_noise"],
                    )
                    transformed_traj = WaypointTrajectory()
                    transformed_traj.add_waypoint_sequence(transformed_seq)
                    # print('transformed_traj[10].pose.shape', transformed_traj[10].pose.shape) # 8 x 4

                    # Merge this trajectory into our trajectory using linear interpolation.
                    # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
                    traj_to_execute.merge(
                        transformed_traj,
                        num_steps_interp=local_task_spec[subtask_ind]["num_interpolation_steps"],
                        num_steps_fixed=local_task_spec[subtask_ind]["num_fixed_steps"],
                        action_noise=(float(local_task_spec[subtask_ind]["apply_noise_during_interpolation"]) * local_task_spec[subtask_ind]["action_noise"]),
                        bimanual=self.bimanual
                    )

                    # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
                    # for us. However, we can safely discard that first waypoint now, and just start by executing
                    # the rest of the trajectory (interpolation segment and transformed subtask segment).
                    traj_to_execute.pop_first()

                    print('*****************************')
                    print('finished processing one subtask for one arm')
                    print('num sequences:', len(traj_to_execute.waypoint_sequences))
                    for seq in traj_to_execute.waypoint_sequences:
                        print('num waypoints:', len(seq.sequence))
                
                    traj_list_all[arm_i].append(traj_to_execute)
                
                traj_to_execute = self.merge_trajs(traj_list_all)

                # now still execute each subtask separately
                import pdb; pdb.set_trace()
        
                # Execute the trajectory and collect data.
                exec_results = traj_to_execute.execute(
                    env=env,
                    env_interface=env_interface,
                    render=render,
                    video_writer=video_writer,
                    video_skip=video_skip,
                    camera_names=camera_names,
                    bimanual=self.bimanual,
                )

            # check that trajectory is non-empty
            if len(exec_results["states"]) > 0:
                generated_states += exec_results["states"]
                generated_obs += exec_results["observations"]
                generated_datagen_infos += exec_results["datagen_infos"]
                generated_actions.append(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]
                generated_src_demo_inds.append(selected_src_demo_ind)
                generated_src_demo_labels.append(selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int))

            # remember last trajectory
            prev_executed_traj = traj_to_execute

            if pause_subtask:
                input("Pausing after subtask {} execution. Press any key to continue...".format(subtask_ind))

        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = np.concatenate(generated_actions, axis=0)
            generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            datagen_infos=generated_datagen_infos,
            actions=generated_actions,
            success=generated_success,
            src_demo_inds=generated_src_demo_inds,
            src_demo_labels=generated_src_demo_labels,
        )
        import pdb; pdb.set_trace()
        print('before retuning the results')
        return results

    # work for other tasks
    def generate_old_version(
        self,
        env,
        env_interface,
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

        # TODO: okay here actually the length of each subtask is randomized
        # example output, all_subtask_inds = array([[[  0, 106], [106, 187]]])

        # import pdb; pdb.set_trace()
        # sample new subtask boundaries
        all_subtask_inds = self.randomize_subtask_boundaries(self.src_subtask_indices, self.task_spec) # shape [N, S, 2], last dim is start and end action lengths

        # some state variables used during generation
        selected_src_demo_ind = None
        prev_executed_traj = None

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_datagen_infos = []
        generated_actions = []
        generated_success = False
        generated_src_demo_inds = [] # store selected src demo ind for each subtask in each trajectory
        generated_src_demo_labels = [] # like @generated_src_demo_inds, but padded to align with size of @generated_actions

        for subtask_ind in range(len(self.task_spec)): # for each subtask

            # some things only happen on first subtask
            is_first_subtask = (subtask_ind == 0)

            # get datagen info in current environment to get required info for selection (e.g. eef pose, object pose)
            cur_datagen_info = env_interface.get_datagen_info()

            # name of object for this subtask
            subtask_object_name = self.task_spec[subtask_ind]["object_ref"]

            # corresponding current object pose
            cur_object_pose = cur_datagen_info.object_poses[subtask_object_name] if (subtask_object_name is not None) else None

            # We need source demonstration selection for the first subtask (always), and possibly for 
            # other subtasks if @select_src_per_subtask is set.
            need_source_demo_selection = (is_first_subtask or select_src_per_subtask)

            # Run source demo selection or use selected demo from previous iteration
            if need_source_demo_selection:
                selected_src_demo_ind = self.select_source_demo(
                    eef_pose=cur_datagen_info.eef_pose,
                    object_pose=cur_object_pose,
                    subtask_ind=subtask_ind,
                    src_subtask_inds=all_subtask_inds[:, subtask_ind],
                    subtask_object_name=subtask_object_name,
                    selection_strategy_name=self.task_spec[subtask_ind]["selection_strategy"],
                    selection_strategy_kwargs=self.task_spec[subtask_ind]["selection_strategy_kwargs"],
                )
            assert (selected_src_demo_ind is not None)

            # selected subtask segment time indices
            selected_src_subtask_inds = all_subtask_inds[selected_src_demo_ind, subtask_ind]

            # get subtask segment, consisting of the sequence of robot eef poses, target poses, gripper actions
            src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]
            src_subtask_eef_poses = src_ep_datagen_info.eef_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
            src_subtask_target_poses = src_ep_datagen_info.target_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
            src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 2
            
            # get reference object pose from source demo
            src_subtask_object_pose = src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]] if (subtask_object_name is not None) else None # 4 x 4, TODO: okay need to make sure the object is static during the subtask?

            if is_first_subtask or transform_first_robot_pose:
                # Source segment consists of first robot eef pose and the target poses. This ensures that
                # we will interpolate to the first robot eef pose in this source segment, instead of the
                # first robot target pose.
                # TODO: not sure about the meaning of this
                src_eef_poses = np.concatenate([src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0) # 107 x 8 x 4
            else:
                # Source segment consiseef_posets of just the target poses.
                src_eef_poses = np.array(src_subtask_target_poses)

            # account for extra timestep added to @src_eef_poses
            src_subtask_gripper_actions = np.concatenate([src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0) # 107 x2

            # Transform source demonstration segment using relevant object pose.
            if subtask_object_name is not None:
                if self.bimanual:
                    # the bimanual setting
                    # src_eef_poses, [num_steps, 8, 4]
                    transformed_eef_poses_left = PoseUtils.transform_source_data_segment_using_object_pose(obj_pose=cur_object_pose, src_eef_poses=src_eef_poses[:,:4,:],src_obj_pose=src_subtask_object_pose)
                    transformed_eef_poses_right = PoseUtils.transform_source_data_segment_using_object_pose(obj_pose=cur_object_pose, src_eef_poses=src_eef_poses[:,4:,:],src_obj_pose=src_subtask_object_pose)
                    transformed_eef_poses = np.concatenate([transformed_eef_poses_left, transformed_eef_poses_right], axis=1)
                else:
                    transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                        obj_pose=cur_object_pose, # 4x4
                        src_eef_poses=src_eef_poses, 
                        src_obj_pose=src_subtask_object_pose, # 4x4
                    )
            else:
                # skip transformation if no reference object is provided
                transformed_eef_poses = src_eef_poses
            
            # We will construct a WaypointTrajectory instance to keep track of robot control targets 
            # that will be executed and then execute it.
            traj_to_execute = WaypointTrajectory()

            # TODO: change the interpolation to curobo motion planner

            if interpolate_from_last_target_pose and (not is_first_subtask):
                # Interpolation segment will start from last target pose (which may not have been achieved).
                assert prev_executed_traj is not None
                last_waypoint = prev_executed_traj.last_waypoint
                init_sequence = WaypointSequence(sequence=[last_waypoint])
            else:
                # Interpolation segment will start from current robot eef pose.
                init_sequence = WaypointSequence.from_poses(
                    poses=cur_datagen_info.eef_pose[None], # 1 x 8 x 4
                    gripper_actions=src_subtask_gripper_actions[0:1],
                    action_noise=self.task_spec[subtask_ind]["action_noise"],
                )
            print('init_sequence[0].pose.shape', init_sequence[0].pose.shape) # 8 x 4
            traj_to_execute.add_waypoint_sequence(init_sequence)

            # Construct trajectory for the transformed segment.
            transformed_seq = WaypointSequence.from_poses(
                poses=transformed_eef_poses, # 107 x 8 x 4
                gripper_actions=src_subtask_gripper_actions,
                action_noise=self.task_spec[subtask_ind]["action_noise"],
            )
            transformed_traj = WaypointTrajectory()
            transformed_traj.add_waypoint_sequence(transformed_seq)
            # print('transformed_traj[10].pose.shape', transformed_traj[10].pose.shape) # 8 x 4

            # Merge this trajectory into our trajectory using linear interpolation.
            # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
            traj_to_execute.merge(
                transformed_traj,
                num_steps_interp=self.task_spec[subtask_ind]["num_interpolation_steps"],
                num_steps_fixed=self.task_spec[subtask_ind]["num_fixed_steps"],
                action_noise=(float(self.task_spec[subtask_ind]["apply_noise_during_interpolation"]) * self.task_spec[subtask_ind]["action_noise"]),
            )

            # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
            # for us. However, we can safely discard that first waypoint now, and just start by executing
            # the rest of the trajectory (interpolation segment and transformed subtask segment).
            traj_to_execute.pop_first()

            # Execute the trajectory and collect data.
            exec_results = traj_to_execute.execute(
                env=env,
                env_interface=env_interface,
                render=render,
                video_writer=video_writer,
                video_skip=video_skip,
                camera_names=camera_names,
            )

            # check that trajectory is non-empty
            if len(exec_results["states"]) > 0:
                generated_states += exec_results["states"]
                generated_obs += exec_results["observations"]
                generated_datagen_infos += exec_results["datagen_infos"]
                generated_actions.append(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]
                generated_src_demo_inds.append(selected_src_demo_ind)
                generated_src_demo_labels.append(selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int))

            # remember last trajectory
            prev_executed_traj = traj_to_execute

            if pause_subtask:
                input("Pausing after subtask {} execution. Press any key to continue...".format(subtask_ind))

        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = np.concatenate(generated_actions, axis=0)
            generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            datagen_infos=generated_datagen_infos,
            actions=generated_actions,
            success=generated_success,
            src_demo_inds=generated_src_demo_inds,
            src_demo_labels=generated_src_demo_labels,
        )
        return results