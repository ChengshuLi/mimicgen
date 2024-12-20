# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of classes used to represent waypoints and trajectories.
"""
import json
import numpy as np
from copy import deepcopy

import mimicgen
import mimicgen.utils.pose_utils as PoseUtils
import pdb
import copy

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection
import torch as th


class Waypoint(object):
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (np.array): 4x4 pose target for robot controller
            gripper_action (np.array): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        self.pose = np.array(pose)
        self.gripper_action = np.array(gripper_action)
        self.noise = noise
        assert len(self.gripper_action.shape) == 1
    
    def merge_wp(self, other):
        """
        Merge another Waypoint object into this one.
        """
        self.pose = np.concatenate([self.pose, other.pose], axis=0)
        self.gripper_action = np.concatenate([self.gripper_action, other.gripper_action], axis=0)
        self.noise = min(self.noise, other.noise)
        # TODO
        self.noise = 0.0


class WaypointSequence(object):
    """
    Represents a sequence of Waypoint objects.
    """
    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be an list of Waypoint objects
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses, 
        gripper actions, and action noise.

        Args:
            poses (np.array): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (np.array): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or np.array): sequence of action noise
                magnitudes that should be applied at each timestep. If a 
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        assert isinstance(action_noise, float) or isinstance(action_noise, np.ndarray)

        # handle scalar to numpy array conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * np.ones((num_timesteps, 1))
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)

    def merge(self, other):
        """
        Merge another WaypointSequence object into this one.
        """
        self.sequence += other.sequence

class WaypointTrajectory(object):
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.
        
        Returns:
            waypoint (Waypoint instance)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (np.array): 4x4 target pose

            gripper_action (np.array): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        if (len(self.waypoint_sequences) == 0):
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = np.array([pose for _ in range(num_steps)])
            gripper_actions = np.array([[gripper_action] for _ in range(num_steps)])
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            if last_waypoint.pose.shape[0] == 8:
                # here is when transforming the two arms altogher, should be corresponding to the bimanual-coordinated phase
                poses_left, num_steps_2_left = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[0:4, :],
                    pose_2=pose[0:4, :],
                    num_steps=num_steps,
                )
                poses_right, num_steps_2_right = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[4:, :],
                    pose_2=pose[4:, :],
                    num_steps=num_steps,
                )
                poses = np.concatenate([poses_left, poses_right], axis=1)
                assert num_steps_2_left == num_steps_2_right
                num_steps_2 = num_steps_2_left
            else:
                # suitable for single arm transformation
                poses, num_steps_2 = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose,
                    pose_2=pose,
                    num_steps=num_steps,
                )
            assert num_steps == num_steps_2
            gripper_actions = np.array([gripper_action for _ in range(num_steps + 2)])
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant 
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = (need_interp or need_fixed)

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose, # 8x4
                    gripper_action=target_for_interpolation.gripper_action, #2,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                    bimanual=bimanual,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                    bimanual=bimanual,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def execute(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
    ):
        """
        Main function to execute the trajectory. Will use env_interface.target_pose_to_action to
        convert each target pose at each waypoint to an action command, and pass that along to
        env.step.

        Args:
            env (robomimic EnvBase instance): environment to use for executing trajectory
            env_interface (MG_EnvInterface instance): environment interface for executing trajectory
            render (bool): if True, render on-screen
            video_writer (imageio writer): video writer
            video_skip (int): determines rate at which environment frames are written to video
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
            cur_subtask_end_step_MP: list of size 2, the end point of motion planner for two arms

        Returns:
            results (dict): dictionary with the following items for the executed trajectory:
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (list): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
        """
        robot = env.env.robots[0]
        # write_video = (video_writer is not None)
        # video_count = 0

        states = []
        actions = []
        observations = []
        datagen_infos = []
        success = {k: False for k in env.is_success()} # success metrics

        assert len(self.waypoint_sequences) == 1
        seq = self.waypoint_sequences[0]
        for end_step in cur_subtask_end_step_MP:
            assert 0 <= end_step <= len(seq)

        # Segment the waypoints into motion planner waypoints and replay waypoints
        left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
        left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
        right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
        right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]

        # Get the last waypoint for padding later
        last_waypoint = seq[-1]

        # print("start")
        # breakpoint()
        # TODO: potentially make waypoints more dense

        # If there are motion planner waypoints
        # 1. make sure the gripper actions are the same
        # 2. get the last waypoint's pose and orientation as the MP target
        # Otherwise, use the current eef pose as the MP target
        if len(left_mp_waypoints) > 0:
            gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
            assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
            left_waypoint = left_mp_waypoints[-1]
            left_gripper_action = left_waypoint.gripper_action
            left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
        else:
            left_gripper_action = None
            left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

        if len(right_mp_waypoints) > 0:
            gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
            assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
            right_waypoint = right_mp_waypoints[-1]
            right_gripper_action = right_waypoint.gripper_action
            right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
        else:
            right_gripper_action = None
            right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")

        # If at least one hand has motion planner waypoints, plan the motion
        if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
            target_pos = {
                robot.eef_link_names["left"]: left_waypoint_pos,
                robot.eef_link_names["right"]: right_waypoint_pos,
            }
            target_quat = {
                robot.eef_link_names["left"]: left_waypoint_ori,
                robot.eef_link_names["right"]: right_waypoint_ori,
            }
            emb_sel = CuroboEmbodimentSelection.ARM

            # Attached the object to the robot for planning
            if attached_obj is None:
                attached_obj_scale = None
            else:
                attached_obj_new = {}
                attached_obj_scale = {}
                for arm, obj_name in attached_obj.items():
                    attached_obj_new[robot.eef_link_names[arm]] = env.env.scene.object_registry("name", obj_name).root_link
                    attached_obj_scale[robot.eef_link_names[arm]] = 0.9
                attached_obj = attached_obj_new

            # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
            successes, traj_paths = env.cmg.compute_trajectories(
                target_pos=target_pos,
                target_quat=target_quat,
                is_local=False,
                max_attempts=50,
                timeout=60.0,
                ik_fail_return=5,
                enable_finetune_trajopt=True,
                finetune_attempts=1,
                return_full_result=False,
                success_ratio=1.0,
                attached_obj=attached_obj,
                attached_obj_scale=attached_obj_scale,
                emb_sel=emb_sel,
            )
            # TODO: These lines are for debugging purposes.
            # successes, traj_paths = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=False, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)
            # full_result = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=True, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)
            # env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
            # env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
            # env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
            # env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)
            success_status, traj_path = successes[0], traj_paths[0]
            # print("success status", success_status)
            # breakpoint()
            assert success_status, "motion planning failed"

            # Convert planned joint trajectory to actions
            q_traj = env.cmg.path_to_joint_trajectory(traj_path, emb_sel).cpu()
            mp_actions = []
            for j_pos in q_traj:
                action = np.zeros(robot.action_dim)
                for name, controller in robot.controllers.items():
                    if "gripper" not in name:
                        command = j_pos[controller.dof_idx]
                        partial_action = controller._reverse_preprocess_command(command)
                        action_idx = robot.controller_action_idx[name]
                        action[action_idx] = partial_action

                # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                if left_gripper_action is not None:
                    action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                if right_gripper_action is not None:
                    action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                mp_actions.append(action)

            # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
            if len(left_mp_waypoints) == 0:
                # We need to pad the left hand waypoints to match the length of the MP trajectory
                if len(left_replay_waypoints) < len(mp_actions):
                    for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                        left_replay_waypoints.append(last_waypoint)

                # We convert the target pose of the left hand to replay_action
                # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                for i, action in enumerate(mp_actions):
                    replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                    action_idx = robot.controller_action_idx["arm_left"]
                    action[action_idx] = replay_action[action_idx]
                    action[env_interface.gripper_action_dim[0]] = left_replay_waypoints[i].gripper_action[0]

                # We remove the waypoints that have been replayed for the left arm
                left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

            # Same logic as above but for the right hand
            elif len(right_mp_waypoints) == 0:
                if len(right_replay_waypoints) < len(mp_actions):
                    for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                        right_replay_waypoints.append(last_waypoint)
                for i, action in enumerate(mp_actions):
                    replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                    action_idx = robot.controller_action_idx["arm_right"]
                    action[action_idx] = replay_action[action_idx]
                    action[env_interface.gripper_action_dim[1]] = right_replay_waypoints[i].gripper_action[1]

                right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

            # For each motion planner action, we repeat it 3 times for the controllers to converge
            num_repeat = 3
            for mp_action in mp_actions:
                for _ in range(num_repeat):
                    state = env.get_state()["states"]
                    obs = env.get_observation()
                    datagen_info = env_interface.get_datagen_info(action=mp_action)
                    env.step(mp_action)
                    states.append(state)
                    actions.append(mp_action)
                    observations.append(obs)
                    datagen_infos.append(datagen_info)
                    cur_success_metrics = env.is_success()
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

            # print("MP actions")
            # breakpoint()

        # Now we move on to the replay phase
        # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
        if len(left_replay_waypoints) < len(right_replay_waypoints):
            for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                left_replay_waypoints.append(last_waypoint)
        elif len(right_replay_waypoints) < len(left_replay_waypoints):
            for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                right_replay_waypoints.append(last_waypoint)

        # print("before replay actions")
        # breakpoint()

        # For each pair of waypoints, we extract the pose for each hand and then convert to action
        # We also overwrite the gripper actions with the ones from the waypoints
        for left_waypoint, right_waypoint in zip(left_replay_waypoints, right_replay_waypoints):
            pose = np.zeros((8, 4))
            pose[:4, :] = left_waypoint.pose[:4, :]
            pose[4:, :] = right_waypoint.pose[4:, :]
            replay_action = env_interface.target_pose_to_action(target_pose=pose)
            replay_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
            replay_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]

            # Update the markers for visualization
            if env.eef_current_marker_left is not None:
                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
            if env.eef_current_marker_right is not None:
                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
            if env.eef_goal_marker_left is not None:
                env.eef_goal_marker_left.set_position_orientation(position=pose[0:3, 3], orientation=T.mat2quat(th.tensor(pose[0:3, 0:3])))
            if env.eef_goal_marker_right is not None:
                env.eef_goal_marker_right.set_position_orientation(position=pose[4:7, 3], orientation=T.mat2quat(th.tensor(pose[4:7, 0:3])))

            state = env.get_state()["states"]
            obs = env.get_observation()
            datagen_info = env_interface.get_datagen_info(action=replay_action)
            env.step(replay_action)
            states.append(state)
            actions.append(replay_action)
            observations.append(obs)
            datagen_infos.append(datagen_info)
            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

        # print("replay actions")
        # breakpoint()

        # iterate over waypoint sequences
        # for seq in self.waypoint_sequences:
        #     # mp_waypoint = seq[int(seq.shape[0] * 0.9)]

        #     # iterate over waypoints in each sequence
        #     for j in range(len(seq)):

        #         # on-screen render
        #         if render:
        #             env.render(mode="human", camera_name=camera_names[0])

        #         # video render
        #         if write_video:
        #             if video_count % video_skip == 0:
        #                 video_img = []
        #                 for cam_name in camera_names:
        #                     video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
        #                 video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
        #                 video_writer.append_data(video_img)
        #             video_count += 1

        #         # current waypoint
        #         waypoint = seq[j]

        #         # current state and obs
        #         state = env.get_state()["states"]
        #         obs = env.get_observation()

        #         if bimanual:
        #             # bimanual setting
        #             # TODO: change the logic based on bimanual indicator
        #             if env.eef_current_marker_left is not None:
        #                 env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
        #             if env.eef_current_marker_right is not None:
        #                 env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
        #             if env.eef_goal_marker_left is not None:
        #                 env.eef_goal_marker_left.set_position_orientation(position=waypoint.pose[0:3, 3], orientation=T.mat2quat(th.tensor(waypoint.pose[0:3, 0:3])))
        #             if env.eef_goal_marker_right is not None:
        #                 env.eef_goal_marker_right.set_position_orientation(position=waypoint.pose[4:7, 3], orientation=T.mat2quat(th.tensor(waypoint.pose[4:7, 0:3])))
        #             # TODO: add debug component when the phase changes, maybe not needed
        #         else:
        #             # single arm setting
        #             if env.eef_current_marker is not None:
        #                 env.eef_current_marker.set_position_orientation(position=robot.get_eef_position())
        #             if env.eef_goal_marker is not None:
        #                 env.eef_goal_marker.set_position_orientation(position=waypoint.pose[0:3, 3])
                
        #         # convert target pose to arm action
        #         # TODO: the postprocessing will make tha action too large and could cause the drifting problem
        #         # Pose -> IK command -> Joint command -> Joint Controller (reload controller with joint controller)
        #         action_pose = env_interface.target_pose_to_action(target_pose=waypoint.pose)
        #         # action_pose = env_interface.target_pose_to_action_no_unprocess(target_pose=waypoint.pose)
        #         # action_pose = env_interface.generate_action(target_pose=waypoint.pose)

        #         # maybe add noise to action
        #         # if waypoint.noise is not None:
        #         #     action_pose += waypoint.noise * np.random.randn(*action_pose.shape)
                
        #         # TODO: the action_pose clip here is important, without this clip the get_datagen_info will raise error when the right hand is in contact with the coffee cup even with all the preprocess and pose process
        #         # action_pose = np.clip(action_pose, -1., 1.)

        #         if bimanual:
        #             # bimanual setting
        #             play_action = copy.deepcopy(action_pose)
        #             play_action[env_interface.gripper_action_dim] = waypoint.gripper_action
        #         else:
        #             # single arm setting
        #             # add in gripper action
        #             play_action = np.concatenate([action_pose, waypoint.gripper_action], axis=0)

        #         # store datagen info too
        #         datagen_info = env_interface.get_datagen_info(action=play_action)

        #         # step environment
        #         env.step(play_action)

        #         # collect data
        #         states.append(state)
        #         play_action_record = play_action
        #         actions.append(play_action_record)
        #         observations.append(obs)
        #         datagen_infos.append(datagen_info)

        #         cur_success_metrics = env.is_success()
        #         for k in success:
        #             success[k] = success[k] or cur_success_metrics[k]

        results = dict(
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=np.array(actions),
            success=bool(success["task"]),
        )
        return results
