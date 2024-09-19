# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
MimicGen environment interface classes for basic robosuite environments.
"""
import numpy as np

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import *

from mimicgen.env_interfaces.base import MG_EnvInterface

import torch as th

import pdb


class OmniGibsonInterface(MG_EnvInterface):
    """
    MimicGen environment interface base class for basic robosuite environments.
    """

    # Note: base simulator interface class must fill out interface type as a class property
    INTERFACE_TYPE = "omnigibson"

    def __init__(self, env):
        super(OmniGibsonInterface, self).__init__(env)
        self._setup_arm_controller()

    def _setup_arm_controller(self):
        """
        Sets up the arm controller for the robot. This is necessary to know where the arm command
        starts and ends in the action vector.
        """
        self.robot = self.env.robots[0]
        start_idx = 0
        end_idx = None
        arm_controller = None
        for controller_type, controller in self.robot.controllers.items():
            if controller_type != f"arm_{self.robot.default_arm}":
                start_idx += controller.command_dim
            else:
                end_idx = start_idx + controller.command_dim
                arm_controller = controller
                break
        assert end_idx is not None and arm_controller is not None
        self.arm_command_start_idx = start_idx
        self.arm_command_end_idx = end_idx
        self.arm_controller = arm_controller

    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """
        return self.get_object_pose(self.robot.eef_links[self.robot.default_arm])

    # Copy from BaseController of OG
    def _preprocess_command(self, command):
        """
        Clips + scales inputted @command according to self.command_input_limits and self.command_output_limits.
        If self.command_input_limits is None, then no clipping will occur. If either self.command_input_limits
        or self.command_output_limits is None, then no scaling will occur.

        Args:
            command (Array[float] or float): Inputted command vector

        Returns:
            Array[float]: Processed command vector
        """
        # Make sure command is a th.tensor
        command = th.tensor([command]) if type(command) in {int, float} else command
        # We only clip and / or scale if self.command_input_limits exists
        if self.arm_controller._command_input_limits is not None:
            # Clip
            command = command.clip(*self.arm_controller._command_input_limits)
            if self.arm_controller._command_output_limits is not None:
                # If we haven't calculated how to scale the command, do that now (once)
                if self.arm_controller._command_scale_factor is None:
                    self.arm_controller._command_scale_factor = abs(
                        self.arm_controller._command_output_limits[1] - self.arm_controller._command_output_limits[0]
                    ) / abs(self.arm_controller._command_input_limits[1] - self.arm_controller._command_input_limits[0])
                    self.arm_controller._command_output_transform = (
                        self.arm_controller._command_output_limits[1] + self.arm_controller._command_output_limits[0]
                    ) / 2.0
                    self.arm_controller._command_input_transform = (
                        self.arm_controller._command_input_limits[1] + self.arm_controller._command_input_limits[0]
                    ) / 2.0
                # Scale command
                command = (
                    command - self.arm_controller._command_input_transform
                ) * self.arm_controller._command_scale_factor + self.arm_controller._command_output_transform

        # Return processed command
        return command

    def _undo_preprocess_command(self, command):
        """
        The reverse of @preprocess_command. Takes a command that has been scaled and convert it back to the original command
        """
        # Make sure command is a th.tensor
        command = th.tensor([command]) if type(command) in {int, float} else command
        # We only clip and / or scale if self.command_input_limits exists
        if self.arm_controller._command_input_limits is not None:
            if self.arm_controller._command_output_limits is not None:
                # If we haven't calculated how to scale the command, do that now (once)
                if self.arm_controller._command_scale_factor is None:
                    self.arm_controller._command_scale_factor = abs(
                        self.arm_controller._command_output_limits[1] - self.arm_controller._command_output_limits[0]
                    ) / abs(self.arm_controller._command_input_limits[1] - self.arm_controller._command_input_limits[0])
                    self.arm_controller._command_output_transform = (
                        self.arm_controller._command_output_limits[1] + self.arm_controller._command_output_limits[0]
                    ) / 2.0
                    self.arm_controller._command_input_transform = (
                        self.arm_controller._command_input_limits[1] + self.arm_controller._command_input_limits[0]
                    ) / 2.0

                # Unscale command
                command = (command - self.arm_controller._command_output_transform) / self.arm_controller._command_scale_factor + self.arm_controller._command_input_transform

            # No need to unclip
        return command

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller (in the world frame) and returns an action
        (usually a normalized delta pose action in the robot frame) to try and achieve that target pose.

        Args:
            target_pose (np.array): 4x4 target eef pose, in the world frame

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation), in the robot frame
        """
        # Legacy
        del relative

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)

        # Compute the eef target pose in the robot frame
        target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

        # Get the current eef pose in the robot frame
        pos_relative, quat_relative = self.robot.get_relative_eef_pose()

        # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
        dpos = target_pos - pos_relative

        dori = T.mat2quat(T.quat2mat(target_quat) @ T.quat2mat(quat_relative).T)
        dori = T.quat2axisangle(dori)

        # Assemble the arm command and undo the preprocessing
        arm_command = th.cat([dpos, dori])
        arm_command = self._undo_preprocess_command(arm_command)

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = th.from_numpy(np.zeros_like(self.robot.action_space.sample())[:-1])
        action[self.arm_command_start_idx:self.arm_command_end_idx] = arm_command

        # Convert to numpy tensor
        action = action.numpy()

        return action

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        # Legacy
        del relative

        # Ensure float32
        action = action.astype(np.float32)

        # Convert to torch tensor
        action = th.from_numpy(action)

        # Extract the arm command part of the action and preprocess it
        arm_command = action[self.arm_command_start_idx:self.arm_command_end_idx]
        arm_command = self._preprocess_command(arm_command)

        # Get the current eef pose in the robot frame
        pos_relative, quat_relative = self.robot.get_relative_eef_pose()

        # Extract the delta pose from the arm command and compute the target pose in the robot frame
        dpos = arm_command[:3]
        target_pos = pos_relative + dpos
        dori = T.quat2mat(T.axisangle2quat(arm_command[3:6]))
        target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Convert the target pose to the world frame
        target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
        target_pose = target_pose.numpy()

        # Sanity check cycle consistency (not technically necessary)
        new_action = self.target_pose_to_action(target_pose)
        # @new_action has one less element than @action because it doesn't have the gripper actuation
        assert th.allclose(action[:-1], th.from_numpy(new_action), atol=1e-2)

        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        # last dimension is gripper action
        return action[-1:]

    def get_object_pose(self, obj):
        """
        Returns 4x4 object pose given the name of the object and the type.

        Args:
            obj (BaseObject): OG object

        Returns:
            obj_pose (np.array): 4x4 object pose
        """
        return T.pose2mat(obj.get_position_orientation())


class MG_TestPenBook(OmniGibsonInterface):
    """
    Corresponds to OG TestPenBook task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        # two relevant objects - eraser and book
        return dict(
            eraser=self.get_object_pose(obj=self.env.task.object_scope["rubber_eraser.n.01_1"]),
            book=self.get_object_pose(obj=self.env.task.object_scope["hardback.n.01_1"]),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # The signal is when the robot grasps the eraser
        signals["grasp"] = int(self.robot.is_grasping(arm="default", candidate_obj=self.env.task.object_scope["rubber_eraser.n.01_1"]))

        # final subtask is placing the eraser on the book (motion relative to book) - but final subtask signal is not needed
        return signals


from omnigibson.object_states import *

class MG_TestCabinet(OmniGibsonInterface):
    """
    Corresponds to OG TestCabinet task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        # one relevant object - cabinet
        return dict(
            cabinet=self.get_object_pose(obj=self.env.task.object_scope["cabinet.n.01_1"]),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # The signal is when the robot touches the cabinet
        signals["grasp"] = int(self.robot.states[Touching].get_value(self.env.task.object_scope["cabinet.n.01_1"]))

        # final subtask is pulling the drawer open - but final subtask signal is not needed
        return signals