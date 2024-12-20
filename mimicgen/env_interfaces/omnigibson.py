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
from omnigibson.utils.control_utils import orientation_error
from omnigibson.controllers import ControlType

from mimicgen.env_interfaces.base import MG_EnvInterface
from mimicgen.datagen.datagen_info import DatagenInfo

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
        signals["touch"] = int(self.robot.states[Touching].get_value(self.env.task.object_scope["cabinet.n.01_1"]))

        # final subtask is pulling the drawer open - but final subtask signal is not needed
        return signals


# bimanual utils
class OmniGibsonInterfaceBimanual(OmniGibsonInterface):
    """
    MimicGen environment interface class for bimanual robots.
    """
    INTERFACE_TYPE = "omnigibson_bimanual"

    def __init__(self, env):
        super(OmniGibsonInterfaceBimanual, self).__init__(env)
        self._setup_arm_controller()
        self.gripper_action_dim = th.cat([self.robot.controller_action_idx["gripper_left"], self.robot.controller_action_idx["gripper_right"]])

    def _setup_arm_controller(self):
        """
        Sets up the arm controller for the robot. This is necessary to know where the arm command
        starts and ends in the action vector.

        base <omnigibson.controllers.joint_controller.JointController object at 0x7fda15a168c0> 3
        camera <omnigibson.controllers.joint_controller.JointController object at 0x7fda15903f70> 2
        arm_left <omnigibson.controllers.ik_controller.InverseKinematicsController object at 0x7fcce648d960> 6
        gripper_left <omnigibson.controllers.multi_finger_gripper_controller.MultiFingerGripperController object at 0x7fda15a16860> 1
        arm_right <omnigibson.controllers.ik_controller.InverseKinematicsController object at 0x7fcd140c5ae0> 6
        gripper_right <omnigibson.controllers.multi_finger_gripper_controller.MultiFingerGripperController object at 0x7fcd140c5600> 1
        """
        self.robot = self.env.robots[0]
        self.arm_command_start_idx = {}
        self.arm_command_end_idx = {}
        self.arm_controller = {}
        for arm_name in ["left", "right"]:
            start_idx = 0
            end_idx = None
            arm_controller = None
            for controller_type, controller in self.robot.controllers.items():
                if controller_type != f"arm_{arm_name}":
                    start_idx += controller.command_dim
                else:
                    end_idx = start_idx + controller.command_dim
                    arm_controller = controller
                    break
            # assert end_idx is not None and arm_controller is not None
            self.arm_command_start_idx[arm_name] = start_idx
            self.arm_command_end_idx[arm_name] = end_idx
            self.arm_controller[arm_name] = arm_controller
        print('self.arm_command_start_idx', self.arm_command_start_idx)
        print('self.arm_command_end_idx', self.arm_command_end_idx)

    # Copy from BaseController of OG
    def _preprocess_command(self, command, arm_name):
        """
        Clips + scales inputted @command according to self.command_input_limits and self.command_output_limits.
        If self.command_input_limits is None, then no clipping will occur. If either self.command_input_limits
        or self.command_output_limits is None, then no scaling will occur.

        Args:
            command (Array[float] or float): Inputted command vector

        Returns:
            Array[float]: Processed command vector
        """
        controller = self.arm_controller[arm_name]

        # TODO: change this to left and right arm controller
        # Make sure command is a th.tensor
        command = th.tensor([command]) if type(command) in {int, float} else command
        # We only clip and / or scale if self.command_input_limits exists
        if controller._command_input_limits is not None:
            # Clip
            command = command.clip(*controller._command_input_limits)
            if controller._command_output_limits is not None:
                # If we haven't calculated how to scale the command, do that now (once)
                if controller._command_scale_factor is None:
                    controller._command_scale_factor = abs(
                        controller._command_output_limits[1] - controller._command_output_limits[0]
                    ) / abs(controller._command_input_limits[1] - controller._command_input_limits[0])
                    controller._command_output_transform = (
                        controller._command_output_limits[1] + controller._command_output_limits[0]
                    ) / 2.0
                    controller._command_input_transform = (
                        controller._command_input_limits[1] + controller._command_input_limits[0]
                    ) / 2.0
                # Scale command
                command = (
                    command - controller._command_input_transform
                ) * controller._command_scale_factor + controller._command_output_transform

        # Return processed command
        return command

    def get_robot_eef_pose(self, name):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """
        assert name == "left" or name == "right"
        return self.get_object_pose(self.robot.eef_links[name])

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
        target_pose_dict = {}
        target_pose_dict["left"] = target_pose[:4,:]
        target_pose_dict["right"] = target_pose[4:,:]

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = th.from_numpy(np.zeros_like(self.robot.action_space.sample()))
        
        for arm_name in ["left", "right"]:
            target_pose = target_pose_dict[arm_name]

            # Compute the eef target pose in the robot frame
            target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
            dpos = target_pos - pos_relative

            dori = T.mat2quat(T.quat2mat(target_quat) @ T.quat2mat(quat_relative).T)
            dori = T.quat2axisangle(dori)

            # Compute delta pose
            err = th.cat([dpos, dori])

            # Replicate the logic from IKController
            control_dict = self.robot.get_control_dict()
            arm_controller = self.robot.controllers[f"arm_{arm_name}"]
            arm_dof_idx = arm_controller.dof_idx
            manipulation_dof_idx = arm_dof_idx

            # Assume the trunk is excluded
            # if arm_name == "left":
            #     trunk_controller = self.robot.controllers["trunk"]
            #     trunk_controller_dof_idx = trunk_controller.dof_idx
            #     manipulation_dof_idx = th.cat([arm_dof_idx, trunk_controller_dof_idx])

            j_eef = control_dict[f"eef_{arm_name}_jacobian_relative"][:, manipulation_dof_idx]
            j_eef_pinv = th.linalg.pinv(j_eef)
            delta_j = j_eef_pinv @ err
            current_joint_pos = control_dict["joint_position"][manipulation_dof_idx]
            target_joint_pos = current_joint_pos + delta_j

            # Clip values to be within the joint limits
            target_joint_pos = target_joint_pos.clamp(
                min=arm_controller._control_limits[ControlType.get_type("position")][0][manipulation_dof_idx],
                max=arm_controller._control_limits[ControlType.get_type("position")][1][manipulation_dof_idx],
            )

            arm_command = target_joint_pos
            if arm_name == "left":
                # arm_command, trunk_command = arm_command[:arm_dof_idx.shape[0]], arm_command[arm_dof_idx.shape[0]:]
                arm_command = arm_controller._reverse_preprocess_command(arm_command)
                # trunk_command = trunk_controller._reverse_preprocess_command(trunk_command)
                action[self.robot.controller_action_idx[f"arm_{arm_name}"]] = arm_command
                # action[self.robot.controller_action_idx["trunk"]] = trunk_command
            else:
                arm_command = arm_controller._reverse_preprocess_command(arm_command)
                action[self.robot.controller_action_idx[f"arm_{arm_name}"]] = arm_command

        # fill in the no operation actions for the base, camera and trunk
        for name, controller in self.robot.controllers.items():
            if name == 'base' or name == 'camera' or name == "trunk":
                partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
                action_idx = self.robot.controller_action_idx[name]
                action[action_idx] = partial_action

        # Convert to numpy tensor
        action = action.numpy()

        return action

    def target_pose_to_action_no_unprocess(self, target_pose, relative=True):
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
        target_pose_dict = {}
        target_pose_dict["left"] = target_pose[:4,:]
        target_pose_dict["right"] = target_pose[4:,:]

        # Get an all-zero action (minus gripper actuation) and set the arm command part
        # This assumes other parts of the action (e.g. base, head) are zero
        action = th.from_numpy(np.zeros_like(self.robot.action_space.sample()))
        
        for arm_name in ["left", "right"]:
            target_pose = target_pose_dict[arm_name]

            # Compute the eef target pose in the robot frame
            target_pos, target_quat = T.relative_pose_transform(*T.mat2pose(target_pose), *self.robot.get_position_orientation())

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Find the relative pose between the current eef pose and the target eef pose in the robot frame (delta pose)
            dpos = target_pos - pos_relative

            dori = T.mat2quat(T.quat2mat(target_quat) @ T.quat2mat(quat_relative).T)
            dori = T.quat2axisangle(dori)

            # Assemble the arm command and undo the preprocessing
            arm_command = th.cat([dpos, dori])

            action[self.arm_command_start_idx[arm_name]:self.arm_command_end_idx[arm_name]] = arm_command

        # fill in the no operation actions for the base and camera
        for name, controller in self.robot.controllers.items():
            if name == 'base' or name == 'camera':
                partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
                action_idx = self.robot.controller_action_idx[name]
                action[action_idx] = partial_action

        # Convert to numpy tensor
        action = action.numpy()

        return action

    def generate_action(self, target_pose):
        """
        Generate a no-op action that will keep the robot still but aim to move the arms to the saved pose targets, if possible

        Returns:
            th.tensor or None: Action array for one step for the robot to do nothing
        """
        # change to quaternion 
        # 

        # Ensure float32
        target_pose = target_pose.astype(np.float32)

        # Convert to torch tensor
        target_pose = th.from_numpy(target_pose)
        target_pose_dict = {}
        target_pose_dict["left"] = T.mat2pose(target_pose[:4,:]) # T.mat2pose(target_pose)
        target_pose_dict["right"] = T.mat2pose(target_pose[4:,:])
        
        arm_targets = {
            'arm_left': (target_pose_dict["left"][0], target_pose_dict["left"][1], 0),
            'arm_right': (target_pose_dict["right"][0], target_pose_dict["right"][1], 0),
        }

        action = th.zeros(self.robot.action_dim)
        for name, controller in self.robot.controllers.items():
            # if desired arm targets are available, generate an action that moves the arms to the saved pose targets
            if name in arm_targets:
                arm = name.replace("arm_", "")
                # change to robot base frame
                target_pos, target_orn, gripper_state = arm_targets[name] # in world fixed frame

                current_pos, current_orn = self.robot.get_eef_pose(arm)
                if target_orn is None:
                    target_orn = current_orn
                if target_pos is None:
                    target_pos = current_pos
                arm_targets[name] = (target_pos, target_orn, gripper_state)

                delta_pos = target_pos - current_pos
                # delta_orn = orientation_error(T.quat2mat(T.axisangle2quat(target_orn_axisangle)), T.quat2mat(current_orn))
                delta_orn = orientation_error(T.quat2mat(target_orn), T.quat2mat(current_orn))
                partial_action = th.cat((delta_pos, delta_orn))
            else:
                partial_action = controller.compute_no_op_action(self.robot.get_control_dict())
            action_idx = self.robot.controller_action_idx[name]
            action[action_idx] = partial_action

            # set the gripper no operation action to 0
            action[11] = 0
            action[-1] = 0
        
        # bug: change to robot base frame

        # Convert to numpy tensor
        action = action.numpy()
        print('generated action')
        print('arm left')
        print(action[5:12])
        print('arm right')
        print(action[12:19])
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

        target_pose_dict = {}

        for arm_name in ["left", "right"]:

            # TODO: here all the actions, arm_command, target_pos are for sinlge arm

            # Extract the arm command part of the action and preprocess it
            arm_command = action[self.arm_command_start_idx[arm_name]:self.arm_command_end_idx[arm_name]]
            arm_command = self._preprocess_command(arm_command, arm_name)

            # Get the current eef pose in the robot frame
            pos_relative, quat_relative = self.robot.get_relative_eef_pose(arm_name)

            # Extract the delta pose from the arm command and compute the target pose in the robot frame
            dpos = arm_command[:3]
            target_pos = pos_relative + dpos
            dori = T.quat2mat(T.axisangle2quat(arm_command[3:6]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

            # Convert the target pose to the world frame
            # TODO: double confirm the meaning of self.robot.get_position_orientation(), is this the robot fixed frame?
            target_pose = T.pose2mat(T.pose_transform(*self.robot.get_position_orientation(), target_pos, target_quat))
            target_pose = target_pose.numpy()
            
            target_pose_dict[arm_name] = target_pose

        target_pose = np.concatenate([target_pose_dict["left"], target_pose_dict["right"]], axis=0) # 8x4

        # Sanity check cycle consistency (not technically necessary)
        new_action = self.target_pose_to_action(target_pose)
        new_action[self.gripper_action_dim] = action[self.gripper_action_dim]
        # TODO: need to correct the camera pose, do not know why in the collected data the camera action is none zero
        new_action[:5] = action[:5]

        # import pdb; pdb.set_trace()
        #print('new_action', new_action)
        #print('action', action)
        #print(th.isclose(action, th.from_numpy(new_action), atol=1e-2))

        # @new_action has one less element than @action because it doesn't have the gripper actuation
        assert th.allclose(action, th.from_numpy(new_action), atol=1e-2)


        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        # gripper_action_left = action[-8]
        # gripper_action_right = action[-1:]
        # gripper_action = np.concatenate([gripper_action_left, gripper_action_right], axis=0)
        gripper_action = action[self.gripper_action_dim]
        return gripper_action

    def get_datagen_info(self, action=None):

        """
        Get information needed for data generation, at the current
        timestep of simulation. If @action is provided, it will be used to 
        compute the target eef pose for the controller, otherwise that 
        will be excluded.

        Returns:
            datagen_info (DatagenInfo instance)
        """

        # current eef pose
        eef_pose_left = self.get_robot_eef_pose('left') # 4x4
        eef_pose_right = self.get_robot_eef_pose('right') # 4x4
        # concatenate the eef poses
        eef_pose = np.concatenate([eef_pose_left, eef_pose_right], axis=0) # 8x4
        
        # object poses
        object_poses = self.get_object_poses()

        # subtask termination signals
        subtask_term_signals = self.get_subtask_term_signals()
        # print('subtask_term_signals', subtask_term_signals)

        # these must be extracted from provided action
        # Only record eef_pose that are actually achieved, not the target_pose
        target_pose = None
        gripper_action = None
        if action is not None:
            # target_pose = self.action_to_target_pose(action=action, relative=True) # 8x4
            gripper_action = self.action_to_gripper_action(action=action)

        datagen_info = DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses,
            subtask_term_signals=subtask_term_signals,
            # target_pose=target_pose,
            gripper_action=gripper_action,
        )
        return datagen_info

class MG_TestTiagoGiftbox(OmniGibsonInterfaceBimanual):
    """
    Corresponds to OG test_tiago_giftbox task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        # one relative object: giftbox
        return dict(
            gift_box=self.get_object_pose(obj=self.env.task.object_scope["gift_box.n.01_1"]),
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

        # The signal is when the left gripper touches the giftbox
        signals["touch"] = int(self.robot.states[Touching].get_value(self.env.task.object_scope["gift_box.n.01_1"]))

        return signals

class MG_TestTiagoNotebook(OmniGibsonInterfaceBimanual):
    """
    Corresponds to OG test_tiago_notebook task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        # TODO: actually two relative object, the notebook and the breakfast_table
        return dict(
            notebook=self.get_object_pose(obj=self.env.task.object_scope["notebook.n.01_1"]),
            breakfast_table=self.get_object_pose(obj=self.env.task.object_scope["breakfast_table.n.01_1"]),
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

        # The signal is when the left gripper touches the giftbox
        # TODO: why here is the left gripper, not the right gripper or anypart of the robot
        signals["touch"] = int(self.robot.states[Touching].get_value(self.env.task.object_scope["notebook.n.01_1"]))

        # TODO: need to check why the grasp signal can be -1 before 1
        signals["grasp"] = abs(int(self.robot.is_grasping(arm="right", candidate_obj=self.env.task.object_scope["notebook.n.01_1"])))

        return signals

class MG_TestTiagoCup(OmniGibsonInterfaceBimanual):
    """
    Corresponds to OG test_tiago_cup task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        # two relative objects: coffee_cup and paper_cup
        return dict(
            coffee_cup=self.get_object_pose(obj=self.env.task.object_scope["coffee_cup.n.01_1"]),
            paper_cup=self.get_object_pose(obj=self.env.task.object_scope["dixie_cup.n.01_1"]),
            breakfast_table=self.get_object_pose(obj=self.env.task.object_scope["breakfast_table.n.01_1"]),
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

        # TODO: need to check why the grasp signal can be -1 before 1
        # TODO: the current setup cannot handle arm role change
        # TODO: need to be changed
        # TRUE = 1
        # UNKNOWN = 0
        # FALSE = -1
        signals["grasp_right"] = abs(int(self.robot.is_grasping(arm="right", candidate_obj=self.env.task.object_scope["coffee_cup.n.01_1"])))
        signals["ungrasp_right"] = abs(1 - abs(int(self.robot.is_grasping(arm="right", candidate_obj=self.env.task.object_scope["coffee_cup.n.01_1"]))))

        signals["grasp_left"] = abs(int(self.robot.is_grasping(arm="left", candidate_obj=self.env.task.object_scope["dixie_cup.n.01_1"])))
        signals["ungrasp_left"] = abs(1-abs(int(self.robot.is_grasping(arm="left", candidate_obj=self.env.task.object_scope["dixie_cup.n.01_1"]))))

        return signals