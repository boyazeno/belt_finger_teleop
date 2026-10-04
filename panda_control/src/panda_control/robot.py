from typing import Optional, List, Dict, Union
from copy import deepcopy
import torch 
import numpy as np
import panda_ikfast
import pytorch3d.transforms as ptransforms
from pytransform3d import rotations as pr

from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG
from panda_control.customized_franka_interface import CustomizedFrankaInterface
from panda_control.customized_franka_interface import CustomizedBeltFrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.log_utils import get_deoxys_example_logger
import time
import yaml
import logging

class Robot:
    def __init__(self, controller_cfg_dir = CONFIG_DIR, interface_cfg = DEFAULT_INTERFACE_CFG, handeye_calibration_file:Optional[str]=None, home_joint_positions:np.ndarray=None, use_belt_gripper:bool=False) -> None:
        self.controller_cfg_dir = controller_cfg_dir
        self.interface_cfg = interface_cfg
        self.handeye_calibration_file = handeye_calibration_file
        self.use_belt_gripper = use_belt_gripper
        
        self._pre_calculate_tf_matrix(offset=0.210) # Approximate TCP -> link8 offset in meters (not yet precisely measured)
        self.transform_tcp_2_camera = None
        if self.handeye_calibration_file is not None:
            self._load_camera_pose()
        self.logger = get_deoxys_example_logger()
        if self.use_belt_gripper:
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self.robot_interface = CustomizedBeltFrankaInterface(
                interface_cfg, use_visualizer=False,automatic_gripper_reset=False
            )
        else:
            self.robot_interface = CustomizedFrankaInterface(
                interface_cfg, use_visualizer=False,automatic_gripper_reset=False
            )


        valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE"]
        self.controller_cfg_dict = {k:YamlConfig(f"{self.controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}

        self.home_joint_positions = home_joint_positions
        pass

    def _load_camera_pose(self):
        with open(self.handeye_calibration_file) as f:
            d=yaml.safe_load(f)
            self.transform_tcp_2_camera = np.array(d["transform_camera_2_ee"]).reshape(4,4).T
            print(f"Load handeye transformation:\n{self.transform_tcp_2_camera}")

    def move_linear(self, target_pose:np.ndarray, opening_width:float, duration:float = 3.0):
        current_pose = self.get_pose()

        # Interpolation
        n = int(duration*20) #! 20Hz
        ts = (-1*np.cos(np.pi/n*np.arange(1,n+1)) + 1.0)*0.5
        delta_p = target_pose[:3,3] - current_pose[:3,3]
        target_poses = []

        current_quat = pr.quaternion_from_matrix(current_pose[:3,:3],strict_check=False)
        target_quat = pr.quaternion_from_matrix(target_pose[:3,:3],strict_check=False)
        for t in ts:
            quat = pr.quaternion_slerp(start=current_quat, end=target_quat, t=t, shortest_path=True)
            pose = np.eye(4)
            pose[:3,3] = delta_p*t + current_pose[:3,3]
            pose[:3,:3] = pr.matrix_from_quaternion(quat)
            target_poses.append(pose)
        
        # Execute
        last_idx = len(target_poses)-1
        for idx, tp in enumerate(target_poses): 
            controller_type = "OSC_POSE"
            controller_cfg = self.controller_cfg_dict[controller_type]

            target_pos = tp[:3,3:]
            target_quat = pr.quaternion_from_matrix(tp[:3,:3], strict_check=False)
            target_quat = pr.quaternion_xyzw_from_wxyz(target_quat)

            while True:
                last_eef_pose = self.robot_interface.last_eef_pose
                dpose = np.linalg.inv(last_eef_pose)@tp
                if (np.max(np.abs(dpose[:3,3]))< 1e-2): #! need to add rotation as well, currently only position
                    break
                
                current_pose = self.robot_interface.last_eef_pose
                current_pos = current_pose[:3, 3:]
                current_rot = current_pose[:3, :3]
                current_quat = transform_utils.mat2quat(current_rot)
                if np.dot(target_quat, current_quat) < 0.0:
                    current_quat = -current_quat
                quat_diff = transform_utils.quat_distance(target_quat, current_quat)
                axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
                action_pos = (target_pos - current_pos).flatten() * 10
                action_axis_angle = axis_angle_diff.flatten() * 3
                action_pos = np.clip(action_pos, -1.0, 1.0)
                action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

                action = action_pos.tolist() + action_axis_angle.tolist() + [opening_width]
                self.logger.info(f"Axis angle action {action}")
                self.logger.info(f"delta xyz: {dpose[:3,3]}")
                self.robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                if idx != last_idx:
                    break
        pass

    def move_joint_to_cartesian(self, target_pose:np.ndarray, opening_width:float, ref_joint_positions:np.ndarray=None):
        success,target_joint_positions = self.get_ik(tcp_pose_in_base=target_pose, ref_joint_positions=ref_joint_positions, default_joint_positions=ref_joint_positions)
        if not success:
            input("IK calculation failed, moving back to the home position, press any key to continue...")
        target_joint_positions = np.concatenate([target_joint_positions,np.array([opening_width])])
        self.move_joint_to_joint(target_joint_positions=target_joint_positions)
        pass

    def move_joint_to_joint(self, target_joint_positions:np.ndarray):
        action = target_joint_positions
        controller_type = "JOINT_POSITION"
        controller_cfg = self.controller_cfg_dict[controller_type]
        last_t = time.time()
        while True:
            if len(self.robot_interface._state_buffer) > 0:
                self.logger.info(f"Current Robot joint: {np.round(self.robot_interface.last_q, 3)}")
                self.logger.info(f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}")
                cur_t = time.time()
                self.logger.info(f"dt: {cur_t-last_t}")
                last_t = cur_t
                if (
                    np.max(
                        np.abs(
                            np.array(self.robot_interface._state_buffer[-1].q)
                            - target_joint_positions[:-1]
                        )
                    )
                    < 5e-3 and np.abs(self.robot_interface._gripper_state_buffer[-1].width - target_joint_positions[-1]) < 1e-3
                ):
                    break
            self.robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        print("[Motion Done]")
        pass

    def move_step(self, target_joint_positions:np.ndarray):
        controller_type = "JOINT_IMPEDANCE"
        controller_cfg = self.controller_cfg_dict[controller_type]
        action = target_joint_positions.tolist()
        print(f"action: {action}")
        self.logger.info(f"Current Robot joint: {np.round(self.robot_interface.last_q, 3)}")
        self.logger.info(f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}")
        self.robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        pass

    def get_joint_positions(self):
        return np.array(self.robot_interface.last_q.tolist()+[self.robot_interface.last_gripper_q])

    def get_desired_joint_positions(self):
        # Note: the gripper entry is the measured gripper position, not the desired one.
        return np.array(self.robot_interface.last_q_d.tolist()+[self.robot_interface.last_gripper_q])
        

    def get_pose(self):
        eef_pose = deepcopy(self.robot_interface.last_eef_pose)
        return eef_pose # Note: not yet verified whether this pose refers to the TCP or to link8

    def get_camera_pose(self):
        if self.transform_tcp_2_camera is None:
            raise ValueError("No hand-eye calibration loaded; pass handeye_calibration_file to Robot.")
        eef_pose = deepcopy(self.robot_interface.last_eef_pose)
        return eef_pose@self.transform_tcp_2_camera

    def get_jacobian(self):
        raise NotImplementedError()
        pass

    def control_gripper(self, gripper_action:np.ndarray):
        self.robot_interface.gripper_control(action=gripper_action) # [opening,transy, rotz, rotx]
        pass

    def open_gripper(self):
        if self.use_belt_gripper:
            self.robot_interface.gripper_control(action=[0.08,0.0,0.0,0.0])
        else:
            self.robot_interface.gripper_control(action=0.08)
        pass

    def close_gripper(self,positions:float=0.0,timeout:float=2.0):
        if self.use_belt_gripper:
            self.robot_interface.gripper_control(action=[positions,0.0,0.0,0.0])
        else:
            self.robot_interface.gripper_control(action=positions)
        t_start = time.time()
        while abs(self.get_joint_positions()[-1]-positions)>2e-3 or time.time()-t_start<timeout:

            time.sleep(0.05)
        pass

    def is_grasped(self):
        raise NotImplementedError()
        pass


    def _pre_calculate_tf_matrix(self, offset:float=0.107):
        self.tf_tcp_to_link8 = np.eye(4)
        self.tf_tcp_to_link8[:3,:3] = pr.matrix_from_quaternion(np.array([0.924, -0.000, -0.000, 0.383]))
        self.tf_tcp_to_link8[2, 3] = -1*offset

        self.tf_link8_to_tcp = np.eye(4)
        self.tf_link8_to_tcp[:3,:3] =pr.matrix_from_quaternion(np.array([0.924, 0.000, 0.000, -0.383]))
        self.tf_link8_to_tcp[2, 3] = offset

    def get_ik(self, tcp_pose_in_base:np.ndarray, ref_joint_positions:np.ndarray, default_joint_positions:np.ndarray)->np.ndarray:
        if len(tcp_pose_in_base.shape) > 2: # Change to N*4*4
            tcp_pose_in_base = tcp_pose_in_base.squeeze(0)

        ref_joint_positions = ref_joint_positions.tolist()
        # convert to link8
        link8_target_in_base = tcp_pose_in_base @ self.tf_tcp_to_link8
        success, joint_positions = panda_ikfast.get_ik(target=link8_target_in_base, ref=ref_joint_positions)
        if success:
            return success, np.array(joint_positions)
        else:
            if default_joint_positions is not None:
                return success, default_joint_positions
            else:
                return success, self.home_joint_positions
            

    def __del__(self):
        self.robot_interface.close()