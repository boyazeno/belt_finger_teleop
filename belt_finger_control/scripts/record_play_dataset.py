# Record both rgbd camera, robot state, action
import numpy as np
from belt_finger_control import JoystickModeDetector
from pick_env.perceptor.vision_perception import VisionPerceptor
from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG
import pickle
from deoxys.utils import YamlConfig
from copy import deepcopy
from dataclasses import dataclass
import time
import os
from enum import Enum, auto
from typing import Callable


@dataclass
class RobotState:
    time_stamp:float=0.0
    q:np.ndarray=None
    dq:np.ndarray=None
    eef_pose:np.ndarray=None
    gripper_width:float = 0.0
    wrist_camera_rgb_image: np.ndarray=None
    wrist_camera_depth_image: np.ndarray=None
    external_camera_rgb_image: np.ndarray=None
    external_camera_depth_image: np.ndarray=None


@dataclass
class RobotAction:
    time_stamp:float=0.0
    eef_action:np.ndarray=None # 6 for x,y,z,rx,ry,rz
    gripper_action:np.ndarray=None # 4 for belt gripper, 1 for parallel gripper, 1 for panda gripper


class GripperType(Enum):
    BELT_GRIPPER = auto()
    PARALLEL_GRIPPER = auto()
    PANDA_GRIPPER = auto()
    NO_GRIPPER = auto()


class JoystickDataRecording():
    def __init__(self, save_path:str,controller_cfg_dir = CONFIG_DIR, interface_cfg = DEFAULT_INTERFACE_CFG, gripper_type:GripperType=GripperType.BELT_GRIPPER):
        super().__init__()
        self.save_path = save_path
        self.controller_cfg_dir =controller_cfg_dir
        self.detector = JoystickModeDetector()
        self.gripper_type=gripper_type

        self.robot_interface = self.get_robot_interface(self.gripper_type)(
                interface_cfg, control_freq=20,use_visualizer=False, automatic_gripper_reset=False
            )
        
        valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE"]
        self.controller_cfg_dict = {k:YamlConfig(f"{self.controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}

        self.current_flow_state = "idle" #"record_end", "record", "quit"
        self.vision_perceptor = VisionPerceptor(buffer_size=5, output_img_size=(224,224),serial_number="846112072071",use_depth=False) #(640,480))#(240,240))
        self.vision_perceptor_external = VisionPerceptor(buffer_size=5, output_img_size=(224,224),serial_number="817412071722",use_depth=False) #(640,480))#(240,240))
        self.vision_perceptor.start()
        self.vision_perceptor_external.start()
        print("Vision Perceptor started...")

    def get_robot_interface(self, gripper_type:GripperType)->Callable:
        if gripper_type == GripperType.BELT_GRIPPER:
            from panda_control.customized_franka_interface import CustomizedBeltFrankaInterface
            return CustomizedBeltFrankaInterface
        elif gripper_type == GripperType.PARALLEL_GRIPPER:
            from panda_control.customized_franka_interface import CustomizedFrankaInterface
            return CustomizedFrankaInterface
        elif gripper_type == GripperType.PANDA_GRIPPER:
            from deoxys.franka_interface import FrankaInterface
            return FrankaInterface
        elif gripper_type == GripperType.NO_GRIPPER:
            from deoxys.franka_interface import FrankaInterface
            from functools import partial
            return partial(FrankaInterface, has_gripper=False)
        else:
            raise ValueError(f"Unknown gripper type: {gripper_type}")

    def start(self):
        traj = []
        self.detector.start()
        np.set_printoptions(precision=3, suppress=True)
        traj_init_time = time.time()
        self.move_robot_to_initial_position()
        while True:
            try:
                robot_state = self.make_state(traj_init_time)
                arm_action, gripper_action = self.detector.control()
                special_button_state = self.detector.joystick_controller.get_special_botton_state()
                
                # Work Flow
                if special_button_state.button_y:
                    self.current_flow_state = "quit"
                    exit(0)
                else:
                    if  special_button_state.start and self.current_flow_state=="idle":
                        self.current_flow_state = "record"
                        self.detector.joystick_controller.vibrate(1)
                        traj = []
                        traj_init_time = time.time()
                        robot_state.time_stamp = 0.0
                    elif special_button_state.start and self.current_flow_state=="record":
                        pass
                    elif not special_button_state.start and self.current_flow_state=="record":
                        self.current_flow_state = "record_stop"
                        print("Trajectory recording end...")
                        self.detector.joystick_controller.vibrate(1,duration=0.2)
                        print(f"Saving trajectory")
                        self.save_traj(traj)
                        print("Trajectory recording end...")
                        print("Moving the robot to the initial position for the next recording...")
                        self.move_robot_to_initial_position()
                        print("Back to Idle")
                        self.current_flow_state = "idle" 

                # Action
                action = np.concatenate([arm_action, gripper_action],axis=0)
                action = self.truncate_action_according_to_gripper_type(action)
                # print(f"action is: {action}")
                action[0:3] *= 200
                action[3:6] *= 75
                # print(f"action is: {action}")
                robot_action = self.make_action(action, traj_init_time)
                self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
                if self.current_flow_state=="record":
                    traj.append([robot_state,robot_action])
            except (KeyboardInterrupt, SystemExit):
                print("\nRecording shutting down")
                break

    def move_robot_to_initial_position(self):
        initial_q = np.array([-1.38244,1.72827,1.81584,-2.3233,-1.64153,1.33047,1.66546, 0.08,0.,0.,0.]) # for belt gripper
        last_q = self.robot_interface.last_q
        # interpolate between last_q and initial_q to create a smooth trajectory
        num_steps = 20*4
        q = initial_q.copy()
        for i in range(num_steps):
            t = (i + 1) / num_steps 
            alpha = 10 * t**3 - 15 * t**4 + 6 * t**5 
            q[:-4] = (1 - alpha) * last_q + alpha * initial_q[:-4]
            self.robot_interface.control(controller_type="JOINT_IMPEDANCE", action=q, controller_cfg=self.controller_cfg_dict["JOINT_IMPEDANCE"])
        print("Robot moved to initial position.")

    def truncate_action_according_to_gripper_type(self, action:np.ndarray):
        if self.gripper_type== GripperType.BELT_GRIPPER:
            return action[:6+4]
        elif self.gripper_type== GripperType.PARALLEL_GRIPPER:
            return action[:6+1]
        elif self.gripper_type== GripperType.PANDA_GRIPPER:
            return action[:6+1]
        elif self.gripper_type== GripperType.NO_GRIPPER:
            return action[:6]
        else:
            raise ValueError(f"Unknown gripper type: {self.gripper_type}")

    def save_traj(self, traj:list):
        # Check if the save_path directory exists, if not, create it
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # Check how many .npy files are in the save_path directory
        existing_files = [f for f in os.listdir(self.save_path) if f.endswith('.npy')]
        file_count = len(existing_files)

        # Name the current file as the next number
        file_name = f"recorded_traj_{file_count}.npy"
        file_path = os.path.join(self.save_path, file_name)

        # Save the trajectory
        with open(file_path, 'wb') as f:
            pickle.dump(traj, f)
        # np.save(file_path, traj)
        pass

    def make_state(self, traj_init_time=0.0):
        robot_state = RobotState()
        robot_state.time_stamp = time.time()-traj_init_time
        robot_state.q = deepcopy(self.robot_interface.last_q) 
        robot_state.dq = deepcopy(self.robot_interface.last_dq) 
        robot_state.eef_pose = deepcopy(self.robot_interface.last_eef_pose)
        robot_state.gripper_width = deepcopy(self.robot_interface.last_gripper_q)

        vision_state, ts_visioin_state = self.vision_perceptor.last_state
        vision_state_external, ts_visioin_state = self.vision_perceptor_external.last_state
        
        robot_state.wrist_camera_rgb_image = vision_state.rgb
        robot_state.external_camera_rgb_image = vision_state_external.rgb
        # robot_state.wrist_camera_depth_image = vision_state.depth
        # robot_state.external_camera_depth_image = vision_state_external.depth
        return robot_state
    
    def make_action(self, action:np.ndarray=None, traj_init_time=0.0):
        robot_action = RobotAction()
        robot_action.time_stamp = time.time()-traj_init_time
        robot_action.eef_action = action[:6]
        robot_action.gripper_action = action[6:]
        return robot_action

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Joystick Data Recording")
    parser.add_argument("--save_path", type=str, default="data/recorded_trajectory/belt_finger_vla_benchmark/pick_comb", help="Path to save the recorded trajectory")
    parser.add_argument("--controller_cfg_dir", type=str, default=CONFIG_DIR, help="Directory containing controller configuration files")
    parser.add_argument("--interface_cfg", type=str, default=DEFAULT_INTERFACE_CFG, help="Path to the robot interface configuration file")
    args = parser.parse_args()
    joystick_data_recording = JoystickDataRecording(save_path=args.save_path, controller_cfg_dir=args.controller_cfg_dir, interface_cfg=args.interface_cfg, gripper_type=GripperType.BELT_GRIPPER)

    joystick_data_recording.start()