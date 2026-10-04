# Record both rgbd camera, robot state, action
import numpy as np
from belt_finger_control import JoystickModeDetector
from belt_finger_control import VRModeDetector
from pick_env.perceptor.vision_perception import VisionPerceptor
import pickle
from deoxys.utils import YamlConfig
from copy import deepcopy
from dataclasses import dataclass
import time
import os
from enum import Enum, auto
from typing import Callable
import re
from pathlib import Path


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


class TrajectoryMerger:
    def __init__(self, dt: float=0.05, blend_duration: float = 0.5, epsilon: float = 1e-4):
        """
        Args:
            dt: The control loop time step in seconds (e.g., 0.02 for 50Hz).
            blend_duration: How long the transition takes in seconds.
            epsilon: Threshold to detect non-zero user commands (avoids floating point noise).
        """
        self.dt = dt
        self.blend_duration = blend_duration
        self.blend_steps = max(1, int(blend_duration / dt))
        self.epsilon = epsilon
        
        self.reset()

    def reset(self):
        """Resets the state of the merger for a new episode/run."""
        self.current_step = 0
        self.intervention_start_step = -1
        self.is_intervening = False

    def merge(self, cmd_policy, cmd_user):
        """
        Merges the policy command with the user command.
        
        Args:
            cmd_policy: [dx, dy, dz, drx, dry, drz] from the autonomous policy.
            cmd_user: [dx, dy, dz, drx, dry, drz] from the user/teleop.
            
        Returns:
            cmd_merged: The blended command array.
        """
        cmd_policy = np.array(cmd_policy, dtype=float)
        cmd_user = np.array(cmd_user, dtype=float)
        
        # 1. Detect if user has started intervening
        if not self.is_intervening:
            # Check magnitude of user command against the zero threshold
            if np.linalg.norm(cmd_user-np.array([0,0,0,0,0,0,0.08,0,0,0])) > self.epsilon:
                self.is_intervening = True
                self.intervention_start_step = self.current_step

        cmd_merged = np.zeros_like(cmd_policy)

        # 2. Compute the merged command
        if not self.is_intervening:
            # Policy is in full control
            cmd_merged = cmd_policy
        else:
            # Calculate how far along the blending process we are (0.0 to 1.0)
            steps_since_intervention = self.current_step - self.intervention_start_step
            progress = min(1.0, steps_since_intervention / self.blend_steps)
            
            # Use Smoothstep interpolation for a continuous velocity profile
            # Formula: 3*t^2 - 2*t^3 (Start and end derivatives are zero)
            alpha = progress * progress * (3.0 - 2.0 * progress)
            
            # Blend the commands
            cmd_merged = (1.0 - alpha) * cmd_policy + alpha * cmd_user

        # 3. Advance internal clock
        self.current_step += 1
        
        return cmd_merged

    def get_merged_step_idx(self):
        """
        Returns the step index where user intervention was first detected.
        Returns -1 if no intervention has happened yet.
        """
        return self.intervention_start_step


class TrajectoryReplayer:
    def __init__(self, root_dir:str):
        """
        Args:
            dt: The control loop time step in seconds (e.g., 0.02 for 50Hz).
        """
        self.root_dir = root_dir
        self.current_step = 0
        self.trajectory = []
        self.total_steps = 0

    def extract_trajectory_files(self, recursive: bool = True):
        """
        Extracts two types of trajectory files from a given folder.
        
        Args:
            root_folder (str): The path to the directory to search.
            recursive (bool): If True, searches all subdirectories as well.
            
        Returns:
            tuple: (base_files, extended_files) containing lists of file paths.
        """
        root = Path(self.root_dir)
        
        # Regex explanations:
        # ^ and $ ensure we match the whole string.
        # \d+ matches one or more digits (your idx and ids).
        # (?:\.\w+)? optionally matches a file extension like .json, .h5, etc.
        base_pattern = re.compile(r"^recorded_traj_\d+\.npy$")
        extended_pattern = re.compile(r"^recorded_traj_\d+_extend_\d+\.npy$")
        
        base_files = []
        extended_files = []
        
        # Check if root exists
        if not root.exists() or not root.is_dir():
            print(f"Error: Directory '{self.root_dir}' does not exist.")
            return base_files, extended_files

        # Choose between recursive search (rglob) or shallow search (iterdir)
        file_iterator = root.rglob("*") if recursive else root.iterdir()
        
        for file_path in file_iterator:
            if file_path.is_file():
                # Check the extended pattern first (to avoid partial matches, though ^$ prevents this)
                if extended_pattern.match(file_path.name):
                    extended_files.append(str(file_path))
                elif base_pattern.match(file_path.name):
                    base_files.append(str(file_path))
                    
        return base_files, extended_files

    def load_trajectory(self, idx:int=-1)->str:
        """
        Loads a trajectory for replay.
        
        Args:
            trajectory: List of RobotAction objects or similar structure.
        """
        base_files, extended_files = self.extract_trajectory_files(recursive=False)
        trajectory_name = None
        file_path = None
        try:
            if idx == -1:
                # Load the last trajectory
                if not base_files:
                    raise FileNotFoundError(f"No main trajectory files found in {self.root_dir}")
                file_path = max(base_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                trajectory_name = file_path.split('/')[-1].split('.')[0]
            else:
                trajectory_name = f"recorded_traj_{idx}"
                file_path = os.path.join(self.root_dir, f"{trajectory_name}.npy")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Trajectory file {file_path} does not exist.")

            with open(file_path, 'rb') as f:
                trajectory = pickle.load(f)
                
            self.trajectory = trajectory
            self.total_steps = len(trajectory)
            self.current_step = 0
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            self.trajectory = []
            self.total_steps = 0
            self.current_step = 0
        return trajectory_name

    def get_action_n_state(self):
        """
        Returns the next action and robot state in the trajectory based on the current step.
        
        Returns:
            Tuple[np.ndarray, np.ndarray] or (None, None) if the end of the trajectory is reached.
        """
        if self.current_step < self.total_steps:
            robot_state, robot_action = self.trajectory[self.current_step]
            robot_action:RobotAction
            action = np.concat([robot_action.eef_action, robot_action.gripper_action])
            robot_state:RobotState
            joint_positions = np.zeros(7+4) # q+gripper
            joint_positions[:7] = robot_state.q[:7]
            joint_positions[7] = robot_state.gripper_width
            print("Trajectory replaying...")
            return action, joint_positions
        else:
            print("Trajectory replay done!")
            return np.zeros(6+4), None

    def advance_step(self)->bool:
        """Advances the current step index by one. Return True if successful, False if no next step available."""
        self.current_step += 1


    def reset(self):
        """Resets the replayer to the beginning of the trajectory."""
        self.current_step = 0


class JoystickDataRecording():
    def __init__(self, save_path:str,controller_cfg_dir:str, interface_cfg:str, gripper_type:GripperType=GripperType.BELT_GRIPPER):
        super().__init__()
        self.save_path = save_path
        self.controller_cfg_dir =controller_cfg_dir
        self.detector = VRModeDetector()
        self.gripper_type=gripper_type

        self.robot_interface = self.get_robot_interface(self.gripper_type)(
                interface_cfg, control_freq=20,use_visualizer=False, automatic_gripper_reset=False
            )
        
        valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE"]
        self.controller_cfg_dict = {k:YamlConfig(f"{self.controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}

        # For trajectory replay and recording
        self.trajectory_merger = TrajectoryMerger(blend_duration=0.5, epsilon=1e-4)
        self.trajectory_merger.reset()
        self.trajectory_replayer = TrajectoryReplayer(root_dir=self.save_path)
        self.is_in_replay_mode = False # for replay the last recorded trajectory with extra intervation
        self.main_traj_name = None # for replay the last recorded trajectory with extra intervation

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
                arm_action, gripper_action = self.detector.control(robot_state)
                special_button_state = self.detector.joystick_controller.get_special_botton_state()
                
                # Work Flow
                if special_button_state.button_y:
                    self.current_flow_state = "quit"
                    exit(0)

                # if special_button_state.tactile: # use tactile button for starting replay
                if self.current_flow_state=="idle":
                    self.is_in_replay_mode = special_button_state.tactile
                    print(f"Replay mode {'ON' if self.is_in_replay_mode else 'OFF'}!")

                # Work Flow
                if special_button_state.button_y:
                    self.current_flow_state = "quit"
                    exit(0)
                else:
                    if  special_button_state.start and self.current_flow_state=="idle":
                        self.current_flow_state = "record"
                        self.detector.joystick_controller.vibrate(1)
                        if self.is_in_replay_mode:
                            self.trajectory_replayer.reset()
                            self.trajectory_merger.reset()
                            self.main_traj_name = self.trajectory_replayer.load_trajectory(idx=-1)
                            if self.main_traj_name is None:
                                print("No trajectory found for replay, switching to record mode.")
                                self.is_in_replay_mode = False
                            print(f"Trajectory replay mode ON, loaded trajectory: {self.main_traj_name}")
                            action_policy, joint_positions = self.trajectory_replayer.get_action_n_state()
                            self.move_robot_to_position(joint_positions)
                            time.sleep(1.0) # Wait for the camera to stabilize after moving the robot
                        else:
                            print(f"Trajectory replay mode OFF, recording new trajectory")
                        print("Trajectory recording start...")

                        traj = []
                        traj_init_time = time.time()
                        robot_state.time_stamp = 0.0
                    elif special_button_state.start and self.current_flow_state=="record":
                        print("Trajectory recording...")
                        pass
                    elif not special_button_state.start and self.current_flow_state=="record":
                        self.current_flow_state = "record_stop"
                        print("Trajectory recording end...")
                        self.detector.joystick_controller.vibrate(1,duration=0.2)
                        print(f"Saving trajectory")
                        if self.is_in_replay_mode:
                            blend_in_steps = self.trajectory_merger.get_merged_step_idx()
                            self.save_traj_blend(self.main_traj_name, traj, blend_in_steps)
                        else:
                            self.save_traj(traj)
                        print("Trajectory recording end...")
                        print("Moving the robot to the initial position for the next recording...")
                        if self.is_in_replay_mode:
                            self.trajectory_replayer.reset()
                            self.trajectory_merger.reset()
                        self.move_robot_to_initial_position()
                        print("Back to Idle")
                        self.current_flow_state = "idle" 

                # Action
                arm_action[0:3] *= 120
                arm_action[3:6] *= 5
                action = np.concatenate([arm_action, gripper_action],axis=0)
                action = self.truncate_action_according_to_gripper_type(action)

                if self.is_in_replay_mode and self.current_flow_state=="record":
                    action_policy, joint_positions = self.trajectory_replayer.get_action_n_state()
                    self.trajectory_replayer.advance_step()
                    if action_policy is not None:
                        action_policy = self.truncate_action_according_to_gripper_type(action_policy)
                        action = self.trajectory_merger.merge(action_policy, action)
                        
                print(f"Action: {action}")
                # print(f"action is: {action}")
                robot_action = self.make_action(action, traj_init_time)
                self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
                if self.current_flow_state=="record":
                    traj.append([robot_state,robot_action])


            except (KeyboardInterrupt, SystemExit):
                print("\nRecording shutting down")
                break

    def move_robot_to_position(self, joint_positions: np.ndarray):
        initial_q = np.array(joint_positions) # for belt gripper
        last_q = self.robot_interface.last_q
        # interpolate between last_q and initial_q to create a smooth trajectory
        num_steps = 20*4
        q = initial_q.copy()
        for i in range(num_steps):
            t = (i + 1) / num_steps 
            alpha = 10 * t**3 - 15 * t**4 + 6 * t**5 
            q[:-4] = (1 - alpha) * last_q + alpha * initial_q[:-4]
            self.robot_interface.control(controller_type="JOINT_IMPEDANCE", action=q, controller_cfg=self.controller_cfg_dict["JOINT_IMPEDANCE"])
        print(f"Robot moved to position: {joint_positions}")

    def move_robot_to_initial_position(self):
        initial_q = np.array([-1.38244,1.72827,1.81584,-2.3233,-1.64153,1.33047,1.66546, 0.08,0.,0.,0.]) # for belt gripper
        self.move_robot_to_position(initial_q)

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
        existing_files = self.trajectory_replayer.extract_trajectory_files(recursive=False)[0]  # Get base files
        file_count = len(existing_files)

        # Name the current file as the next number
        file_name = f"recorded_traj_{file_count}.npy"
        file_path = os.path.join(self.save_path, file_name)

        # Save the trajectory
        with open(file_path, 'wb') as f:
            pickle.dump(traj, f)
        # np.save(file_path, traj)
        pass

    def save_traj_blend(self, main_traj_name:str, traj:list, blend_in_steps:int):
        # Check if the save_path directory exists, if not, create it
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # Check how many .npy files are in the save_path directory
        existing_files = self.trajectory_replayer.extract_trajectory_files(recursive=False)[1]  # Get extended files
        file_count = len(existing_files)

        # Name the current file as the next number
        file_name = f"{main_traj_name}_extend_{file_count}.npy"
        file_path = os.path.join(self.save_path, file_name)

        # Save the trajectory
        with open(file_path, 'wb') as f:
            pickle.dump({"traj": traj, "blend_in_steps": blend_in_steps}, f)
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
    parser.add_argument("--save_path", type=str, default="/root/ws/home_ws/data/recorded_trajectory/test", help="Path to save the recorded trajectory")
    parser.add_argument("--controller_cfg_dir", type=str, default="/root/ws/3rd_party_libs/panda_control/config", help="Directory containing controller configuration files")
    parser.add_argument("--interface_cfg", type=str, default="/root/ws/3rd_party_libs/panda_control/config/charmander.yml", help="Path to the robot interface configuration file")
    args = parser.parse_args()
    joystick_data_recording = JoystickDataRecording(save_path=args.save_path, controller_cfg_dir=args.controller_cfg_dir, interface_cfg=args.interface_cfg, gripper_type=GripperType.BELT_GRIPPER)

    joystick_data_recording.start()