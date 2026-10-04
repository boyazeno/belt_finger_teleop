from typing import Optional, List, Dict, Union, Tuple
from copy import deepcopy
import torch 
import numpy as np
from robot_grasping_sim.env.states import State, InternalState
import time
from pytransform3d import rotations as pr
from datetime import datetime

class Logger:
    def __init__(self, path:str) -> None:
        self.path = path
        self.buffer = []
        pass

    def info(self, msg:str):
        self.buffer.append(("[INFO]", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), msg))

    def debug(self, msg:str):
        self.buffer.append(("[DEBUG]", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), msg))

    def warn(self, msg:str):
        self.buffer.append(("[WARN]", datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), msg))
    
    def save(self):
        try:
            with open(self.path, 'w') as file:
                for level, stamp, line in self.buffer:
                    file.write(f"{level}[{stamp}] "+line + '\n')
            print(f"Successfully written logger to {self.path}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")

    def __del__(self):
        self.save()


class Env:
    def __init__(self,actor_config:str, device:str="cuda", dtype=torch.float64, log_path:str="") -> None:
        self.actor_config = actor_config
        self.dtype = dtype
        self.device = device

        #* Init robot
        self.home_joint_positions = np.array([0.151513,-0.392271,-0.0530594,-2.26665,-0.0162304,2.13577,0.823705, 0.08])
        # self.home_joint_positions = np.array([0.0361503,-0.342101,-0.0437198,-2.00251,-0.0301629,1.66727,0.807994, 0.08])
        # self.home_joint_positions = np.array([0.0866, -0.2599, -0.1151, -2.1711, -0.0589,  2.0169,  0.8845, 0.08])
        self.base_tf = np.eye(4) 
        self.base_tf[:3,:3] = pr.matrix_from_quaternion(np.array(actor_config["components"][0]["config"]["actor"]["initial_pose"]["orientation"]))
        self.base_tf[:3,3] = np.array(actor_config["components"][0]["config"]["actor"]["initial_pose"]["position"])
        self.base_tf = torch.from_numpy(self.base_tf).to(dtype=self.dtype)
        self.robot = None
        self.use_pose_perceptor = False
        self.mesh_path = ""
        self.pose_perceptor = None
        self.tactile_perceptor = None
        self.vision_perceptor = None

        self.logger = self.init_logger(log_path)
        self.init_robot()
        self.init_perception()

        self.use_ros_debugger = True
        if self.use_ros_debugger:
            from pick_env.utils import RosDebugger
            self.ros_debugger = RosDebugger(env=self)
        pass

    def init_logger(self,log_path:str):
        # logger = logging.getLogger('env_logger')
        # logger.setLevel(logging.DEBUG)
        # file_handler = logging.FileHandler(log_path)
        # formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
        logger = Logger(path=log_path)
        return logger

    def init_robot(self, controller_cfg_dir:Optional[str]=None, interface_cfg:Optional[str]=None):
        import panda_control
        controller_cfg_dir = controller_cfg_dir or panda_control.CONFIG_DIR
        interface_cfg = interface_cfg or panda_control.DEFAULT_INTERFACE_CFG
        import panda_control.robot
        self.robot = panda_control.robot.Robot(controller_cfg_dir = controller_cfg_dir, interface_cfg = interface_cfg, home_joint_positions=self.home_joint_positions)
        
        range_radius = 0.1
        self.robot_tcp_position_ranges = torch.stack([torch.tensor(self.actor_config["task"]["target_object_pose"]["position"]) + torch.tensor(self.actor_config["task"]["target_object_pose"]["position_random_min"]) - range_radius, torch.tensor(self.actor_config["task"]["target_object_pose"]["position"]) + torch.tensor(self.actor_config["task"]["target_object_pose"]["position_random_max"]) + range_radius],dim=0).type(dtype=self.dtype).to(self.device)
        time.sleep(1)
        self.logger.info(f"[CALL open_gripper]")
        self.robot.open_gripper()
        self.logger.info(f"[DONE open_gripper]")
        pass

    def init_perception(self):
        #* Camera module: capture image, downsample to 40,40,3 (need to be checked)
        #* Tactile module: capture images, extract FVF forces
        from pick_env.perceptor.tactile_perception import TactilePerceptor
        from pick_env.perceptor.vision_perception import VisionPerceptor
        buffer_size = 5
        self.tactile_perceptor = TactilePerceptor(buffer_size=buffer_size, ignore_left=False) # Set ignore_left=True if the left tactile sensor is unavailable
        self.vision_perceptor = VisionPerceptor(buffer_size=buffer_size)
        if self.use_pose_perceptor:
            from pick_env.perceptor.pose_perception import PosePerceptor
            # choose mesh path and target object name onehot
            self.mesh_path = "data/objects/meshes/004_/004_.stl" #"data/objects/meshes/002_/002_.stl" #"data/objects/meshes/005/005.stl" #  # For pose perceptor
            self.target_object_name_onehot = torch.tensor([0.0,1.0,0.0]) #torch.tensor([1.0,0.0,0.0]) #torch.tensor([0.0,0.0,1.0]) # #TODO Add mesh onehot
            self.pose_perceptor = PosePerceptor(buffer_size=5, log_dir="./", mesh_path=self.mesh_path) #TODO Add mesh path and color
        time.sleep(1)
        self.logger.info(f"[START tactile_perceptor]")
        self.tactile_perceptor.start()
        self.logger.info(f"[START vision_perceptor]")
        self.vision_perceptor.start()
        if self.use_pose_perceptor:
            self.logger.info(f"[START pose_perceptor]")
            self.pose_perceptor.start()
        pass

    def get_state(self) -> Tuple[State, InternalState]:
        t_start = time.time()
        tactile_state, ts_tactile_state = self.tactile_perceptor.last_state
        vision_state, ts_visioin_state = self.vision_perceptor.last_state
        raw_joint_positions = self.robot.get_joint_positions()
        if raw_joint_positions[-1] is None:
            raw_joint_positions[-1] = 0.08
            raw_joint_positions = raw_joint_positions.astype(np.float64)
            self.logger.warn(f"[GET STATE] no gripper position available!")
        full_robot_joint_positions = torch.from_numpy(raw_joint_positions).to(device=self.device, dtype=self.dtype)
        robot_tcp_poses = torch.from_numpy(self.robot.get_pose()).to(dtype=self.dtype)
        robot_tcp_poses = self.base_tf@robot_tcp_poses
        force_map_left = robot_tcp_poses[:3,:3]@torch.from_numpy(tactile_state.force_map_left).reshape(3,-1)
        force_map_left = force_map_left.reshape(3,40,40).permute(1,2,0)
        force_map_right = robot_tcp_poses[:3,:3]@torch.from_numpy(tactile_state.force_map_right).reshape(3,-1)
        force_map_right = force_map_right.reshape(3,40,40).permute(1,2,0)
        sum_force_left = robot_tcp_poses[:3,:3]@torch.from_numpy(tactile_state.sum_force_left)
        sum_force_right = robot_tcp_poses[:3,:3]@torch.from_numpy(tactile_state.sum_force_right)
        discret_forces_left = (robot_tcp_poses[:3,:3]@torch.from_numpy(tactile_state.discret_forces_left).unsqueeze(-1)).squeeze(-1)
        discret_forces_right = (robot_tcp_poses[:3,:3]@torch.from_numpy(tactile_state.discret_forces_right).unsqueeze(-1)).squeeze(-1)
        
        
        target_object_pose_in_world = torch.eye(4)
        target_object_name_onehot = torch.zeros(3)
        if self.use_pose_perceptor:
            pose_state, ts_pose_state = self.pose_perceptor.last_state
            target_object_pose_in_robot = pose_state.pose
            target_object_pose_in_world = (self.base_tf@torch.from_numpy(target_object_pose_in_robot))
            target_object_name_onehot = self.target_object_name_onehot
 
        state = State()
        state.robot_joint_positions = torch.tensor(full_robot_joint_positions[:7]).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        state.robot_tcp_position_ranges = self.robot_tcp_position_ranges
        state.robot_tcp_opening = full_robot_joint_positions[-1].reshape(1,1)*0.5
        state.robot_tcp_poses = (robot_tcp_poses.reshape(1,4,4)).to(device=self.device, dtype=self.dtype)
        
        state.tactile_force_image = [force_map_left.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                     force_map_right.to(device=self.device, dtype=self.dtype).unsqueeze(0)] # B*W*H*C
        state.sum_forces = [sum_force_left.to(device=self.device, dtype=self.dtype).unsqueeze(0)*40, 
                            sum_force_right.to(device=self.device, dtype=self.dtype).unsqueeze(0)*40]
        state.discret_forces = [discret_forces_left.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                discret_forces_right.to(device=self.device, dtype=self.dtype).unsqueeze(0)]

        state.remaining_steps = None # Set outside here
        state.wrist_camera_rgb_image = torch.from_numpy(vision_state.rgb).to(device=self.device, dtype=self.dtype).unsqueeze(0) # B*W*H*C, need to check if 255 or 0-1
        state.wrist_camera_depth_image = torch.from_numpy(vision_state.depth).to(device=self.device, dtype=self.dtype).unsqueeze(0) # B*W*H, need to check if 255 or 0-1
        state.wrist_camera_mask_image = torch.from_numpy(vision_state.mask).to(device=self.device, dtype=self.dtype).unsqueeze(0) # B*W*H, need to check if 255 or 0-1
        
        range_radius = 0.1
        state.target_object_position_ranges = torch.tensor([[-1*range_radius, -1*range_radius, -1*range_radius],[range_radius, range_radius, range_radius]]).to(device=self.device, dtype=self.dtype)
        tc_state = time.time()-t_start

        internal_state = InternalState()
        internal_state.target_object_poses = target_object_pose_in_world.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        internal_state.target_object_name_onehot = target_object_name_onehot.to(device=self.device, dtype=self.dtype).unsqueeze(0)

        self.logger.info(f"[GET STATE CALLED AT] {t_start} [GET STATE USE] {tc_state} [GET TACTILE AT] {ts_tactile_state} [GET VISION AT] {ts_visioin_state}")
        
        if self.use_ros_debugger:
            self.ros_debugger.update_force(name="left",position=robot_tcp_poses[:3,3].cpu().numpy(),force=sum_force_left, scale=0.1)
            self.ros_debugger.update_force(name="right",position=robot_tcp_poses[:3,3].cpu().numpy(),force=sum_force_right, scale=0.1)
            self.ros_debugger.update_pose(name="tcp", mat=robot_tcp_poses.cpu().numpy())
            self.ros_debugger.update_mesh(name="obj",path=self.mesh_path, mat=target_object_pose_in_world.cpu().numpy())
            self.ros_debugger.update_image(name="rgb", image=vision_state.rgb)
            self.ros_debugger.update_image(name="depth", image=np.abs(vision_state.depth))
            self.ros_debugger.update_image(name="mask", image=np.stack([vision_state.mask]*3,axis=-1).astype(np.uint8)*85)
            img_name = f"{time.time_ns()}.png"
            
            # if state.tactile_force_image is not None:
            #     self.ros_debugger.update_image(name="left_force_map", image=(state.tactile_force_image[0][0].norm(dim=-1).cpu().numpy()/0.08*255).astype(np.uint8))
            #     self.ros_debugger.update_image(name="right_force_map", image=(state.tactile_force_image[1][0].norm(dim=-1).cpu().numpy()/0.08*255).astype(np.uint8))
            # if state.discret_forces is not None:
            #     for df_idx in range(state.discret_forces[0].shape[1]):
            #         self.ros_debugger.update_force(f"left_discret_{df_idx}", position=robot_tcp_poses[:3,3].cpu().numpy(),force=state.discret_forces[0][0,df_idx], scale=0.1)
            #         self.ros_debugger.update_force(f"right_discret_{df_idx}", position=robot_tcp_poses[:3,3].cpu().numpy(),force=state.discret_forces[1][0,df_idx], scale=0.1)
        return state, internal_state

    def reset(self) -> State:
        #* Move back to home pose
        self.robot.move_joint_to_joint(target_joint_positions=self.home_joint_positions)

        #* Wait for put object
        input("Please setup the environment. Press any key to continue...")

        #* Select target object
        self.vision_perceptor.select_object()

        #* Restart pose estimation
        if self.use_pose_perceptor:
            self.pose_perceptor.restart()
        time.sleep(1.0)
        
        #* Do perception
        state, internal_state = self.get_state()
        state.remaining_steps = torch.tensor([0.0]).to(device=self.device, dtype=self.dtype)
        return state, internal_state

    def step(self, a_env:np.ndarray, remaining_steps:float) -> Tuple[State, InternalState]:
        self.logger.info(f"[CALL step AT] {time.time()}")
        #* Execute one step
        self.robot.move_step(target_joint_positions=a_env)
        self.logger.info(f"[DONE move_step AT] {time.time()}")

        #* Get state
        self.logger.info(f"[CALL get_state AT] {time.time()}")
        state, internal_state = self.get_state()
        state.remaining_steps = torch.tensor([remaining_steps]).to(device=self.device, dtype=self.dtype)
        self.logger.info(f"[DONE get_state AT] {time.time()}")
        return state, internal_state
    
    def post_steps(self, a_env:np.ndarray,steps:int=20):
        self.logger.info(f"[CALL post_steps AT] {time.time()}")
        for _ in range(steps):
            self.robot.move_step(target_joint_positions=a_env)
        self.logger.info(f"[DONE post_steps AT] {time.time()}")

    def lift(self, distance:float=0.05,opening_width:float=0.08):
        #* Move upwards
        pose = self.robot.get_pose()
        pose[2,3] += distance
        self.robot.move_linear(target_pose=pose,opening_width=opening_width)
        pass