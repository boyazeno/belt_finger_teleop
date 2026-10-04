import panda_control
import panda_control.robot
import numpy as np
from copy import deepcopy
import time
import numpy as np
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import torch
from typing import List
from robot_grasping_sim.env.states import State, InternalState

def save_states(path:str, states:List[State]):
    def to_np(data):
        if data is None:
            return data
        np_data = None
        if isinstance(data, list):
            np_data = [d.detach().cpu().numpy() if not isinstance(d, np.ndarray) else d for d in data]
        else:
            np_data = data.detach().cpu().numpy() if not isinstance(data, np.ndarray) else data
        return np_data
    
    np_states = []
    for state in states:
        np_state = {}
        np_state["wrist_camera_rgb_image"] = to_np(state.wrist_camera_rgb_image)
        np_state["remaining_steps"] = to_np(state.remaining_steps)
        np_state["robot_joint_positions"] = to_np(state.robot_joint_positions)
        np_state["robot_tcp_opening"] = to_np(state.robot_tcp_opening)
        np_state["robot_tcp_poses"] = to_np(state.robot_tcp_poses)
        np_state["robot_tcp_position_ranges"] = to_np(state.robot_tcp_position_ranges)
        np_state["sum_force_positions"] = to_np(state.sum_force_positions)
        np_state["sum_forces"] = to_np(state.sum_forces)
        np_state["tactile_force_image"] = to_np(state.tactile_force_image)
        np_state["wrist_camera_rgb_image"] = to_np(state.wrist_camera_rgb_image)
        np_states.append(np_state)
    np.save(file=path, arr=np_states, allow_pickle=True)

def load_states(path:str,device:str="cuda")->List[State]:
    def from_np(data):
        if data is None:
            return data
        torch_data = None
        if isinstance(data, list):
            torch_data = [torch.from_numpy(d).to(device=device) if isinstance(d, np.ndarray) else d for d in data]
        else:
            torch_data = torch.from_numpy(data).to(device=device) if isinstance(data, np.ndarray) else data
        return torch_data
    
    torch_states = []
    np_states = np.load(file=path, allow_pickle=True)
    np_states = np_states.tolist()
    for state in np_states:
        torch_state = State()
        torch_state.wrist_camera_rgb_image = from_np(state["wrist_camera_rgb_image"])
        torch_state.remaining_steps = from_np(state["remaining_steps"])
        torch_state.robot_joint_positions = from_np(state["robot_joint_positions"])
        torch_state.robot_tcp_opening = from_np(state["robot_tcp_opening"])
        torch_state.robot_tcp_poses = from_np(state["robot_tcp_poses"])
        torch_state.robot_tcp_position_ranges = from_np(state["robot_tcp_position_ranges"])
        torch_state.sum_force_positions = from_np(state["sum_force_positions"])
        torch_state.sum_forces = from_np(state["sum_forces"])
        torch_state.tactile_force_image = from_np(state["tactile_force_image"])
        torch_state.wrist_camera_rgb_image = from_np(state["wrist_camera_rgb_image"])
        torch_states.append(torch_state)
    return torch_states


def init_robot(controller_cfg_dir:str= panda_control.CONFIG_DIR, interface_cfg:str = panda_control.DEFAULT_INTERFACE_CFG):
    home_joint_positions = np.array([0.180677,-0.11398,-0.13509,-2.03183,0.0351661,1.89308,0.811476, 0.08])
    robot = panda_control.robot.Robot(controller_cfg_dir = controller_cfg_dir, interface_cfg = interface_cfg, handeye_calibration_file=panda_control.HANDEYE_CALIBRATION_FILE, home_joint_positions=home_joint_positions)
    return robot

def interpolate(start_joint_positions, end_joint_positions, t):
    n = int(t*20)
    delta_joint_positions = end_joint_positions - start_joint_positions
    
    adjusted_percentage = (-1*np.cos(np.pi/n*np.arange(n)) + 1.0)*0.5

    return [start_joint_positions + adjusted_p*delta_joint_positions for adjusted_p in adjusted_percentage]

def main():
    robot = init_robot()
    time.sleep(1.0)
    
    ##################################################
    camera_pose = robot.get_camera_pose()
    
    joint_positions = robot.get_joint_positions()
    pass
    pose =robot.get_pose()
    pass
    
    ik = robot.get_ik(tcp_pose_in_base=pose, ref_joint_positions=joint_positions[:7], default_joint_positions=np.zeros_like(7))
    pass

    target_joint_positions = deepcopy(joint_positions)
    # pass
    # target_joint_positions[-1] = 0.01
    # pass
    # robot.move_joint_to_joint(target_joint_positions=np.array([0.826282,0.152055,-0.0238605,-2.01336,0.157569,1.64278,1.38587,0.06]))
    # pass

    # target_joint_positions[-1] = 0.08
    # pass
    # target_joint_positions[1] += 0.1
    # target_joint_positions[3] += 0.1
    # pass
    # robot.move_joint_to_joint(target_joint_positions=target_joint_positions)

    # pass
    # robot.move_joint_to_cartesian(target_pose=pose, opening_width=target_joint_positions[-1], ref_joint_positions=target_joint_positions[:7])

    # pass

    # target_pose = deepcopy(pose)
    # target_pose[0,3] += 0.04
    # target_pose[2,3] += 0.04
    # target_pose = np.array([[ 0.95612771, -0.21079807, -0.20338517,  0.45305489],
    #    [-0.21098359, -0.97725363,  0.02102415,  0.01547975],
    #    [-0.20319075,  0.02280916, -0.97887306,  0.27356835],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # pass
    # robot.move_linear(target_pose=target_pose, opening_width=target_joint_positions[-1])

    pass

    ##################################################
    # robot.close_gripper()
    
    # start_joint_positions=robot.get_joint_positions()
    # end_joint_positions=deepcopy(start_joint_positions)
    # end_joint_positions[5] += 0.2
    # end_joint_positions[6] -= 0.2
    # end_joint_positions[7] = 0.08
    # pass
    # target_joint_positions_list_1 = interpolate(start_joint_positions=start_joint_positions, end_joint_positions=end_joint_positions,t=3)
    # target_joint_positions_list_2 = interpolate(start_joint_positions=end_joint_positions, end_joint_positions=start_joint_positions,t=1)
    # pass
    # for target_joint_positions in target_joint_positions_list_1:
    #     robot.move_step(target_joint_positions=target_joint_positions)
    #     time.sleep(0.01)
    # for target_joint_positions in target_joint_positions_list_2:
    #     robot.move_step(target_joint_positions=target_joint_positions)
    #     time.sleep(0.01)
    test_states = load_states(path="data/recorded_trajectory/0_0-08_0-032_duplo-3-blocks_force.npy", device="cuda")
    target_joint_positions = np.concatenate([test_states[0].robot_joint_positions.cpu().numpy()[0],test_states[0].robot_tcp_opening.cpu().numpy()[0]],axis=-1)
    robot.move_joint_to_joint(target_joint_positions)
    #################################################
    real_joint_positions = []
    record_joint_positions = []
    for state in test_states:
        target_joint_positions = np.concatenate([state.robot_joint_positions.cpu().numpy()[0],state.robot_tcp_opening.cpu().numpy()[0]],axis=-1)
        robot.move_step(target_joint_positions=target_joint_positions)
        real_joint_positions.append(robot.get_joint_positions())
        record_joint_positions.append(np.array(target_joint_positions))

    np.save("real_traj_demo",np.stack(real_joint_positions,axis=0),allow_pickle=True)
    np.save("record_traj_demo",np.stack(record_joint_positions,axis=0),allow_pickle=True)
    pass

if __name__  == "__main__":
    main()