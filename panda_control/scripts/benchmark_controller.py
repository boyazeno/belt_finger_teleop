from typing import Optional, List, Dict, Union, Callable
from copy import deepcopy
import torch 
import numpy as np
import pick_env
import pick_env.env
import panda_control
from rl_methods.mpo.mpo import MPO
from robot_grasping_sim.env.states import State, InternalState
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
from pick_env.utils import save_states, load_states
from pick_env.utils import save_states, load_actions
import itertools
import os


def get_config():
    joint_kp_kd_list = [
                        # ([160., 160., 160., 160., 160., 160., 100.],
                        #  [11., 11., 11., 11., 11, 11.0, 8.0]),
                        # ([180., 180., 180., 180., 180., 180., 120.],
                        #  [11., 11., 11., 11., 11, 11.0, 8.0]),
                        #  ([180., 180., 180., 180., 180., 180., 120.],
                        #  [11., 11., 11., 8., 8., 8.0, 6.0]),
                         ([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 100.0],
                         [40.0, 40.0, 40.0, 40.0, 25.0, 11.0, 8.0]),
                        ]
    traj_interpolator_type_list = ["MIN_JERK_JOINT_POSITION",] #"LINEAR_JOINT_POSITION", 
    time_fraction_list = [0.1,0.3]

    for config_pair in itertools.product(joint_kp_kd_list,traj_interpolator_type_list,time_fraction_list):
        yield {"joint_kp":config_pair[0][0], "joint_kd":config_pair[0][1], "traj_interpolator_type":config_pair[1], "time_fraction":config_pair[2]}
    return config

def set_config(env:pick_env.env.Env, joint_kp, joint_kd, traj_interpolator_type, time_fraction):
    env.robot.controller_cfg_dict["JOINT_IMPEDANCE"].joint_kp = joint_kp
    env.robot.controller_cfg_dict["JOINT_IMPEDANCE"].joint_kd = joint_kd
    env.robot.controller_cfg_dict["JOINT_IMPEDANCE"].traj_interpolator_cfg.traj_interpolator_type = traj_interpolator_type
    env.robot.controller_cfg_dict["JOINT_IMPEDANCE"].traj_interpolator_cfg.time_fraction = time_fraction
    print(f"#"*30)
    print(f"Update config:\n")
    print(f"joint_kp: {joint_kp}")
    print(f"joint_kd: {joint_kd}")
    print(f"traj_interpolator_type: {traj_interpolator_type}")
    print(f"time_fraction: {time_fraction}")

def load_traj(path:str, device:str):
    actions = load_actions(path, device)
    traj = [action.cpu().numpy().squeeze() for action in actions]
    return traj

def save_config(config:Dict, path:str):
    import yaml 
    with open(path, 'w') as file:
        data  = {}
        data["traj_interpolator_cfg"] = {}
        data["traj_interpolator_cfg"]["traj_interpolator_type"] = config["traj_interpolator_type"]
        data["traj_interpolator_cfg"]["time_fraction"] = config["time_fraction"]
        data["joint_kp"] = config["joint_kp"]
        data["joint_kd"] = config["joint_kd"]
        yaml.dump(config, file)

def evaluate(states:List[State], traj:List[np.ndarray]):
    result = 0.0
    for state, action in zip(states, traj):
        result += np.linalg.norm(state.robot_joint_positions.cpu().numpy().squeeze() - action[:7])
    return result/len(states)

log_file_path = "data/recorded_trajectory/bench_mark_trajectory"
config_file_path = os.path.join(os.environ.get("RL_METHODS_ROOT", "rl_methods"), "config", "mpo_batch.yaml") # rl_methods is not part of this repository
path = "data/recorded_trajectory/real_trajectory/20240820-192552"
device = f'cuda:0'
dtype=torch.float64


#* Init ENV
from robot_grasping_sim.utils.io import load_config
actor_config = load_config(path=config_file_path)
env = pick_env.env.Env(actor_config=actor_config, log_path=f"{log_file_path}/{__name__}_log.txt")


traj = load_traj(path=path+"/actions.npy", device=device)
total_steps = len(traj)-1
time.sleep(3.0)
for config_id, config in enumerate(get_config()):
    env.robot.move_joint_to_joint(target_joint_positions=traj[0])
    set_config(env, **config)
    time.sleep(2.0)
    states = []
    for idx, tj in enumerate(traj[1:]):
        remaining_steps = (total_steps-idx-1.)/total_steps
        s, internal_s = env.step(a_env=tj, remaining_steps=remaining_steps)
        states.append(s)
    result = evaluate(states, traj[1:])
    save_states(states=states, path=f"{log_file_path}/{config_id}.npy")
    save_config(config, path=f"{log_file_path}/{config_id}-{result}.yaml")
    print(f"Config {config_id} result: {result}")
    time.sleep(1.0)