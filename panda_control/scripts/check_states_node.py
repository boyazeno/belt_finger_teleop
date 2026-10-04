from pick_env.utils import RosDebugger
import numpy as np
from pytransform3d import rotations as pr
import torch

class DummyEnv:
    def __init__(self) -> None:
        self.dtype = torch.float32
        self.base_tf = np.eye(4) 
        self.base_tf[:3,:3] = pr.matrix_from_quaternion(np.array([0.7071067811865476, -0.7071067811865476, 0, 0]))
        self.base_tf[:3,3] = np.array([0.1, 0.53, 0.0])
        self.base_tf = torch.from_numpy(self.base_tf).to(dtype=self.dtype)

        self.pose_perceptor = None

from pick_env.utils import save_states, load_states
from robot_grasping_sim.env.states import State, InternalState
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

states = load_states("state_cache.npy", "cpu")
debugger = RosDebugger(DummyEnv())

tactile_forces_l = []
tactile_forces_r = []
env_idx = 2
for s in states:
    s:State
    tactile_forces_l.append(s.sum_forces[0])
    tactile_forces_r.append(s.sum_forces[1])
    time.sleep(0.05)
    debugger.update_force(name="left",position=s.robot_tcp_poses[env_idx,:3,3].cpu().numpy(),force=s.sum_forces[0].cpu().numpy()[env_idx], scale=0.1)
    debugger.update_force(name="right",position=s.robot_tcp_poses[env_idx,:3,3].cpu().numpy(),force=s.sum_forces[1].cpu().numpy()[env_idx], scale=0.1)