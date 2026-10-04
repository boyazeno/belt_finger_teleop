from typing import Optional, List, Dict, Union, Callable
from copy import deepcopy
import torch 
import numpy as np
import pick_env
import pick_env.env
import panda_control
from rl_methods.mpo.mpo import MPO
from robot_grasping_sim.env.states import State, InternalState
from pick_env.utils import save_states, save_actions, save_internal_states
from robot_grasping_sim.utils.motion_filter import MotionFilterEMA
import time
import os

def to_s_actor(s_env:State, internal_s_env:InternalState, postprocess:Callable, inference_step_num:int):
    res = postprocess(s=s_env,
                      internal_s=internal_s_env,
                      r=torch.zeros([]),
                      d=torch.zeros([]),
                      valid_tuple_mask=torch.zeros([]),
                      sequence_lengths=inference_step_num,
                      )
    s_actor = res[0]
    return s_actor

def to_a_env(a_actor:torch.Tensor, preprocess:Callable, last_target_joint_positions:np.ndarray)->np.ndarray:
    a_env, _ = preprocess(model_a=a_actor, model_d=torch.zeros([]))
    a_env[-1] *= 2.0 # Finger joint is symmtric 
    target_joint_positions = last_target_joint_positions + a_env.cpu().numpy()
    target_joint_positions = np.clip(target_joint_positions, a_min=[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973,0.0], a_max=[2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973,0.08])
    return target_joint_positions

def init_actor(actor_config:str, dtype=torch.float64, device:str=f'cuda:0'):
    #! MPO only
    from rl_methods.mpo.mpo import MPO
    actor = MPO(config=actor_config, dtype=dtype, device=device)
    if actor_config["history"]["resume"]:
        actor.load_check_point(checkpoint_path=actor_config["history"]["checkpoint_path"])
    
    use_mock = True
    if use_mock:
        from pick_env.mock.mock_actor import MockActor
        actor = MockActor(real_actor=actor)
    return actor, actor_config

def main():
    config_file_path = os.path.join(os.environ.get("RL_METHODS_ROOT", "rl_methods"), "config", "mpo_batch.yaml") # rl_methods is not part of this repository
    path = f"data/recorded_trajectory/real_trajectory/{time.strftime('%Y%m%d-%H%M%S')}"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    
    device = f'cuda:0'
    dtype=torch.float64

    #* Init ENV
    from robot_grasping_sim.utils.io import load_config
    actor_config = load_config(path=config_file_path)
    env = pick_env.env.Env(actor_config=actor_config, log_path=path+"/log.txt")
    
    #* Init actor
    actor, actor_config = init_actor(actor_config=actor_config, dtype=dtype, device=device)
    preprocess = actor.get_register_func(func_name="preprocess")
    postprocess = actor.get_register_func(func_name="postprocess")
    
    while input("q for quit otherwise continue") != "q":
        inference_step_num = 0
        #* Do loop
        time.sleep(3.0)
        s_env, internal_s_env = env.reset()
        inference_step_num += 1
        last_target_joint_positions = np.concatenate([s_env.robot_joint_positions.cpu().numpy(),s_env.robot_tcp_opening.cpu().numpy()*2], axis=-1)
        s_actor = to_s_actor(s_env=s_env, internal_s_env=internal_s_env, postprocess=postprocess, inference_step_num=inference_step_num)

        if actor_config["train"]["use_successive_frames"]:
            successive_frame_buffer = torch.concat([s_actor.clone() for _ in range(actor_config["train"]["num_successive_frames"])], dim=-1)

        state_list = []
        internal_state_list = []
        action_list = []

        #* motion filters
        
        motion_filter = None
        motion_filter = MotionFilterEMA(w=0.98)
        motion_filter.set_initial_motion(initial_motion=last_target_joint_positions)

        while inference_step_num <= actor.max_sequence_length: # Iterate until all env terminated at least once
            inference_step_num += 1
            if actor_config["train"]["use_successive_frames"]:
                successive_frame_buffer[:,:-actor.dim_s] = successive_frame_buffer[:,actor.dim_s:].clone()
                successive_frame_buffer[:,-actor.dim_s:] = s_actor
                s_actor = successive_frame_buffer                        

            env.logger.info(f"[CALL policy inference AT] {time.time()}")
            a_actor = actor.inference(s=s_actor)
            env.logger.info(f"[DONE policy inference AT] {time.time()}")
            env.logger.info(f"[START postprocess AT] {time.time()}")
            a_env = to_a_env(a_actor=a_actor, preprocess=preprocess, last_target_joint_positions=last_target_joint_positions)
            last_target_joint_positions = a_env

            if motion_filter:
                a_env = motion_filter.step(motion=a_env)

            remaining_steps = (actor.max_sequence_length - inference_step_num)/(actor.max_sequence_length)
            env.logger.info(f"[DONE postprocess AT] {time.time()}")
            s_env, internal_s_env = env.step(a_env=a_env.squeeze(0),remaining_steps=remaining_steps)
            state_list.append(deepcopy(s_env))
            internal_state_list.append(deepcopy(internal_s_env))
            action_list.append(deepcopy(a_env))
            s_actor = to_s_actor(s_env=s_env, internal_s_env=internal_s_env, postprocess=postprocess, inference_step_num=inference_step_num)
        
        # env.post_steps(a_env=a_env.squeeze(0),steps=20)
        # env.robot.close_gripper(positions=0.03)
        print("Episode finished!")
        env.vision_perceptor.stop_mask()
        save_states(path=path+"/states",states=state_list)
        save_internal_states(path=path+"/internal_states",internal_states=internal_state_list)
        save_actions(path=path+"/actions",actions=action_list)
        print(f"Saved states and actions to {path}")
        while pressed_key := input("Press l key to lift, q to quit..."):
            if pressed_key == "l":
                # env.lift(distance=0.10, opening_width=last_target_joint_positions[0,-1])
                lift_joint_position = deepcopy(env.home_joint_positions)
                lift_joint_position[-1] = last_target_joint_positions[0,-1]
                env.robot.move_joint_to_joint(target_joint_positions=lift_joint_position)
                break
            elif pressed_key == "q":
                return 
        env.robot.open_gripper()

if __name__=="__main__":
    main()