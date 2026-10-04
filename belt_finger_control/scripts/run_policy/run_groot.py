import select
import sys
import numpy as np
from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG
from panda_control.robot import Robot
from panda_control.customized_franka_interface import CustomizedBeltFrankaInterface
from deoxys.utils import YamlConfig, transform_utils
import zmq
import json
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
import time
from pick_env.perceptor.vision_perception import VisionPerceptor
import pickle
from typing import Dict

def _check_quit() -> bool:
    """Non-blocking stdin check: returns True if the user typed 'q' + Enter."""
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower() == "q"
    return False


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
    eef_action:np.ndarray=None # 3
    gripper_action:np.ndarray=None # 4

def pose_4x4_to_9d(pose_matrix: np.ndarray) -> np.ndarray:
    pos_3d = pose_matrix[..., :3, 3]  # Shape: (..., 3)
    R_3x3 = pose_matrix[..., :3, :3]  # Shape: (..., 3, 3)
    col1 = R_3x3[..., :, 0]  # Shape: (..., 3)
    col2 = R_3x3[..., :, 1]  # Shape: (..., 3)
    state_9d = np.concatenate([pos_3d, col1, col2], axis=-1) # Shape: (..., 9)
    return state_9d

def move_robot_to_initial_position(robot_interface, controller_cfg_dict):
    initial_q = np.array([-1.39108,1.73569,1.73376,-2.241236,-1.62938,1.36136,1.62669, 0.08,0.,0.,0.]) # for belt gripper
    last_q = robot_interface.last_q
    # interpolate between last_q and initial_q to create a smooth trajectory
    num_steps = 20*4
    q = initial_q.copy()
    for i in range(num_steps):
        t = (i + 1) / num_steps 
        alpha = 10 * t**3 - 15 * t**4 + 6 * t**5 
        q[:-4] = (1 - alpha) * last_q + alpha * initial_q[:-4]
        robot_interface.control(controller_type="JOINT_IMPEDANCE", action=q, controller_cfg=controller_cfg_dict["JOINT_IMPEDANCE"])
    print("Robot moved to initial position.")

def warm_up_robot(robot_interface, controller_cfg_dict):
    for _ in range(2*20):
        robot_interface.control(controller_type="OSC_POSE", action=np.array([0.,0.,0.,0.,0.,0.,0.08,0.,0.,0.]), controller_cfg=controller_cfg_dict["OSC_POSE"])

def get_state(robot_interface:CustomizedBeltFrankaInterface, vision_perceptor_dict:Dict, task="shift this object much higher"):
    robot_state = RobotState()
    robot_state.time_stamp = time.time()
    robot_state.q = deepcopy(robot_interface.last_q) 
    robot_state.dq = deepcopy(robot_interface.last_dq) 
    robot_state.eef_pose = deepcopy(robot_interface.last_eef_pose)
    robot_state.gripper_width = deepcopy(robot_interface.last_gripper_q)

    vision_state, _ = vision_perceptor_dict["vision_perceptor"].last_state
    vision_state_external, _ = vision_perceptor_dict["vision_perceptor_external"].last_state
    robot_state.wrist_camera_rgb_image = vision_state.rgb
    robot_state.external_camera_rgb_image = vision_state_external.rgb

    policy_input_state = {}
    policy_input_state["observation.images.wrist"] = robot_state.wrist_camera_rgb_image
    policy_input_state["observation.images.external"] = robot_state.external_camera_rgb_image
    policy_input_state["observation.state"] = np.concatenate([robot_state.q, robot_state.gripper_width.reshape(1,), pose_4x4_to_9d(robot_state.eef_pose)], axis=0)
    policy_input_state["task"] = task
    return policy_input_state
        
    
def get_action(state, socket):
    meta = []
    payloads = []

    # Add cmd first
    meta.append({"name": "cmd", "shape": 1, "dtype": "text"})
    payloads.append("inference".encode('utf-8')) 

    # Add state
    for key in  state:
        if key == "task":
            meta.append({"name": key, "shape": 1, "dtype": "text"})
            payloads.append(state[key].encode('utf-8')) 
        else:
            arr = state[key]
            meta.append({"name": key, "shape": arr.shape, "dtype": str(arr.dtype)})
            payloads.append(arr.tobytes())
    meta_bytes  = json.dumps(meta).encode('utf-8')
    socket.send_multipart([meta_bytes, *payloads])

    reply_parts = socket.recv_multipart()

    out_meta = json.loads(reply_parts[0].decode('utf-8'))
    out_data = reply_parts[1]
    result  = np.frombuffer(out_data, dtype=out_meta['dtype']).reshape(out_meta['shape'])
    return result


def reset_policy(socket):
    meta = [{"name": "cmd", "shape": 1, "dtype": "text"}]
    payloads = ["reset".encode('utf-8')]
    meta_bytes  = json.dumps(meta).encode('utf-8')
    socket.send_multipart([meta_bytes, *payloads])
    reply_parts = socket.recv_multipart()
    print("Policy reset.")


def main():
    tasks = ["pick up the blue badminton ball and insert it into the white cup with orange inside.",
             "rotate the blue lego brick for 90 degrees.",
             "rotate the orange lego brick for 90 degrees.",
             "pick up the tea bag and insert it into the tea box.",
             "pick up the comb and rotate it vertically.",
             "Pick up the knight, place it vertically on the table."]


    # Prepare
    controller_cfg_dir = CONFIG_DIR
    interface_cfg = DEFAULT_INTERFACE_CFG
    robot_interface = CustomizedBeltFrankaInterface(
                interface_cfg, control_freq=20, use_visualizer=False,automatic_gripper_reset=False
            )
    valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE"]
    controller_cfg_dict = {k:YamlConfig(f"{controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}

    vision_perceptor = VisionPerceptor(buffer_size=5, output_img_size=(224,224),serial_number="846112072071",use_depth=False) #(640,480))#(240,240))
    vision_perceptor_external = VisionPerceptor(buffer_size=5, output_img_size=(224,224),serial_number="817412071722",use_depth=False) #(640,480))#(240,240))
    vision_perceptor.start()
    vision_perceptor_external.start()
    vision_perceptor_dict = {
        "vision_perceptor": vision_perceptor,
        "vision_perceptor_external": vision_perceptor_external
    }


    # Start ZMQ connection
    ctx    = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:6008")


    # Start roll out
    total_steps = 20*60 # inference
    for ex_id in range(20*40):
        move_robot_to_initial_position(robot_interface, controller_cfg_dict)
        reset_policy(socket)
        task_id = input(f"Please input the task id for the robot: \n{'\n'.join([f'{i}: {tk}' for i, tk in enumerate(tasks)])}")
        task = tasks[int(task_id)]
        warm_up_robot(robot_interface, controller_cfg_dict)
        print(f"Start {ex_id}-th experiments with task: {task}")
        current_step = 0
        while current_step < total_steps:
            if _check_quit():
                print("[run_groot] 'q' pressed — stopping episode.")
                break
            state = get_state(robot_interface, vision_perceptor_dict,task)
            actions = get_action(state, socket)
            for action_idx in range(actions.shape[0]): #actions.shape[0]
                action = deepcopy(actions[action_idx])
                print(f"Step {current_step+1}/{total_steps}, Action: {action}")
                robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=controller_cfg_dict["OSC_POSE"])
                current_step += 1
quit
if __name__ == "__main__":
    main()    