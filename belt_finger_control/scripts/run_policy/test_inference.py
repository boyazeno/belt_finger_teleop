import zmq
import json
import numpy as np
from dataclasses import dataclass
import time

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

# test reset
def reset_policy(socket):
    meta = [{"name": "cmd", "shape": 1, "dtype": "text"}]
    payloads = ["reset".encode('utf-8')]
    meta_bytes  = json.dumps(meta).encode('utf-8')
    socket.send_multipart([meta_bytes, *payloads])
    reply_parts = socket.recv_multipart()
    print("Policy reset.")

ctx    = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:6008")

reset_policy(socket)

policy_input_state = {}
policy_input_state["observation.images.wrist"] = np.zeros((224,224,3), dtype=np.uint8)
policy_input_state["observation.images.external"] = np.zeros((224,224,3), dtype=np.uint8)
policy_input_state["observation.state"] = np.concatenate([np.zeros(7), np.zeros(1), np.zeros(9)], axis=0)
policy_input_state["task"] = "shift this object much higher"

action = get_action(policy_input_state, socket)
t_start = time.time()
for _ in range(100):
    action = get_action(policy_input_state, socket)
print(f"Average inference time per step: {(time.time() - t_start)/100:.4f} seconds")
print("Received action from policy:", action.shape, action.dtype, action[:,0])