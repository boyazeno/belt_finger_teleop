# Record both rgbd camera, robot state, action
import numpy as np
import pickle
from dataclasses import dataclass
import os
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
import tqdm

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


ACTION_KEYS = ["dx.pos", "dy.pos", "dz.pos", "drx.pos", "dry.pos", "drz.pos", "opening.pos", "translate.pos", "roll.pos", "pitch.pos"]
JOINT_KEYS = ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos", "joint_6.pos", "joint_7.pos", "opening.pos"]
CARTESIAN_KEYS = ["x.pos", "y.pos", "z.pos", "rot_1.pos", "rot_2.pos", "rot_3.pos", "rot_4.pos", "rot_5.pos", "rot_6.pos"]
CAMERA_IDS = {0: ("external", 224, 224), 1: ("wrist", 224, 224)}

import numpy as np

def pose_4x4_to_9d(pose_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 pose matrix into a 9D vector (3D position + 6D rotation).
    Supports batched input. Input shape: (..., 4, 4)
    Output shape: (..., 9)
    """
    # 1. Extract the 3D position (translation): 4th column of the first 3 rows
    pos_3d = pose_matrix[..., :3, 3]  # Shape: (..., 3)
    
    # 2. Extract the 3x3 rotation matrix: first 3 rows and columns
    R_3x3 = pose_matrix[..., :3, :3]  # Shape: (..., 3, 3)
    
    # 3. Extract the 6D rotation: first two columns of the rotation matrix (column indices 0 and 1)
    col1 = R_3x3[..., :, 0]  # Shape: (..., 3)
    col2 = R_3x3[..., :, 1]  # Shape: (..., 3)
    
    # Concatenate into the final 9D state vector
    state_9d = np.concatenate([pos_3d, col1, col2], axis=-1) # Shape: (..., 9)
    
    return state_9d


def rot_6d_to_3x3(rot_6d: np.ndarray) -> np.ndarray:
    """
    Decode a 6D rotation back into a valid 3x3 rotation matrix
    using Gram-Schmidt orthogonalization.
    Input shape: (..., 6)
    Output shape: (..., 3, 3)
    """
    eps = 1e-8 # Avoid division by zero
    
    # Split into two 3D vectors
    x_raw = rot_6d[..., 0:3]
    y_raw = rot_6d[..., 3:6]
    
    # 1. L2-normalize the first vector to get the first column
    x_norm = np.linalg.norm(x_raw, axis=-1, keepdims=True)
    x = x_raw / (x_norm + eps)
    
    # 2. Subtract the projection onto the first vector to make the second one orthogonal to x
    dot_product = np.sum(x * y_raw, axis=-1, keepdims=True)
    y_orthogonal = y_raw - dot_product * x
    
    # 3. Normalize the orthogonalized vector to get the second column
    y_norm = np.linalg.norm(y_orthogonal, axis=-1, keepdims=True)
    y = y_orthogonal / (y_norm + eps)
    
    # 4. The third column is the cross product of the first two
    z = np.cross(x, y, axis=-1)
    
    # 5. Stack the three column vectors along the last axis into a 3x3 matrix
    # (np.stack with axis=-1 turns (..., 3) into (..., 3, 3) with the vectors as columns)
    R_3x3 = np.stack([x, y, z], axis=-1)
    
    return R_3x3

def export_lerobot_episode(
    dataset: LeRobotDataset,
    raw_dataset: list,
    task_description: str,
):
    for obs, act in raw_dataset:
        act:RobotAction
        obs:RobotState

        act_dict = {k: v for k, v in zip(ACTION_KEYS, act.eef_action.tolist() + act.gripper_action.tolist())}
        act_frame = build_dataset_frame(dataset.features, act_dict, prefix=ACTION)

        obs_dict = {k: v for k, v in zip(JOINT_KEYS, obs.q.tolist()+[obs.gripper_width])}

        obs_dict["external"] = obs.external_camera_rgb_image
        obs_dict["wrist"] = obs.wrist_camera_rgb_image

        obs_dict.update({k:v for k, v in zip(CARTESIAN_KEYS, pose_4x4_to_9d(obs.eef_pose.reshape(4,4)).tolist())})

        obs_frame = build_dataset_frame(dataset.features, obs_dict, prefix=OBS_STR)

        frame = {**act_frame, **obs_frame, "task": task_description}
        dataset.add_frame(frame)

    dataset.save_episode()

def initialize_lerobot_dataset(repo_id:str, root:str):
    action_features = {key: float for key in ACTION_KEYS}
    obs_joint = {key: float for key in JOINT_KEYS}
    obs_cartesian = {key: float for key in CARTESIAN_KEYS}
    obs_state = {**obs_joint, **obs_cartesian}
    obs_cameras = {name: (h, w, 3) for _, (name, h, w) in CAMERA_IDS.items()}

    dataset_features = combine_feature_dicts(
        hw_to_dataset_features(action_features, ACTION, use_video=True),
        hw_to_dataset_features(obs_state, OBS_STR, use_video=True),
        hw_to_dataset_features(obs_cameras, OBS_STR, use_video=True),
    )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        root=root,
        robot_type="franka_panda",
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )

class RawDataset:
    def __init__(self, data_path:str):
        self.data_path = data_path
        pass

    def __len__(self):
        files = os.listdir(self.data_path)
        return len(files) - 1 # exclude descriptions.txt

    def load_episode(self, episode_id:int):
        episode_path = os.path.join(self.data_path, f"recorded_traj_{episode_id}.npy")
        with open(episode_path, "rb") as f:
            episode_data = pickle.load(f)
        return episode_data

def mutate_task_description(task_description:str, num_mutations:int=5) -> list[str]:
    from openai import OpenAI
    client = OpenAI()
    prompt = (
        f"Rephrase the following task description {num_mutations} times. "
        "Each rephrasing must preserve exactly the same meaning and all key details. "
        "Do not add, remove, or invent any information. "
        "Return only the rephrased descriptions, one per line, without numbering or extra formatting.\n\n"
        f"Task description: {task_description}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise paraphrasing assistant. "
                    "You rephrase task descriptions using different wording while keeping all key points intact. "
                    "Never add information that was not in the original description."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    text = response.choices[0].message.content.strip()
    mutations = [line.strip() for line in text.splitlines() if line.strip()]
    return mutations[:num_mutations]


def main():
    combine_all_datasets = True
    # dataset_names = ["badminton_in_cup",
    #                  "lego_rotation_blue",
    #                  "lego_rotation_orange",  
    #                  "pick_and_insert_tea_bag",  
    #                  "pick_comb",  
    #                  "place_chess"]
    dataset_names = ["green_cup_to_blue", 
                     "green_cup_to_green", 
                     "green_cup_to_red", 
                     "white_cup_to_blue", 
                     "white_cup_to_green", 
                     "white_cup_to_red"]
    
    if combine_all_datasets == True:
        repo_id = f"belt_finger/pick_place_cups_on_circles"
        root = f"data/recorded_trajectory/ood_trajectories_normal_lerobot/{repo_id}"
        lerobot_dataset = initialize_lerobot_dataset(repo_id, root)

        for dataset_name in tqdm.tqdm(dataset_names):
            raw_dataset_path = f"data/recorded_trajectory/ood_trajectories_normal/{dataset_name}"
            task_description_path = f"{raw_dataset_path}/descriptions.txt"
            with open(task_description_path, "r") as f:
                task_description = f.read().strip()

            raw_dataset = RawDataset(raw_dataset_path)
            for episode_id in tqdm.tqdm(range(len(raw_dataset))):
                raw_episode = raw_dataset.load_episode(episode_id)
                export_lerobot_episode(lerobot_dataset, raw_episode, task_description=task_description)
        
        lerobot_dataset.finalize()

    else:
        for dataset_name in tqdm.tqdm(dataset_names):
            repo_id = f"belt_finger/{dataset_name}"
            root = f"data/recorded_trajectory/ood_trajectories_normal/{repo_id}"
            raw_dataset_path = f"data/recorded_trajectory/{dataset_name}"
            task_description_path = f"{raw_dataset_path}/descriptions.txt"
            with open(task_description_path, "r") as f:
                task_description = f.read().strip()

            raw_dataset = RawDataset(raw_dataset_path)
            lerobot_dataset = initialize_lerobot_dataset(repo_id, root)
            for episode_id in tqdm.tqdm(range(len(raw_dataset))):
                raw_episode = raw_dataset.load_episode(episode_id)
                export_lerobot_episode(lerobot_dataset, raw_episode, task_description=task_description)
            lerobot_dataset.finalize()
    
if __name__ == "__main__":
    main()