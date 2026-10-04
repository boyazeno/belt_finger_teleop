from pick_env.perceptor.pose_perception import PosePerceptor
from o3d_vis import Visualizer
import numpy as np
import time
import matplotlib.pyplot as plt 
pose_perceptor = PosePerceptor(buffer_size=5, log_dir="./", mesh_path="data/objects/meshes/005/005.stl")

pose_perceptor.thread_get_pose_info()

# while len(pose_perceptor.buffer)==0:
#     time.sleep(0.1)
# for _ in range(100):
#     pose_state, ts_pose_state = pose_perceptor.last_state
#     target_object_pose_in_robot = pose_state.pose
#     print(target_object_pose_in_robot)
#     time.sleep(0.05)
#     # plt.imshow(pose_state.rgb)
#     # plt.show()
#     Visualizer().add_coords(coords=target_object_pose_in_robot,scale=0.1).add_coords(coords=np.eye(4),scale=1.0).show()