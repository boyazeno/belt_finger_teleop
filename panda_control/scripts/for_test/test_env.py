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
import rclpy
# from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations
import os

config_file_path = os.path.join(os.environ.get("RL_METHODS_ROOT", "rl_methods"), "config", "mpo_batch.yaml") # rl_methods is not part of this repository
device = f'cuda:0'
dtype=torch.float64

# rclpy.init()
# node = rclpy.create_node("test_env")
# left_pub = node.create_publisher(Marker, "left", 10)
# l_pub = node.create_publisher(Vector3, "l", 10)
# right_pub = node.create_publisher(Marker, "right", 10)
# r_pub = node.create_publisher(Vector3, "r", 10)

#* Init ENV
from robot_grasping_sim.utils.io import load_config
actor_config = load_config(path=config_file_path)
env = pick_env.env.Env(actor_config=actor_config, log_path="test_env_log.txt")

def vec_2_msg(force:np.ndarray, origin:np.ndarray, idx:int, color=(1.0,0.0,0.0)):
    msg = Marker()
    msg.action = Marker.ADD
    msg.id = idx
    msg.header.frame_id = "map"
    msg.type = Marker.ARROW
    msg.scale.x = 0.1
    msg.scale.y = 0.1
    msg.scale.z = 0.1

    msg.color.r = color[0]
    msg.color.g = color[1]
    msg.color.b = color[2]
    msg.color.a = 1.0

    start_point = Point()
    start_point.x = origin[0]
    start_point.y = origin[1]
    start_point.z = origin[2]

    end_point = Point()
    end_point.x = origin[0] + force[0]
    end_point.y = origin[1] + force[1]
    end_point.z = origin[2] + force[2]

    msg.points.append(start_point)
    msg.points.append(end_point)
    return msg

def matrix_to_transform(matrix):
    """Convert a 4x4 transformation matrix to a TransformStamped message."""
    transform = TransformStamped()

    # Extract translation
    translation = tf_transformations.translation_from_matrix(matrix)

    # Extract rotation (as a quaternion)
    rotation = tf_transformations.quaternion_from_matrix(matrix)

    # Fill TransformStamped
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = translation[2]

    transform.transform.rotation.x = rotation[0]
    transform.transform.rotation.y = rotation[1]
    transform.transform.rotation.z = rotation[2]
    transform.transform.rotation.w = rotation[3]

    return transform

#* Do loop#
# br = TransformBroadcaster(node)

time.sleep(3.0)
while rclpy.ok():
    time.sleep(0.05)
    s,internal_s = env.get_state()
    left_force = s.sum_forces[0].cpu().numpy()
    right_force = s.sum_forces[1].cpu().numpy()
    matrix = internal_s.target_object_poses[0].cpu().numpy()
    delta = 1e-2
    print("left:",s.sum_forces[0].norm(dim=-1)>delta)
    print("right:",s.sum_forces[1].norm(dim=-1)>delta)
    # transform = matrix_to_transform(matrix)
    # transform.header.stamp = node.get_clock().now().to_msg()
    # transform.header.frame_id = "map"
    # child_frame = "target_object"

    # Publish the transform
    # br.sendTransform(transform)

    # left_pub.publish(vec_2_msg(left_force[0], origin=(0.,0.,0.), idx=0, color=(1.0,0.0,0.0)))
    # right_pub.publish(vec_2_msg(right_force[0], origin=(0.,0.,0.), idx=1, color=(0.0,1.0,0.0)))
    # l_pub.publish(Vector3(x=left_force[0][0],y=left_force[0][1],z=left_force[0][2]))
    # r_pub.publish(Vector3(x=right_force[0][0],y=right_force[0][1],z=right_force[0][2]))
# env.lift(distance=0.05, opening_width=0.04)