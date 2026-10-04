from typing import List
import numpy as np
import torch
from robot_grasping_sim.env.states import State, InternalState
import pick_env
import pick_env.env

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import tf_transformations
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy

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

def load_states(path:str,device:str="cuda"):
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

def save_actions(path:str, actions:List):
    np_actions = []
    for action in actions:
        if isinstance(action, np.ndarray):
            pass
        else:
            action = action.detach().cpu().numpy()
        np_actions.append(action)
    
    np.save(file=path, arr=np.stack(np_actions,axis=0), allow_pickle=True)

def load_actions(path:str, device:str="cuda"):
    np_actions = np.load(file=path, allow_pickle=True)
    actions = []
    for action in np_actions:
        actions.append(torch.tensor(action).to(device=device))
    return actions


def save_internal_states(path:str, internal_states:List[InternalState]):
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
    for state in internal_states:
        np_state = {}
        np_state["target_object_poses"] = to_np(state.target_object_poses)
        np_state["target_object_name_onehot"] = to_np(state.target_object_name_onehot)
        np_states.append(np_state)
    np.save(file=path, arr=np_states, allow_pickle=True)


def load_internal_states(path:str,device:str="cuda"):
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
        torch_state = InternalState()
        torch_state.target_object_poses = from_np(state["target_object_poses"])
        torch_state.target_object_name_onehot = from_np(state["target_object_name_onehot"])
        torch_states.append(torch_state)
    return torch_states


class RosDebugger(Node):
    def __init__(self, env) -> None:
        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        super().__init__('ros_debugger')
        
        self.env = env
        self.link_dict = {}
        self.marker_dict = {}
        self.force_dict = {}
        self.image_dict = {}
        self.pc_dict = {}

        self.br = TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)
        
        # Create QoS profile for latched topics (transient_local durability)
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(Marker, 'mesh_debug', latched_qos)
        self.bridge = CvBridge()

        time.sleep(1.0)
        self.update_static_pose("robot", "world", np.linalg.inv(self.env.base_tf))
        time.sleep(1.0)
        if self.env.pose_perceptor is not None:
            self.update_static_pose("robot", "external_camera", self.env.pose_perceptor.tf_robot_2_external_camera)
        time.sleep(1.0)

    def update_static_pose(self, parent:str, child:str, mat:np.ndarray):
        transform = self.matrix_to_transform(mat, parent, child)

        # Publish the transform
        self.static_br.sendTransform(transform)
        pass
    
    def update_pose(self, name:str, mat:np.ndarray):
        transform = self.matrix_to_transform(mat, "world", name)

        # Publish the transform
        self.br.sendTransform(transform)
        pass

    def update_mesh(self, name:str, path:str, mat:np.ndarray):
        if name not in self.marker_dict:
            self.marker_dict[name] = len(self.marker_dict)

        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = name
        marker.id = self.marker_dict[name]
        
        marker.type = Marker.MESH_RESOURCE
        
        marker.mesh_resource = "file://" + path
        
        marker.pose = Pose()
        marker.pose.position.x = mat[0, 3]
        marker.pose.position.y = mat[1, 3]
        marker.pose.position.z = mat[2, 3]
        
        quat = tf_transformations.quaternion_from_matrix(mat)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 0.8  # Set alpha to 1.0 for full opacity
        
        self.marker_pub.publish(marker)
        pass

    def update_force(self, name:str, position:np.ndarray, force:np.ndarray, scale:float = 0.1):
        if name not in self.marker_dict:
            self.marker_dict[name] = len(self.marker_dict)
            latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.force_dict[name] = self.create_publisher(Float32, f'force_debug/{name}', latched_qos)
        marker = Marker()
        
        marker.header.frame_id = "world" 
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = name
        marker.id = self.marker_dict[name]
        
        marker.type = Marker.ARROW
        
        # Set the scale of the arrow (x: shaft diameter, y: head diameter, z: head length)
        marker.scale.x = scale  # Shaft diameter
        marker.scale.y = 2 * scale  # Head diameter
        marker.scale.z = 3 * scale  # Head length
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8 
        
        start_point = Point()
        start_point.x = position[0]
        start_point.y = position[1]
        start_point.z = position[2]
        
        end_point = Point()
        end_point.x = position[0] + force[0]
        end_point.y = position[1] + force[1]
        end_point.z = position[2] + force[2]
        
        marker.points = [start_point, end_point]
        
        self.marker_pub.publish(marker)
        float_msg = Float32()
        float_msg.data = float(np.linalg.norm(force))
        self.force_dict[name].publish(float_msg)
        pass

    def update_image(self, name:str, image:np.ndarray):
        if name not in self.image_dict:
            latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.image_dict[name] = self.create_publisher(Image, f'image_debug/{name}', latched_qos)
        
        # Determine the encoding based on the image channels
        if len(image.shape) == 2:
            encoding = "passthrough"  # Grayscale or depth
        elif len(image.shape) == 3 and image.shape[2] == 3:
            encoding = "rgb8"  # RGB image
        else:
            self.get_logger().error("Unsupported image format")
            return

        # Convert the image to a ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding)
        
        # Set the frame_id and timestamp
        ros_image.header.frame_id = name
        ros_image.header.stamp = self.get_clock().now().to_msg()
        
        self.image_dict[name].publish(ros_image)
        pass

    def update_pc(self, name:str, pc:np.ndarray):
        pass

    def matrix_to_transform(self, matrix:np.ndarray, parent_frame, child_frame):    
        """Convert a 4x4 transformation matrix to a TransformStamped message."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        translation = tf_transformations.translation_from_matrix(matrix)
        rotation = tf_transformations.quaternion_from_matrix(matrix)
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]

        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]
        return transform