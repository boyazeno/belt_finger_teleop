#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformBroadcaster
import numpy as np
from geometry_msgs.msg import TransformStamped
import yaml
import tf_transformations
from panda_control import HANDEYE_CALIBRATION_FILE

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

class TFPublisherNode(Node):
    def __init__(self, matrix):
        super().__init__('world_to_camera_tf_publisher')
        
        # Create a tf broadcaster
        self.br = TransformBroadcaster(self)
        self.matrix = matrix
        
        # Create timer at 100Hz
        self.timer = self.create_timer(0.01, self.publish_transform_callback)
        
    def publish_transform_callback(self):
        # Create the transform message
        transform = matrix_to_transform(self.matrix)

        # Set the header information
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "panda_EE"
        child_frame = "camera_color_optical_frame"
        transform.child_frame_id = child_frame
        # transform.header.frame_id = "camera_color_optical_frame"
        # child_frame = "board"#"camera_color_optical_frame"

        # Publish the transform
        self.br.sendTransform(transform)

def publish_transform(matrix):
    rclpy.init()
    node = TFPublisherNode(matrix)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    with open(HANDEYE_CALIBRATION_FILE, 'r') as file:
        config = yaml.safe_load(file)["transform_camera_2_ee"]
        transformation_matrix = np.array(config).reshape(4,4).T
#         transformation_matrix = np.array(
#  [[ -0.46835,   0.865765,  -0.176348, -0.0863496],
#  [-0.857964,  -0.397961,   0.324847,  0.0670063],
#  [ 0.211062,   0.303443,    0.92918,   0.497714],
#  [        0,          0,          0,          1],]

        # ).reshape(4,4)

    print(transformation_matrix)
    # Publish the transform
    publish_transform(transformation_matrix)
